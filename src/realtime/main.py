import os
import asyncio
import aiohttp
import base64
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
import pyaudio
from pynput import keyboard
import logging
import json
import pygame
import tempfile
import threading
import queue
import wave
import traceback
from queue import Queue

# Load environment variables first
load_dotenv()

# Configure logging based on LOG_LEVEL environment variable
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PyAudio configuration - Azure OpenAI Realtime API requires 24kHz
PYAUDIO_FORMAT = pyaudio.paInt16
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = 24000  # 24kHz sample rate required by Azure OpenAI Realtime API
PYAUDIO_CHUNK = 1024  # Number of frames per buffer

class VoiceChatClientError(Exception):
    """Custom exception for client errors."""
    pass


class AzureOpenAIRealtimeClient:
    def __init__(self, config: dict, yaml_config: dict = None):
        """Initialize the Azure OpenAI Realtime client.
        
        Args:
            config: Environment configuration (Azure endpoints, etc.)
            yaml_config: YAML configuration for VAD modes and parameters (optional)
        """
        self.config = config
        self.yaml_config = yaml_config or {}
        self._websocket_url = self._construct_websocket_url()

        self._credential = DefaultAzureCredential()
        self._token_provider = get_bearer_token_provider(
            self._credential, "https://cognitiveservices.azure.com/.default"
        )

        self._pyaudio_instance = None
        self._pyaudio_stream = None
        self._keyboard_listener = None

        self._is_recording = False
        self._audio_chunk_queue = asyncio.Queue()  # For streaming audio chunks
        self._space_released_event = asyncio.Event()
        self._quit_event = asyncio.Event()
        self._ws = None # WebSocket connection

        # For managing async tasks
        self._tasks = []

        # VAD mode configuration
        self._vad_mode = self.yaml_config.get('turn_detection', {}).get('type')
        self._is_push_to_talk = self._vad_mode is None

        # Initialize pygame mixer for audio playback - match Azure OpenAI's 24kHz output
        # Use specific settings for Windows compatibility
        try:
            pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=512)
            pygame.mixer.init()
            # Set mixer volume to maximum
            pygame.mixer.set_num_channels(8)  # Allow multiple concurrent sounds
            logger.debug("Pygame mixer initialized successfully")
        except pygame.error as e:
            logger.error(f"Failed to initialize pygame mixer: {e}")
            # Try fallback initialization
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)
                logger.debug("Pygame mixer initialized with fallback settings")
            except pygame.error as e2:
                logger.error(f"Pygame mixer fallback also failed: {e2}")
        
        # Audio playback queue and thread
        self._audio_playback_queue = Queue()
        self._audio_playback_thread = None
        self._stop_audio_playback = threading.Event()

        # Current response text buffer
        self._current_response_text = ""

        # Display configuration info
        if self.yaml_config.get('scenario'):
            self._display_message("status_info", f"Mode: {self.yaml_config['scenario']}")
        if self.yaml_config.get('description'):
            self._display_message("status_info", f"Description: {self.yaml_config['description']}")
        self._display_message("status_info", f"VAD Mode: {self._vad_mode or 'Push-to-Talk'}")
        self._display_message("status_info", f"Voice: alloy")
        self._display_message("status_info", f"Temperature: 0.8")

    def _construct_websocket_url(self) -> str:
        endpoint = self.config.get("AZURE_OPENAI_ENDPOINT")
        deployment = self.config.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        api_version = self.config.get("OPENAI_API_VERSION")

        if not all([endpoint, deployment, api_version]):
            raise VoiceChatClientError("Missing Azure OpenAI configuration variables.")

        hostname = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
        return f"wss://{hostname}/openai/realtime?api-version={api_version}&deployment={deployment}"

    async def _get_auth_headers(self) -> dict:
        access_token = await self._token_provider()
        return {"Authorization": f"Bearer {access_token}"}

    def _display_message(self, m_type: str, content: str = "", end: str = '\n', flush: bool = False):
        """Handles all console output with emojis."""
        prefixes = {
            "status_connect": "🔌 CONNECT:", "status_ok": "✅ SUCCESS:", "status_info": "ℹ️  INFO:",
            "status_warn": "⚠️  WARNING:", "status_error": "❌ ERROR:", "mic_ready": "🎙️  MIC:",
            "mic_rec_start": "🔴 REC:", "mic_rec_stop": "⏹️  STOP:", "audio_send": "📤 SEND:",
            "audio_sent": "📬 SENT:", "ai_thinking": "🤔 AI:", "ai_resp_text": "💬 RESP:",
            "ai_resp_audio": "🔊 AUDIO:", "ai_transcript": "📝 TRANS:", "user_action": "👉 ACTION:",
            "system_event": "⚙️  SYS:", "quit": "👋 BYE:"
        }
        prefix = prefixes.get(m_type, "➡️  INFO:")
        # Determine if flushing should be forced for non-delta status messages
        force_flush = flush or m_type not in ["ai_resp_text_delta", "ai_transcript_delta"]

        if m_type == "ai_resp_text_delta": # No prefix for delta text - clean text output
            print(content, end=end, flush=flush) # Deltas are flushed based on explicit param
        elif m_type == "ai_transcript_delta":
            print(f"{prefix} {content}", end=end, flush=flush)
        else:
            print(f"{prefix} {content}", end=end, flush=force_flush)

    def _audio_playback_worker(self):
        """Background thread worker for audio playback."""
        while not self._stop_audio_playback.is_set():
            try:
                # Get audio data from queue with timeout
                audio_data = self._audio_playback_queue.get(timeout=0.1)
                if audio_data is None:  # Poison pill to stop thread
                    break
                
                logger.debug(f"Processing audio chunk for playback: {len(audio_data)} bytes")
                
                # Create a proper WAV file with headers
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_file_path = temp_file.name
                        
                    # Create a WAV file with proper headers
                    # Azure OpenAI sends PCM audio at 24kHz, 16-bit, mono
                    sample_rate = 24000
                    channels = 1
                    sample_width = 2  # 16-bit = 2 bytes
                    
                    with wave.open(temp_file_path, 'wb') as wav_file:
                        wav_file.setnchannels(channels)
                        wav_file.setsampwidth(sample_width)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_data)
                    
                    logger.debug(f"Created WAV file: {temp_file_path}")
                    
                    # Load and play the audio with volume boost
                    sound = pygame.mixer.Sound(temp_file_path)
                    
                    # Set volume to maximum
                    sound.set_volume(1.0)
                    
                    # Play the sound
                    channel = sound.play()
                    logger.debug(f"Started audio playback on channel: {channel}")
                    
                    # Wait for the sound to finish playing
                    while pygame.mixer.get_busy():
                        if self._stop_audio_playback.is_set():
                            pygame.mixer.stop()
                            break
                        pygame.time.wait(10)
                    
                    logger.debug("Audio playback completed")
                        
                except (pygame.error, wave.Error) as e:
                    logger.error(f"Error playing audio: {e}")
                    # Try alternative playback method
                    self._try_alternative_playback(audio_data)
                except Exception as e:
                    logger.error(f"Unexpected error in audio playback: {e}")
                finally:
                    # Clean up temporary file
                    if temp_file_path:
                        try:
                            os.unlink(temp_file_path)
                        except OSError:
                            pass
                        
                self._audio_playback_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                if not self._stop_audio_playback.is_set():
                    logger.error(f"Error in audio playback worker: {e}")
                continue

    def _try_alternative_playback(self, audio_data: bytes):
        """Try alternative audio playback using system commands."""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file_path = temp_file.name
                
            # Write WAV data
            with wave.open(temp_file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz
                wav_file.writeframes(audio_data)
            
            # Try playing with system command
            import subprocess
            if os.name == 'nt':  # Windows
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{temp_file_path}").PlaySync()'], 
                             check=False, capture_output=True)
            
            # Clean up
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
        except Exception as e:
            logger.error(f"Alternative playback also failed: {e}")

    def _play_audio(self, audio_data: bytes):
        """Queue audio data for playback."""
        if not self._stop_audio_playback.is_set():
            logger.debug(f"Queuing audio data for playback: {len(audio_data)} bytes")
            self._audio_playback_queue.put(audio_data)
            # Removed audio queue message display for cleaner output

    def _start_audio_playback_thread(self):
        """Start the audio playback thread."""
        if self._audio_playback_thread is None or not self._audio_playback_thread.is_alive():
            self._stop_audio_playback.clear()
            self._audio_playback_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
            self._audio_playback_thread.start()
            self._display_message("system_event", "Audio playback thread started.")

    def _stop_audio_playback_thread(self):
        """Stop the audio playback thread."""
        if self._audio_playback_thread and self._audio_playback_thread.is_alive():
            self._stop_audio_playback.set()
            self._audio_playback_queue.put(None)  # Poison pill
            self._audio_playback_thread.join(timeout=2.0)
            self._display_message("system_event", "Audio playback thread stopped.")


    def _on_keyboard_press(self, key):
        if key == keyboard.Key.space:
            if self._is_push_to_talk and not self._is_recording:
                self._display_message("mic_rec_start", "Recording...")
                self._is_recording = True
                # Clear any existing audio buffer when starting new recording
                if self._ws and not self._ws.closed:
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(self._clear_audio_buffer())
                    except Exception as e:
                        logger.error(f"Error scheduling audio buffer clear: {e}")
            elif not self._is_push_to_talk:
                # In VAD modes, spacebar can be used to manually trigger response
                if self._ws and not self._ws.closed:
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(self._manual_response_trigger())
                    except Exception as e:
                        logger.error(f"Error scheduling manual response trigger: {e}")
        elif hasattr(key, 'char') and key.char == 'q':
            self._display_message("quit", "Quit signal received. Stopping...")
            if not self._quit_event.is_set():
                self._quit_event.set()
            if self._keyboard_listener: # Attempt to stop listener from its own thread
                 return False

    async def _manual_response_trigger(self):
        """Manually trigger a response in VAD modes."""
        if self._ws and not self._ws.closed:
            try:
                # Commit the audio buffer and trigger response
                commit_event = {"type": "input_audio_buffer.commit"}
                await self._ws.send_json(commit_event)
                
                response_event = {"type": "response.create"}
                await self._ws.send_json(response_event)
                
                self._display_message("ai_thinking", "Manual response triggered...")
                logger.debug("Manually triggered response in VAD mode")
            except Exception as e:
                logger.error(f"Error triggering manual response: {e}")

    async def _clear_audio_buffer(self):
        """Clear the audio input buffer to start fresh recording."""
        if self._ws and not self._ws.closed:
            try:
                clear_event = {"type": "input_audio_buffer.clear"}
                await self._ws.send_json(clear_event)
                logger.debug("Cleared input audio buffer")
            except Exception as e:
                logger.error(f"Error clearing audio buffer: {e}")


    def _on_keyboard_release(self, key):
        if key == keyboard.Key.space and self._is_push_to_talk:
            if self._is_recording:
                self._display_message("mic_rec_stop", "Recording stopped.")
                self._is_recording = False
                self._space_released_event.set()

    async def _audio_capture_loop(self):
        self._display_message("system_event", "Audio capture loop started.")
        
        # In VAD modes, we start recording immediately
        if not self._is_push_to_talk:
            self._display_message("mic_ready", "Voice activity detection enabled - speak naturally...")
            self._is_recording = True
        
        while not self._quit_event.is_set():
            should_capture = self._is_recording
            
            # In push-to-talk mode, only capture when spacebar is pressed
            # In VAD modes, capture continuously
            if should_capture and self._pyaudio_stream:
                try:
                    # Capture audio chunk (≈ 43ms at 24kHz for 1024 samples)
                    data = await asyncio.to_thread(self._pyaudio_stream.read, PYAUDIO_CHUNK, exception_on_overflow=False)
                    
                    # Stream the audio chunk immediately to the API
                    await self._audio_chunk_queue.put(data)
                    
                except IOError as e:
                    if e.errno == pyaudio.paInputOverflowed: # type: ignore
                        self._display_message("status_warn", "Microphone input overflowed. Skipping frame.")
                    else:
                        self._display_message("status_error", f"PyAudio read error: {e}")
                        self._quit_event.set() # Critical error, stop client
                        break
                except Exception as e:
                    self._display_message("status_error", f"Unexpected error in audio capture: {e}")
                    self._quit_event.set()
                    break
            else:
                await asyncio.sleep(0.01)
        self._display_message("system_event", "Audio capture loop finished.")


    async def _stream_audio_loop(self):
        """Stream audio chunks to OpenAI using input_audio_buffer.append events."""
        self._display_message("system_event", "Audio streaming loop started.")
        
        while not self._quit_event.is_set() and self._ws and not self._ws.closed:
            try:
                # Get audio chunk from queue (blocks until available)
                audio_chunk = await asyncio.wait_for(self._audio_chunk_queue.get(), timeout=0.1)
                
                if self._quit_event.is_set() or not self._ws or self._ws.closed:
                    break
                
                # Encode audio chunk as base64
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                # Send audio chunk using input_audio_buffer.append
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": audio_base64
                }
                
                await self._ws.send_json(audio_event)
                logger.debug(f"Streamed {len(audio_chunk)} bytes of audio data")
                
                self._audio_chunk_queue.task_done()
                
            except asyncio.TimeoutError:
                continue  # No audio chunk available, continue loop
            except Exception as e:
                if not self._quit_event.is_set():
                    logger.error(f"Error in audio streaming loop: {e}")
                break
                
        self._display_message("system_event", "Audio streaming loop finished.")

    async def _handle_recording_completion(self):
        """Handle the completion of recording and request AI response (push-to-talk mode only)."""
        if not self._is_push_to_talk:
            # In VAD modes, recording completion is handled automatically by the server
            self._display_message("system_event", "Recording completion handler skipped (VAD mode).")
            return
            
        self._display_message("system_event", "Recording completion handler started (push-to-talk mode).")
        
        while not self._quit_event.is_set() and self._ws and not self._ws.closed:
            await self._space_released_event.wait()
            self._space_released_event.clear()

            if self._quit_event.is_set() or not self._ws or self._ws.closed: 
                break

            # Commit the audio buffer and request response
            self._display_message("audio_send", "Finalizing audio input...")
            
            try:
                # Commit the audio buffer
                commit_event = {"type": "input_audio_buffer.commit"}
                await self._ws.send_json(commit_event)
                logger.debug("Sent input_audio_buffer.commit")
                
                # Request AI response
                response_create = {"type": "response.create"}
                await self._ws.send_json(response_create)
                logger.debug("Sent response.create")
                
                self._display_message("ai_thinking", "Audio submitted, waiting for AI response...")
                
            except Exception as e:
                logger.error(f"Error handling recording completion: {e}")
                self._display_message("status_error", f"Error submitting audio: {e}")
                
        self._display_message("system_event", "Recording completion handler finished.")

    async def _receive_server_messages_loop(self):
        """Receive server messages loop with debug logging."""
        self._display_message("system_event", "Message receiving loop started.")
        try:
            async for msg in self._ws: # type: ignore
                if self._quit_event.is_set(): 
                    break

                logger.debug(f"WebSocket message type: {msg.type}")
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = msg.json()
                        logger.debug(f"Received JSON message: {json.dumps(data, indent=2)}")
                        
                        msg_type = data.get("type")
                        if msg_type == "response.text.delta":
                            # Accumulate text for clean display
                            delta_text = data.get("delta", "")
                            self._current_response_text += delta_text
                            self._display_message("ai_resp_text_delta", delta_text, end="", flush=True)
                        elif msg_type == "response.text.done":
                            logger.debug("AI text response completed")
                            self._display_message("ai_resp_text_delta", "", end='\n')
                            self._current_response_text = ""  # Reset for next response
                        elif msg_type == "response.audio.delta":
                            # Handle audio data chunks - decode base64 and queue for playback
                            audio_b64 = data.get("delta", "")
                            if audio_b64:
                                try:
                                    audio_bytes = base64.b64decode(audio_b64)
                                    self._play_audio(audio_bytes)
                                    logger.debug(f"Queued {len(audio_bytes)} bytes of audio for playback")
                                except Exception as e:
                                    logger.warning(f"Failed to decode audio delta: {e}")
                        elif msg_type == "response.audio.done":
                            logger.debug("AI audio response completed")
                            # Removed audio playback complete message for cleaner output
                        elif msg_type == "response.audio_transcript.delta":
                            # Display transcript text cleanly without prefix
                            transcript_text = data.get("delta", "")
                            self._display_message("ai_resp_text_delta", transcript_text, end="", flush=True)
                        elif msg_type == "response.audio_transcript.done":
                            logger.debug("AI audio transcript completed")
                            self._display_message("ai_resp_text_delta", " [Done]", end='\n')
                        elif msg_type == "response.done":
                            logger.debug("AI response completed")
                            self._display_message("status_ok", "--- End of AI Response ---")
                        elif msg_type == "error":
                            error_msg = data.get('message', 'Unknown error')
                            error_code = data.get('code', 'No code')
                            error_details = data.get('details', 'No details')
                            logger.error(f"Server error - Code: {error_code}, Message: {error_msg}, Details: {error_details}")
                            logger.debug(f"Full error response: {json.dumps(data, indent=2)}")
                            self._display_message("status_error", f"Server error: {error_msg}")
                            self._quit_event.set()
                            break
                        elif msg_type == "session.created":
                            logger.debug(f"Session created: {data.get('session', {})}")
                        elif msg_type == "session.updated":
                            logger.debug(f"Session updated: {data.get('session', {})}")
                        elif msg_type == "conversation.item.created":
                            logger.debug(f"Conversation item created: {data.get('item', {})}")
                        elif msg_type == "response.created":
                            logger.debug(f"Response created: {data.get('response', {})}")
                        else:
                            logger.debug(f"Unhandled message type: {msg_type}")
                    except ValueError as e:
                        logger.warning(f"Failed to parse JSON message: {e}")
                        logger.debug(f"Raw message data: {msg.data}")
                        self._display_message("status_warn", f"Received non-JSON text: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    logger.debug(f"Received {len(msg.data)} bytes of binary data")
                    # Handle binary audio data - directly queue for playback
                    self._play_audio(msg.data)
                    self._display_message("ai_resp_audio", f"Playing {len(msg.data)} bytes of binary audio data.")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection error: {self._ws.exception()}")
                    self._display_message("status_error", f"WebSocket connection error: {self._ws.exception()}")
                    self._quit_event.set()
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket connection closed by server")
                    self._display_message("status_info", "WebSocket connection closed by server.")
                    self._quit_event.set()
                    break
        except Exception as e:
            if not self._quit_event.is_set():
                logger.exception("Error in message receiving loop")
                self._display_message("status_error", f"Error in message receiving loop: {e}")
            self._quit_event.set()
        finally:
            self._display_message("system_event", "Message receiving loop finished.")
            if not self._quit_event.is_set():
                self._quit_event.set()


    def _construct_session_config(self) -> dict:
        """Construct session configuration based on YAML config.
        
        Returns:
            dict: Session configuration for OpenAI Realtime API
        """
        session_config = {
            "modalities": ["text", "audio"],
            "instructions": "You are a helpful AI assistant. Respond naturally and conversationally.",
            "voice": "alloy",
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                "model": "whisper-1"
            },
            "temperature": 0.8
        }
        
        # Configure turn detection based on YAML config
        turn_detection_config = self.yaml_config.get('turn_detection', {})
        
        if turn_detection_config.get('type') is None:
            # Push-to-talk mode - no automatic turn detection
            session_config["turn_detection"] = None
            logger.debug("Configured for push-to-talk mode (no automatic turn detection)")
            
        elif turn_detection_config.get('type') == 'server_vad':
            # Server VAD configuration
            vad_config = {
                "type": "server_vad"
            }
            
            # Add optional server VAD parameters if specified
            if 'threshold' in turn_detection_config:
                vad_config["threshold"] = turn_detection_config['threshold']
            if 'prefix_padding_ms' in turn_detection_config:
                vad_config["prefix_padding_ms"] = turn_detection_config['prefix_padding_ms']
            if 'silence_duration_ms' in turn_detection_config:
                vad_config["silence_duration_ms"] = turn_detection_config['silence_duration_ms']
            if 'create_response' in turn_detection_config:
                vad_config["create_response"] = turn_detection_config['create_response']
            if 'interrupt_response' in turn_detection_config:
                vad_config["interrupt_response"] = turn_detection_config['interrupt_response']
                
            session_config["turn_detection"] = vad_config
            logger.debug(f"Configured for server VAD mode: {vad_config}")
            
        elif turn_detection_config.get('type') == 'semantic_vad':
            # Semantic VAD configuration
            vad_config = {
                "type": "semantic_vad"
            }
            
            # Add optional semantic VAD parameters if specified
            if 'create_response' in turn_detection_config:
                vad_config["create_response"] = turn_detection_config['create_response']
            if 'interrupt_response' in turn_detection_config:
                vad_config["interrupt_response"] = turn_detection_config['interrupt_response']
                
            session_config["turn_detection"] = vad_config
            logger.debug(f"Configured for semantic VAD mode: {vad_config}")
        
        return session_config

    async def run(self):
        self._display_message("status_info", "Starting Voice Chat Client...")
        self._pyaudio_instance = pyaudio.PyAudio()
        try:
            # Start audio playback thread
            self._start_audio_playback_thread()
            
            self._pyaudio_stream = self._pyaudio_instance.open(
                format=PYAUDIO_FORMAT, channels=PYAUDIO_CHANNELS, rate=PYAUDIO_RATE,
                input=True, frames_per_buffer=PYAUDIO_CHUNK
            )
            self._display_message("mic_ready", f"Microphone stream opened (24kHz, 16-bit, mono).")

            headers = await self._get_auth_headers()
            self._display_message("status_connect", f"Connecting to {self._websocket_url}...")
            
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.ws_connect(self._websocket_url) as ws_connection:
                    self._ws = ws_connection
                    self._display_message("status_ok", "Successfully connected to Azure OpenAI.")
                    
                    # Configure session with YAML-based configuration
                    session_data = self._construct_session_config()
                    session_config = {
                        "type": "session.update",
                        "session": session_data
                    }
                    logger.debug(f"Sending session update: {json.dumps(session_config, indent=2)}")
                    
                    await self._ws.send_json(session_config)
                    self._display_message("status_info", f"Sent session configuration ({self.yaml_config['scenario']}).")

                    # Display appropriate instructions based on VAD mode
                    if self._is_push_to_talk:
                        self._display_message("user_action", "Press SPACEBAR to talk, 'q' to quit.")
                    else:
                        if self._vad_mode == 'server_vad':
                            self._display_message("user_action", "Speak naturally (Server VAD enabled). Press SPACEBAR to manually trigger response, 'q' to quit.")
                        elif self._vad_mode == 'semantic_vad':
                            self._display_message("user_action", "Speak naturally (Semantic VAD enabled). Press SPACEBAR to manually trigger response, 'q' to quit.")
                        else:
                            self._display_message("user_action", "Speak naturally (VAD enabled), 'q' to quit.")

                    loop = asyncio.get_event_loop()
                    self._keyboard_listener = keyboard.Listener(
                        on_press=lambda k: loop.call_soon_threadsafe(self._on_keyboard_press, k),
                        on_release=lambda k: loop.call_soon_threadsafe(self._on_keyboard_release, k)
                    )
                    self._keyboard_listener.start()
                    self._display_message("system_event", "Keyboard listener started.")

                    self._tasks = [
                        asyncio.create_task(self._audio_capture_loop()),
                        asyncio.create_task(self._stream_audio_loop()),
                        asyncio.create_task(self._handle_recording_completion()),
                        asyncio.create_task(self._receive_server_messages_loop())
                    ]
                    
                    logger.debug("All async tasks started, waiting for quit event")
                    await self._quit_event.wait() # Keep running until quit is signaled

        except aiohttp.ClientConnectorError as e:
            self._display_message("status_error", f"Connection failed: {e}")
        except Exception as e:
            if "pyaudio" in str(e).lower():
                self._display_message("status_error", f"PyAudio error: {e}. Check microphone/PortAudio.")
            elif isinstance(e, VoiceChatClientError):
                self._display_message("status_error", f"Client setup error: {e}")
            else:
                self._display_message("status_error", f"An unexpected error occurred in run: {e}")
            logger.error(f"Critical error during client execution\n{traceback.format_exc()}")
            logger.exception("Unexpected error in run method") # Log full traceback
        finally:
            self._display_message("system_event", "Shutting down...")
            
            # Signal quit event again to ensure all loops are aware if not already set
            if not self._quit_event.is_set():
                self._quit_event.set()

            # Stop audio playback thread
            self._stop_audio_playback_thread()

            # Stop keyboard listener
            if self._keyboard_listener:
                self._keyboard_listener.stop()
                # self._keyboard_listener.join() # pynput listener join can block if not stopped from its own thread
                self._display_message("system_event", "Keyboard listener stopped.")
            
            # Cancel and await all tasks
            for task in self._tasks:
                if task and not task.done():
                    task.cancel()
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            self._display_message("system_event", "All async tasks processed.")

            # Close WebSocket
            if self._ws and not self._ws.closed:
                await self._ws.close()
                self._display_message("status_info", "WebSocket connection closed.")

            # Clean up PyAudio
            if self._pyaudio_stream:
                self._pyaudio_stream.stop_stream()
                self._pyaudio_stream.close()
                self._display_message("system_event", "Microphone stream closed.")
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
                self._display_message("system_event", "PyAudio terminated.")
            
            # Clean up pygame mixer
            try:
                pygame.mixer.quit()
                self._display_message("system_event", "Pygame mixer stopped.")
            except Exception as e:
                logger.debug(f"Error stopping pygame mixer: {e}")
            
            # Close Azure credential
            if self._credential:
                await self._credential.close()
                self._display_message("system_event", "Azure credential closed.")
            
            self._display_message("quit", "Voice Chat Client shut down gracefully.")

async def main():
    """Main application function with command line argument support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Azure OpenAI Realtime Voice Chat Client with configurable VAD modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                                    # Use default push-to-talk mode
  python main.py -f configs/server-vad.yaml        # Use server VAD configuration
  python main.py -f configs/semantic-vad.yaml      # Use semantic VAD configuration
  python main.py -f configs/server-vad-sensitive.yaml  # Use sensitive server VAD

Available configurations:
  - configs/push-to-talk.yaml: Manual push-to-talk mode (spacebar control)
  - configs/server-vad.yaml: Server-side voice activity detection
  - configs/semantic-vad.yaml: Semantic voice activity detection
  - configs/server-vad-sensitive.yaml: High sensitivity server VAD
  - configs/server-vad-robust.yaml: Low sensitivity server VAD for noisy environments
        '''
    )
    parser.add_argument(
        '-f', '--config-file',
        type=str,
        help='Path to YAML configuration file (default: use built-in push-to-talk config)'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load YAML configuration
    yaml_config = None
    if args.config_file:
        try:
            config_file = Path(args.config_file)
            if not config_file.exists():
                print(f"❌ Configuration file not found: {args.config_file}")
                return
            
            with open(config_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration: {yaml_config.get('scenario', 'Unknown')} - {yaml_config.get('description', 'No description')}")
        except yaml.YAMLError as e:
            print(f"❌ Invalid YAML syntax in {args.config_file}: {e}")
            return
        except Exception as e:
            print(f"❌ Error loading configuration from {args.config_file}: {e}")
            return
    else:
        print("ℹ️  Using default push-to-talk mode. Use -f to specify a config file for VAD modes.")
    
    # Load Azure environment configuration
    app_config = {
        "AZURE_OPENAI_ENDPOINT": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_DEPLOYMENT_NAME": os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME"),
        "OPENAI_API_VERSION": os.environ.get("OPENAI_API_VERSION"),
    }
    
    # Validate required environment variables
    missing_vars = [k for k, v in app_config.items() if not v]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file or environment variables.")
        return
    
    # Create and run client
    client = AzureOpenAIRealtimeClient(config=app_config, yaml_config=yaml_config)
    try:
        await client.run()
    except Exception as e: # Catch-all for unexpected errors during client.run() setup or early exit
        logger.exception("Critical error during client execution")
        print(f"❌❌❌ Critical client error: {e} ❌❌❌")
    finally:
        print("➡️  INFO: Application main function finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Application interrupted by user (Ctrl+C).")
    finally:
        print("➡️  INFO: Main program exit.")
