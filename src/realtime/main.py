import os
import asyncio
import base64
import argparse
import yaml
import uuid
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from jinja2 import Environment, FileSystemLoader, select_autoescape
from openai import AsyncOpenAI
import pyaudio
from pynput import keyboard
import logging
import json
import sounddevice as sd
import numpy as np
import tempfile
import threading
import queue
import wave
import traceback
import subprocess
import time
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
        self._base_url = self._construct_base_url()
        self._deployment_name = config.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        
        if not self._deployment_name:
            raise VoiceChatClientError("Missing AZURE_OPENAI_DEPLOYMENT_NAME configuration variable.")

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
        self._connection = None  # OpenAI Realtime connection

        # For managing async tasks
        self._tasks = []

        # VAD mode configuration
        self._vad_mode = self.yaml_config.get('turn_detection', {}).get('type')
        self._is_push_to_talk = self._vad_mode is None
        
        # Load system prompt template
        self._system_prompt = self._load_system_prompt()

        # Initialize sounddevice for audio playback - direct streaming without temp files
        # This is much better for Bluetooth as it doesn't create file I/O overhead
        try:
            # Configure sounddevice for real-time audio streaming
            sd.default.samplerate = 24000
            sd.default.channels = 1
            sd.default.dtype = 'int16'
            # Larger latency for Bluetooth stability
            sd.default.latency = 'high'  # Use system's high-latency setting
            
            # Test audio output
            devices = sd.query_devices()
            logger.debug(f"Audio output device: {sd.default.device}")
            logger.debug("Sounddevice initialized for direct audio streaming")
        except Exception as e:
            logger.error(f"Failed to initialize sounddevice: {e}")
            raise VoiceChatClientError(f"Audio initialization failed: {e}")
        
        # Audio playback queue and thread
        self._audio_playback_queue = Queue()
        self._audio_playback_thread = None
        self._stop_audio_playback = threading.Event()
        self._interrupt_audio_playback = threading.Event()  # For interrupting current playback
        
        # Audio streaming with sounddevice - no file creation needed
        self._audio_stream = None
        self._audio_queue_sd = queue.Queue(maxsize=100)
        self._stream_lock = threading.Lock()

        # Current response text buffer
        self._current_response_text = ""
        
        # Function calling configuration
        self._function_call_attempt_counter = 0  # Round-robin counter for mock
        self._async_completion_delay = 10.0  # Delay for async job completion in seconds
        self._pending_jobs = {}  # Track pending async jobs: {job_id: result}
        self._response_active = False  # Track if a response is currently active

        # Display configuration info
        if self.yaml_config.get('scenario'):
            self._display_message("status_info", f"Mode: {self.yaml_config['scenario']}")
        if self.yaml_config.get('description'):
            self._display_message("status_info", f"Description: {self.yaml_config['description']}")
        self._display_message("status_info", f"VAD Mode: {self._vad_mode or 'Push-to-Talk'}")
        self._display_message("status_info", f"Voice: alloy")
        self._display_message("status_info", f"Audio: Direct streaming (sounddevice)")
        self._display_message("status_info", f"Sample rate: 24kHz, Latency: high (Bluetooth optimized)")

    def _load_system_prompt(self) -> str:
        """Load and render system prompt from Jinja2 template.
        
        Returns:
            str: Rendered system prompt
        """
        try:
            # Get template directory
            template_dir = Path(__file__).parent / "templates"
            
            # Set up Jinja2 environment
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
            # Load template
            template = env.get_template('system_prompt.j2')
            
            # Render with any context variables from YAML config
            context = self.yaml_config.get('prompt_context', {})
            rendered_prompt = template.render(**context)
            
            logger.debug(f"Loaded system prompt from template ({len(rendered_prompt)} chars)")
            return rendered_prompt
            
        except Exception as e:
            logger.warning(f"Failed to load system prompt template: {e}")
            # Fallback to default prompt
            return "You are a helpful AI assistant. Respond naturally and conversationally."
    
    def _load_system_prompt(self) -> str:
        """Load and render system prompt from Jinja2 template.
        
        Returns:
            str: Rendered system prompt
        """
        try:
            # Get template directory
            template_dir = Path(__file__).parent / "templates"
            
            # Set up Jinja2 environment
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            
            # Load template
            template = env.get_template('system_prompt.j2')
            
            # Render with any context variables from YAML config
            context = self.yaml_config.get('prompt_context', {})
            rendered_prompt = template.render(**context)
            
            logger.debug(f"Loaded system prompt from template ({len(rendered_prompt)} chars)")
            return rendered_prompt
            
        except Exception as e:
            logger.warning(f"Failed to load system prompt template: {e}")
            # Fallback to default prompt
            return "You are a helpful AI assistant. Respond naturally and conversationally."
    
    def _construct_base_url(self) -> str:
        """Construct base URL for OpenAI client."""
        endpoint = self.config.get("AZURE_OPENAI_ENDPOINT")

        if not endpoint:
            raise VoiceChatClientError("Missing AZURE_OPENAI_ENDPOINT configuration variable.")

        # Convert https:// to wss:// and add /openai/v1 path
        return endpoint.replace("https://", "wss://").rstrip("/") + "/openai/v1"

    async def _get_auth_token(self) -> str:
        """Get authentication token for OpenAI client."""
        return self._token_provider()

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

    def _audio_callback(self, outdata, frames, time_info, status):
        """Sounddevice callback for audio playback - called by audio system."""
        if status:
            logger.debug(f"Audio callback status: {status}")
        
        try:
            # Get audio data from queue
            data = self._audio_queue_sd.get_nowait()
            # Convert bytes to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            
            # Fill output buffer
            if len(audio_array) < len(outdata):
                # Pad with zeros if not enough data
                outdata[:len(audio_array)] = audio_array.reshape(-1, 1)
                outdata[len(audio_array):] = 0
            else:
                outdata[:] = audio_array[:len(outdata)].reshape(-1, 1)
                
        except queue.Empty:
            # No data available - output silence
            outdata.fill(0)
    
    def _audio_playback_worker(self):
        """Background thread worker for audio streaming with sounddevice."""
        try:
            # Start audio stream without callback - we'll write directly
            with sd.OutputStream(samplerate=24000, channels=1, dtype='int16',
                               blocksize=0,  # Auto blocksize
                               latency='high') as stream:
                logger.debug("Audio stream started")
                
                # Buffer to accumulate audio for smooth playback
                audio_buffer = bytearray()
                min_buffer_size = 9600  # 200ms worth of audio (24000 * 2 bytes * 0.2)
                
                # Process audio data from queue
                while not self._stop_audio_playback.is_set():
                    try:
                        # Get audio data from main queue
                        audio_data = self._audio_playback_queue.get(timeout=0.1)
                        if audio_data is None:  # Poison pill
                            break
                        
                        # Add to buffer
                        audio_buffer.extend(audio_data)
                        self._audio_playback_queue.task_done()
                        
                        # Play accumulated audio when we have enough
                        while len(audio_buffer) >= min_buffer_size:
                            # Extract chunk to play
                            chunk = bytes(audio_buffer[:min_buffer_size])
                            audio_buffer = audio_buffer[min_buffer_size:]
                            
                            # Convert to numpy array and play
                            audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
                            stream.write(audio_array)
                            logger.debug(f"Played {len(chunk)} bytes")
                        
                    except queue.Empty:
                        # Play any remaining audio in buffer during idle
                        if len(audio_buffer) > 0:
                            chunk = bytes(audio_buffer)
                            audio_buffer.clear()
                            audio_array = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
                            stream.write(audio_array)
                        continue
                        
        except Exception as e:
            logger.error(f"Error in audio playback worker: {e}")
        finally:
            logger.debug("Audio stream stopped")

    def _play_audio(self, audio_data: bytes):
        """Queue audio data for playback."""
        if not self._stop_audio_playback.is_set():
            logger.debug(f"Queuing audio data for playback: {len(audio_data)} bytes")
            self._audio_playback_queue.put(audio_data)

    def _start_audio_playback_thread(self):
        """Start the audio playback thread."""
        if self._audio_playback_thread is None or not self._audio_playback_thread.is_alive():
            self._stop_audio_playback.clear()
            self._interrupt_audio_playback.clear()
            self._audio_playback_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
            self._audio_playback_thread.start()
            self._display_message("system_event", "Audio playback thread started.")

    def _stop_audio_playback_thread(self):
        """Stop the audio playback thread."""
        if self._audio_playback_thread and self._audio_playback_thread.is_alive():
            self._stop_audio_playback.set()
            self._interrupt_audio_playback.set()
            self._audio_playback_queue.put(None)  # Poison pill
            self._audio_playback_thread.join(timeout=2.0)
            self._display_message("system_event", "Audio playback thread stopped.")

    def _flush_audio_playback_buffer(self):
        """Flush the audio playback buffer and stop current playback for interruption."""
        logger.debug("Flushing audio playback buffer due to interruption")
        
        # Clear the sounddevice queue
        try:
            while not self._audio_queue_sd.empty():
                try:
                    self._audio_queue_sd.get_nowait()
                except queue.Empty:
                    break
        except Exception as e:
            logger.debug(f"Error clearing audio queue: {e}")
        
        # Also clear the main playback queue
        try:
            while not self._audio_playback_queue.empty():
                try:
                    self._audio_playback_queue.get_nowait()
                    self._audio_playback_queue.task_done()
                except queue.Empty:
                    break
        except Exception as e:
            logger.debug(f"Error clearing playback queue: {e}")
            
        self._display_message("system_event", "Audio playback interrupted and buffer flushed")

    def _on_keyboard_press(self, key):
        if key == keyboard.Key.space:
            if self._is_push_to_talk and not self._is_recording:
                self._display_message("mic_rec_start", "Recording...")
                self._is_recording = True
                # Clear any existing audio buffer when starting new recording
                if self._connection:
                    try:
                        loop = asyncio.get_event_loop()
                        loop.create_task(self._clear_audio_buffer())
                    except Exception as e:
                        logger.error(f"Error scheduling audio buffer clear: {e}")
            elif not self._is_push_to_talk:
                # In VAD modes, spacebar can be used to manually trigger response
                # Also interrupt current AI playback when user wants to speak
                self._handle_user_interruption()
                if self._connection:
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
        if self._connection:
            try:
                # Commit the audio buffer and trigger response using OpenAI SDK
                await self._connection.input_audio_buffer.commit()
                await self._connection.response.create()
                
                self._display_message("ai_thinking", "Manual response triggered...")
                logger.debug("Manually triggered response in VAD mode")
            except Exception as e:
                logger.error(f"Error triggering manual response: {e}")

    async def _clear_audio_buffer(self):
        """Clear the audio input buffer to start fresh recording."""
        if self._connection:
            try:
                await self._connection.input_audio_buffer.clear()
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
        """Stream audio chunks to OpenAI using SDK's append method."""
        self._display_message("system_event", "Audio streaming loop started.")
        
        while not self._quit_event.is_set() and self._connection:
            try:
                # Get audio chunk from queue (blocks until available)
                audio_chunk = await asyncio.wait_for(self._audio_chunk_queue.get(), timeout=0.1)
                
                if self._quit_event.is_set() or not self._connection:
                    break
                
                # Encode audio chunk as base64
                audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                # Send audio chunk using OpenAI SDK
                await self._connection.input_audio_buffer.append(audio=audio_base64)
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
        
        while not self._quit_event.is_set() and self._connection:
            await self._space_released_event.wait()
            self._space_released_event.clear()

            if self._quit_event.is_set() or not self._connection: 
                break

            # Commit the audio buffer and request response
            self._display_message("audio_send", "Finalizing audio input...")
            
            try:
                # Commit the audio buffer using OpenAI SDK
                await self._connection.input_audio_buffer.commit()
                logger.debug("Committed input audio buffer")
                
                # Request AI response using OpenAI SDK
                await self._connection.response.create()
                logger.debug("Created response request")
                
                self._display_message("ai_thinking", "Audio submitted, waiting for AI response...")
                
            except Exception as e:
                logger.error(f"Error handling recording completion: {e}")
                self._display_message("status_error", f"Error submitting audio: {e}")
                
        self._display_message("system_event", "Recording completion handler finished.")

    async def _receive_server_messages_loop(self):
        """Receive server messages using OpenAI SDK event stream."""
        self._display_message("system_event", "Message receiving loop started.")
        try:
            async for event in self._connection:
                if self._quit_event.is_set(): 
                    break

                logger.debug(f"Received event type: {event.type}")
                
                msg_type = event.type
                
                # Handle different event types using OpenAI SDK patterns
                if msg_type == "response.text.delta":
                    # Accumulate text for clean display
                    delta_text = event.delta
                    self._current_response_text += delta_text
                    self._display_message("ai_resp_text_delta", delta_text, end="", flush=True)
                elif msg_type == "response.text.done":
                    logger.debug("AI text response completed")
                    self._display_message("ai_resp_text_delta", "", end='\n')
                    self._current_response_text = ""  # Reset for next response
                elif msg_type == "response.output_audio.delta":
                    # Handle audio data chunks - SDK provides base64 string
                    audio_b64 = event.delta
                    if audio_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_b64)
                            self._play_audio(audio_bytes)
                            logger.debug(f"Queued {len(audio_bytes)} bytes of audio for playback")
                        except Exception as e:
                            logger.warning(f"Failed to decode audio delta: {e}")
                elif msg_type == "response.output_audio.done":
                    logger.debug("AI audio response completed")
                    # Removed audio playback complete message for cleaner output
                elif msg_type == "response.output_audio_transcript.delta":
                    # Display transcript text cleanly without prefix
                    transcript_text = event.delta
                    self._display_message("ai_resp_text_delta", transcript_text, end="", flush=True)
                elif msg_type == "response.output_audio_transcript.done":
                    logger.debug("AI audio transcript completed")
                    self._display_message("ai_resp_text_delta", " [Done]", end='\n')
                elif msg_type == "response.done":
                    logger.debug("AI response completed")
                    self._response_active = False
                    self._display_message("status_ok", "--- End of AI Response ---")
                elif msg_type == "conversation.interrupted":
                    logger.debug("Conversation interrupted by server")
                    self._display_message("system_event", "🛑 AI response interrupted")
                    # Flush audio playback buffer to stop current playback immediately
                    self._flush_audio_playback_buffer()
                elif msg_type == "response.cancelled":
                    logger.debug("AI response cancelled by server")
                    self._response_active = False
                    self._display_message("system_event", "🛑 AI response cancelled")
                    # Flush audio playback buffer to stop current playback immediately
                    self._flush_audio_playback_buffer()
                elif msg_type == "input_audio_buffer.speech_started":
                    logger.debug("User speech detected - interrupting AI playback")
                    self._display_message("system_event", "🎤 Speech detected - interrupting AI")
                    # Handle user interruption when speech is detected
                    self._handle_user_interruption()
                elif msg_type == "error":
                    error_msg = event.error.message if hasattr(event.error, 'message') else 'Unknown error'
                    error_code = event.error.code if hasattr(event.error, 'code') else 'No code'
                    
                    # Handle benign errors that shouldn't stop the application
                    if error_code == "response_cancel_not_active":
                        # This is expected when user speaks but AI isn't responding yet
                        logger.debug(f"Benign error - {error_code}: {error_msg}")
                        continue
                    
                    # Fatal errors that should stop the application
                    logger.error(f"Server error - Code: {error_code}, Message: {error_msg}")
                    self._display_message("status_error", f"Server error: {error_msg}")
                    self._quit_event.set()
                    break
                elif msg_type == "session.created":
                    logger.debug(f"Session created: {event.session.id}")
                elif msg_type == "session.updated":
                    logger.debug(f"Session updated: {event.session.id}")
                elif msg_type == "conversation.item.created":
                    logger.debug(f"Conversation item created")
                elif msg_type == "response.created":
                    logger.debug(f"Response created")
                    self._response_active = True
                elif msg_type == "response.function_call_arguments.done":
                    # Function call with complete arguments
                    logger.debug(f"Function call arguments complete")
                    call_id = event.call_id if hasattr(event, 'call_id') else None
                    function_name = event.name if hasattr(event, 'name') else None
                    arguments_str = event.arguments if hasattr(event, 'arguments') else '{}'
                    
                    if call_id and function_name:
                        try:
                            arguments = json.loads(arguments_str)
                            # Handle function call in background
                            asyncio.create_task(self._handle_function_call(call_id, function_name, arguments))
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse function arguments: {e}")
                else:
                    logger.debug(f"Unhandled message type: {msg_type}")
        except Exception as e:
            if not self._quit_event.is_set():
                logger.exception("Error in message receiving loop")
                self._display_message("status_error", f"Error in message receiving loop: {e}")
            self._quit_event.set()
        finally:
            self._display_message("system_event", "Message receiving loop finished.")
            if not self._quit_event.is_set():
                self._quit_event.set()


    async def _handle_function_call(self, call_id: str, function_name: str, arguments: dict):
        """Handle function call from the model with round-robin async mock.
        
        Args:
            call_id: The function call ID from the model
            function_name: Name of the function to call
            arguments: Function arguments as dict
        """
        try:
            logger.info(f"Function call: {function_name} with args: {arguments}")
            self._display_message("system_event", f"📞 Function call: {function_name}")
            self._display_message("system_event", f"   Destination: {arguments.get('destination', 'N/A')}")
            self._display_message("system_event", f"   Vacation Type: {arguments.get('vacation_type', 'N/A')}")
            self._display_message("system_event", f"   Summary: {arguments.get('conversation_summary', 'N/A')}")
            self._display_message("system_event", f"   Mood: {arguments.get('user_mood', 'N/A')}")
            self._display_message("system_event", f"   Happiness: {arguments.get('happiness_score', 0):.2f}")
            
            if function_name == "get_destination_info":
                # Increment attempt counter for round-robin
                self._function_call_attempt_counter += 1
                attempt_num = self._function_call_attempt_counter
                is_odd_attempt = attempt_num % 2 == 1
                
                destination = arguments.get('destination', 'Unknown')
                vacation_type = arguments.get('vacation_type', 'general vacation')
                happiness = arguments.get('happiness_score', 0.0)
                
                if is_odd_attempt:
                    # Odd attempts: complete quickly (4 seconds) with full result
                    self._display_message("system_event", f"⏳ Attempt #{attempt_num} (ODD): Simulating quick API call (4s)...")
                    await asyncio.sleep(4.0)
                    
                    # Generate full mock response
                    result = {
                        "destination": destination,
                        "available_packages": 3,
                        "price_range": "15,000 - 45,000 CZK",
                        "best_season": "duben - září",
                        "highlights": [
                            "Luxusní hotely s all-inclusive",
                            "Průvodce v češtině",
                            "Speciální nabídky pro rodiny"
                        ],
                        "customer_sentiment": "positive" if happiness > 0 else "neutral" if happiness == 0 else "needs_attention"
                    }
                    
                    # Send function result back to model
                    await self._connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(result, ensure_ascii=False)
                        }
                    )
                    
                    # Trigger response generation
                    await self._connection.response.create()
                    
                    self._display_message("system_event", "✅ Function call completed synchronously")
                    
                else:
                    # Even attempts: timeout after 5s, return pending status with job_id
                    self._display_message("system_event", f"⏳ Attempt #{attempt_num} (EVEN): Simulating slow API (5s timeout)...")
                    await asyncio.sleep(5.0)
                    
                    # Generate job ID
                    job_id = str(uuid.uuid4())
                    
                    # Return pending status
                    pending_result = {
                        "status": "pending",
                        "job_id": job_id,
                        "message": "Your request has been acknowledged and is being processed in the background. This may take a moment."
                    }
                    
                    # Send pending result back to model
                    await self._connection.conversation.item.create(
                        item={
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(pending_result, ensure_ascii=False)
                        }
                    )
                    
                    # Trigger response so model can acknowledge
                    await self._connection.response.create()
                    
                    self._display_message("system_event", f"⏳ Function call returned PENDING status (job_id: {job_id})")
                    self._display_message("system_event", f"📋 Starting background async job (will complete in {self._async_completion_delay}s)")
                    
                    # Start background task to simulate async completion
                    asyncio.create_task(
                        self._handle_async_job_completion(job_id, destination, vacation_type, happiness)
                    )
                
            else:
                logger.warning(f"Unknown function: {function_name}")
                
        except Exception as e:
            logger.error(f"Error handling function call: {e}")
            self._display_message("status_error", f"Function call error: {e}")
    
    async def _handle_async_job_completion(self, job_id: str, destination: str, vacation_type: str, happiness: float):
        """Handle async job completion by polling and injecting result when ready.
        
        Args:
            job_id: The job ID to track
            destination: Destination name
            vacation_type: Type of vacation
            happiness: Happiness score
        """
        try:
            # Simulate async work with configurable delay
            elapsed = 0.0
            poll_interval = 1.0  # Check every 1 second
            
            while elapsed < self._async_completion_delay:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval
                logger.debug(f"Polling async job {job_id}: {elapsed:.1f}s / {self._async_completion_delay:.1f}s")
                
                # Check if we should quit
                if self._quit_event.is_set():
                    logger.info(f"Async job {job_id} cancelled due to quit")
                    return
            
            # Job completed - generate final result
            self._display_message("system_event", f"✅ Async job {job_id} completed! Injecting result...")
            
            # Cancel active response if one is in progress (to interrupt filler speech)
            if self._response_active:
                self._display_message("system_event", "⚡ Cancelling active response to deliver async result")
                try:
                    await self._connection.response.cancel()
                    # Flush audio playback buffer to stop filler speech immediately
                    self._flush_audio_playback_buffer()
                    await asyncio.sleep(0.1)  # Brief pause for cancel to take effect
                except Exception as e:
                    logger.warning(f"Could not cancel active response: {e}")
            
            result = {
                "destination": destination,
                "available_packages": 3,
                "price_range": "15,000 - 45,000 CZK",
                "best_season": "duben - září",
                "highlights": [
                    "Luxusní hotely s all-inclusive",
                    "Průvodce v češtině",
                    "Speciální nabídky pro rodiny"
                ],
                "customer_sentiment": "positive" if happiness > 0 else "neutral" if happiness == 0 else "needs_attention"
            }
            
            # Inject result as a new system message in conversation
            await self._connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"ASYNC JOB COMPLETED: Job {job_id} has finished. The destination information is now available: {json.dumps(result, ensure_ascii=False)}"
                        }
                    ]
                }
            )
            
            # Trigger response generation so model can speak the result
            await self._connection.response.create()
            
            self._display_message("system_event", f"📢 Triggered response for async result (job_id: {job_id})")
            
        except Exception as e:
            logger.error(f"Error in async job completion handler: {e}")
            self._display_message("status_error", f"Async job error: {e}")
    
    def _construct_session_config(self) -> dict:
        """Construct session configuration based on YAML config.
        
        Returns:
            dict: Session configuration for OpenAI Realtime API (official format)
        """
        session_config = {
            "type": "realtime",
            "instructions": self._system_prompt,
            "output_modalities": ["audio"],
            "tools": [
                {
                    "type": "function",
                    "name": "get_destination_info",
                    "description": "Retrieves vacation information, packages, hotels, and activities for a specific destination. IMPORTANT: Call this function EVERY TIME the customer mentions a new destination OR a new type of vacation for any destination (e.g., if they first ask about Paris for sightseeing, then about Paris for romantic getaway - call again). Always call when destination or vacation type changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "destination": {
                                "type": "string",
                                "description": "The destination name (e.g., 'Paris', 'Bali', 'Tokyo', 'Prague')"
                            },
                            "vacation_type": {
                                "type": "string",
                                "description": "Type of vacation the customer is interested in (e.g., 'beach vacation', 'city sightseeing', 'romantic getaway', 'family trip', 'adventure travel', 'wellness retreat', 'cultural tour')"
                            },
                            "conversation_summary": {
                                "type": "string",
                                "description": "Brief summary of what the customer has shared so far and what they want. Include: their preferences, constraints (budget, dates, family size), any concerns mentioned, and their main goal for this vacation. Example: 'Customer wants beach vacation for family of 4, budget around 30k CZK, prefers all-inclusive, worried about kids activities'"
                            },
                            "user_mood": {
                                "type": "string",
                                "description": "Describe the customer's emotional state and attitude in your own words based on the conversation. Examples: 'excited and enthusiastic', 'neutral and inquiring', 'frustrated with previous options', 'disappointed but hopeful', 'very happy and eager', 'slightly annoyed', 'calm and considerate'. Be specific and natural."
                            },
                            "happiness_score": {
                                "type": "number",
                                "description": "Customer's emotional state as a numeric score. Range: -1.0 (extremely angry/frustrated) to 1.0 (extremely happy/excited). 0.0 is neutral. This is a simplified metric - use user_mood for detailed description.",
                                "minimum": -1.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["destination", "vacation_type", "conversation_summary", "user_mood", "happiness_score"]
                    }
                }
            ],
            "audio": {
                "input": {
                    "transcription": {
                        "model": "whisper-1"
                    },
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    }
                },
                "output": {
                    "voice": "alloy",
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    }
                }
            }
        }
        
        # Configure turn detection based on YAML config in official format
        turn_detection_config = self.yaml_config.get('turn_detection', {})
        
        if turn_detection_config.get('type') is None:
            # Push-to-talk mode - no automatic turn detection
            session_config["audio"]["input"]["turn_detection"] = None
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
                
            session_config["audio"]["input"]["turn_detection"] = vad_config
            logger.debug(f"Configured for server VAD mode: {vad_config}")
            
        elif turn_detection_config.get('type') == 'semantic_vad':
            # Semantic VAD configuration
            vad_config = {
                "type": "semantic_vad"
            }
            
            # Add optional semantic VAD parameters if specified
            if 'create_response' in turn_detection_config:
                vad_config["create_response"] = turn_detection_config['create_response']
                
            session_config["audio"]["input"]["turn_detection"] = vad_config
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

            # Get auth token for OpenAI client
            token = await self._get_auth_token()
            self._display_message("status_connect", f"Connecting to Azure OpenAI...")
            
            # Create OpenAI client with Azure configuration
            client = AsyncOpenAI(
                websocket_base_url=self._base_url,
                api_key=token
            )
            
            # Connect using OpenAI SDK's realtime connection
            async with client.realtime.connect(model=self._deployment_name) as connection:
                self._connection = connection
                self._display_message("status_ok", "Successfully connected to Azure OpenAI.")
                
                # Configure session with YAML-based configuration
                session_data = self._construct_session_config()
                logger.debug(f"Sending session update: {json.dumps(session_data, indent=2)}")
                
                await connection.session.update(session=session_data)
                self._display_message("status_info", f"Sent session configuration ({self.yaml_config.get('scenario', 'default')}).")

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
                await self._quit_event.wait()  # Keep running until quit is signaled

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

            # Connection is closed automatically by context manager
            if self._connection:
                self._display_message("status_info", "Connection closed.")
                self._connection = None

            # Clean up PyAudio
            if self._pyaudio_stream:
                self._pyaudio_stream.stop_stream()
                self._pyaudio_stream.close()
                self._display_message("system_event", "Microphone stream closed.")
            if self._pyaudio_instance:
                self._pyaudio_instance.terminate()
                self._display_message("system_event", "PyAudio terminated.")
            
            # Close Azure credential
            if self._credential:
                self._credential.close()
                self._display_message("system_event", "Azure credential closed.")
            
            self._display_message("quit", "Voice Chat Client shut down gracefully.")

    def _handle_user_interruption(self):
        """Handle user interruption (speaking, key press, etc.) by flushing audio and notifying server."""
        logger.debug("Handling user interruption")
        
        # Immediately flush audio buffer
        self._flush_audio_playback_buffer()
        
        # Send cancel request to server if connection is open
        if self._connection:
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self._send_response_cancel())
            except Exception as e:
                logger.error(f"Error scheduling response cancel: {e}")

    async def _send_response_cancel(self):
        """Send response.cancel event to stop ongoing AI response."""
        try:
            await self._connection.response.cancel()
            logger.debug("Sent response.cancel to server")
        except Exception as e:
            logger.error(f"Error sending response.cancel: {e}")

    def _detect_bluetooth_audio(self):
        """Detect if we're likely using Bluetooth audio and adjust settings accordingly."""
        try:
            import subprocess
            # Check for Bluetooth audio devices on Windows
            result = subprocess.run(['powershell', '-Command', 
                                   'Get-WmiObject -Class Win32_PnPEntity | Where-Object {$_.Name -like "*Bluetooth*" -and $_.Name -like "*Audio*"}'],
                                   capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0 and result.stdout.strip():
                self._display_message("status_info", "🔵 Bluetooth audio device detected - using optimized settings")
                self._bluetooth_delay = 0.02  # Minimal delay with very large chunks
                self._min_chunk_size = 144000  # Very large chunks for Bluetooth (~3000ms = 3 seconds)
                return True
        except Exception as e:
            logger.debug(f"Could not detect Bluetooth audio: {e}")
        
        return False

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
