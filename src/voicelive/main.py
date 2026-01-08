"""
Azure Voice Live API Client

A real-time voice assistant using Microsoft's Voice Live API with configurable
VAD modes, voice options, and function calling capabilities.

This implementation mirrors the Azure OpenAI Realtime API client structure
but uses the Voice Live API for enhanced conversational features.
"""

import os
import asyncio
import base64
import argparse
import yaml
import uuid
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Union, Callable, Mapping

from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.identity.aio import DefaultAzureCredential
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pynput import keyboard
import pyaudio

# Load environment variables first
load_dotenv()

# Configure logging based on LOG_LEVEL environment variable
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PyAudio configuration - Voice Live API supports 16kHz and 24kHz
PYAUDIO_FORMAT = pyaudio.paInt16
PYAUDIO_CHANNELS = 1
PYAUDIO_RATE = 24000  # 24kHz sample rate (also supports 16kHz)
PYAUDIO_CHUNK = 1200  # ~50ms chunks at 24kHz


class VoiceLiveClientError(Exception):
    """Custom exception for Voice Live client errors."""
    pass


class AudioProcessor:
    """
    Handles real-time audio capture and playback for the voice assistant.
    
    Threading Architecture:
    - Main thread: Event loop and UI
    - Capture thread: PyAudio input stream callback
    - Playback thread: PyAudio output stream callback
    """
    
    class AudioPlaybackPacket:
        """Represents a packet that can be sent to the audio playback queue."""
        def __init__(self, seq_num: int, data: Optional[bytes]):
            self.seq_num = seq_num
            self.data = data
    
    def __init__(self, connection, loop: asyncio.AbstractEventLoop):
        """Initialize audio processor with Voice Live connection.
        
        Args:
            connection: VoiceLiveConnection instance
            loop: asyncio event loop for thread-safe coroutine scheduling
        """
        self.connection = connection
        self.loop = loop
        self.audio = pyaudio.PyAudio()
        
        # Audio configuration - PCM16, 24kHz, mono
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = PYAUDIO_RATE
        self.chunk_size = PYAUDIO_CHUNK
        
        # Capture and playback state
        self.input_stream: Optional[pyaudio.Stream] = None
        self.output_stream: Optional[pyaudio.Stream] = None
        
        # Playback queue management
        self.playback_queue: queue.Queue[AudioProcessor.AudioPlaybackPacket] = queue.Queue()
        self.playback_base = 0
        self.next_seq_num = 0
        
        # Control flags
        self._is_capturing = False
        
        logger.info(f"AudioProcessor initialized with {self.rate}Hz PCM16 mono audio")
    
    def start_capture(self):
        """Start capturing audio from microphone."""
        def _capture_callback(in_data, _frame_count, _time_info, _status_flags):
            """Audio capture callback - runs in background thread."""
            if self._is_capturing:
                audio_base64 = base64.b64encode(in_data).decode("utf-8")
                asyncio.run_coroutine_threadsafe(
                    self.connection.input_audio_buffer.append(audio=audio_base64), 
                    self.loop
                )
            return (None, pyaudio.paContinue)
        
        if self.input_stream:
            return
        
        try:
            self.input_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_capture_callback,
            )
            self._is_capturing = True
            logger.info("Started audio capture")
        except Exception:
            logger.exception("Failed to start audio capture")
            raise
    
    def stop_capture(self):
        """Stop audio capture."""
        self._is_capturing = False
    
    def resume_capture(self):
        """Resume audio capture."""
        self._is_capturing = True
    
    def start_playback(self):
        """Initialize audio playback system."""
        if self.output_stream:
            return
        
        remaining = bytes()
        
        def _playback_callback(_in_data, frame_count, _time_info, _status_flags):
            """Audio playback callback - runs in background thread."""
            nonlocal remaining
            frame_count_bytes = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
            
            out = remaining[:frame_count_bytes]
            remaining = remaining[frame_count_bytes:]
            
            while len(out) < frame_count_bytes:
                try:
                    packet = self.playback_queue.get_nowait()
                except queue.Empty:
                    out = out + bytes(frame_count_bytes - len(out))
                    break
                except Exception:
                    logger.exception("Error in audio playback")
                    raise
                
                if not packet or not packet.data:
                    logger.info("End of playback queue.")
                    break
                
                if packet.seq_num < self.playback_base:
                    # Skip packet (interrupted)
                    if len(remaining) > 0:
                        remaining = bytes()
                    continue
                
                num_to_take = frame_count_bytes - len(out)
                out = out + packet.data[:num_to_take]
                remaining = packet.data[num_to_take:]
            
            if len(out) >= frame_count_bytes:
                return (out, pyaudio.paContinue)
            else:
                return (out, pyaudio.paComplete)
        
        try:
            self.output_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=_playback_callback
            )
            logger.info("Audio playback system ready")
        except Exception:
            logger.exception("Failed to initialize audio playback")
            raise
    
    def _get_and_increase_seq_num(self):
        """Get current sequence number and increment."""
        seq = self.next_seq_num
        self.next_seq_num += 1
        return seq
    
    def queue_audio(self, audio_data: Optional[bytes]) -> None:
        """Queue audio data for playback.
        
        Args:
            audio_data: Raw PCM audio bytes to play, or None to signal end
        """
        self.playback_queue.put(
            AudioProcessor.AudioPlaybackPacket(
                seq_num=self._get_and_increase_seq_num(),
                data=audio_data
            )
        )
    
    def skip_pending_audio(self):
        """Skip/interrupt current audio in playback queue."""
        self.playback_base = self._get_and_increase_seq_num()
        logger.debug("Skipped pending audio playback")
    
    def shutdown(self):
        """Clean up audio resources."""
        self._is_capturing = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        logger.info("Stopped audio capture")
        
        if self.output_stream:
            self.skip_pending_audio()
            self.queue_audio(None)
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        logger.info("Stopped audio playback")
        
        if self.audio:
            self.audio.terminate()
        logger.info("Audio processor cleaned up")


class AzureVoiceLiveClient:
    """
    Azure Voice Live API client with configurable VAD modes and function calling.
    
    Supports:
    - Multiple VAD modes: server_vad, azure_semantic_vad, push-to-talk
    - Azure Neural voices and OpenAI voices
    - Function calling with async job handling
    - Noise suppression and echo cancellation
    - End-of-utterance detection
    """
    
    def __init__(self, config: dict, yaml_config: dict = None):
        """Initialize the Azure Voice Live client.
        
        Args:
            config: Environment configuration (Azure endpoints, etc.)
            yaml_config: YAML configuration for VAD modes and parameters (optional)
        """
        self.config = config
        self.yaml_config = yaml_config or {}
        
        self._endpoint = config.get("AZURE_VOICELIVE_ENDPOINT")
        self._model = config.get("AZURE_VOICELIVE_MODEL", "gpt-4o")
        self._api_key = config.get("AZURE_VOICELIVE_API_KEY")
        self._use_token_credential = config.get("USE_TOKEN_CREDENTIAL", False)
        
        if not self._endpoint:
            raise VoiceLiveClientError("Missing AZURE_VOICELIVE_ENDPOINT configuration variable.")
        
        # Connection and audio processor
        self.connection = None
        self.audio_processor: Optional[AudioProcessor] = None
        
        # State management
        self._keyboard_listener = None
        self._is_recording = False
        self._quit_event = asyncio.Event()
        self._space_released_event = asyncio.Event()
        self._session_ready = False
        self._conversation_started = False
        self._active_response = False
        self._response_api_done = False
        self._current_response_text = ""  # Buffer for text responses
        
        # VAD mode configuration
        self._vad_mode = self.yaml_config.get('turn_detection', {}).get('type')
        self._is_push_to_talk = self._vad_mode is None or self._vad_mode == 'disabled'
        
        # Load system prompt template
        self._system_prompt = self._load_system_prompt()
        
        # Function calling configuration
        self._function_call_attempt_counter = 0
        self._async_completion_delay = 10.0
        self._pending_function_call: Optional[Dict[str, Any]] = None
        self._pending_jobs = {}  # Track pending async jobs: {job_id: result}
        
        # Display configuration info
        self._display_config_info()
    
    def _display_config_info(self):
        """Display configuration information at startup."""
        if self.yaml_config.get('scenario'):
            self._display_message("status_info", f"Mode: {self.yaml_config['scenario']}")
        if self.yaml_config.get('description'):
            self._display_message("status_info", f"Description: {self.yaml_config['description']}")
        self._display_message("status_info", f"Model: {self._model}")
        self._display_message("status_info", f"VAD Mode: {self._vad_mode or 'Push-to-Talk'}")
        voice_config = self.yaml_config.get('voice', {})
        voice_name = voice_config.get('name', 'en-US-Ava:DragonHDLatestNeural')
        self._display_message("status_info", f"Voice: {voice_name}")
        self._display_message("status_info", f"Sample rate: {PYAUDIO_RATE}Hz")
    
    def _load_system_prompt(self) -> str:
        """Load and render system prompt from Jinja2 template.
        
        Returns:
            str: Rendered system prompt
        """
        try:
            template_dir = Path(__file__).parent / "templates"
            env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
            template = env.get_template('system_prompt.j2')
            context = self.yaml_config.get('prompt_context', {})
            rendered_prompt = template.render(**context)
            logger.debug(f"Loaded system prompt from template ({len(rendered_prompt)} chars)")
            return rendered_prompt
        except Exception as e:
            logger.warning(f"Failed to load system prompt template: {e}")
            return "You are a helpful AI assistant. Respond naturally and conversationally."
    
    def _display_message(self, m_type: str, content: str = "", end: str = '\n', flush: bool = False):
        """Handles all console output with emojis."""
        prefixes = {
            "status_connect": "🔌 CONNECT:", "status_ok": "✅ SUCCESS:", "status_info": "ℹ️  INFO:",
            "status_warn": "⚠️  WARNING:", "status_error": "❌ ERROR:", "mic_ready": "🎙️  MIC:",
            "mic_rec_start": "🔴 REC:", "mic_rec_stop": "⏹️  STOP:", "audio_send": "📤 SEND:",
            "audio_sent": "📬 SENT:", "ai_thinking": "🤔 AI:", "ai_resp_text": "💬 RESP:",
            "ai_resp_audio": "🔊 AUDIO:", "ai_transcript": "📝 TRANS:", "user_transcription": "🎤 USER:",
            "user_action": "👉 ACTION:", "system_event": "⚙️  SYS:", "quit": "👋 BYE:", 
            "function_call": "📞 FUNC:"
        }
        prefix = prefixes.get(m_type, "➡️  INFO:")
        force_flush = flush or m_type not in ["ai_resp_text_delta", "ai_transcript_delta"]
        
        if m_type == "ai_resp_text_delta":
            print(content, end=end, flush=flush)
        elif m_type == "ai_transcript_delta":
            print(f"{prefix} {content}", end=end, flush=flush)
        else:
            print(f"{prefix} {content}", end=end, flush=force_flush)
    
    def _on_keyboard_press(self, key):
        """Handle keyboard press events."""
        if key == keyboard.Key.space:
            if self._is_push_to_talk and not self._is_recording:
                self._display_message("mic_rec_start", "Recording...")
                self._is_recording = True
                if self.audio_processor:
                    self.audio_processor.resume_capture()
            elif not self._is_push_to_talk:
                # In VAD modes, spacebar can trigger manual response
                self._handle_user_interruption()
        elif hasattr(key, 'char') and key.char == 'q':
            self._display_message("quit", "Quit signal received. Stopping...")
            if not self._quit_event.is_set():
                self._quit_event.set()
            if self._keyboard_listener:
                return False
    
    def _on_keyboard_release(self, key):
        """Handle keyboard release events."""
        if key == keyboard.Key.space and self._is_push_to_talk:
            if self._is_recording:
                self._display_message("mic_rec_stop", "Recording stopped.")
                self._is_recording = False
                if self.audio_processor:
                    self.audio_processor.stop_capture()
                self._space_released_event.set()
    
    def _handle_user_interruption(self):
        """Handle user interruption of AI speech."""
        if self.audio_processor:
            self.audio_processor.skip_pending_audio()
        self._display_message("system_event", "🛑 Audio playback interrupted")
    
    def _construct_voice_config(self):
        """Construct voice configuration from YAML config.
        
        Returns:
            Voice configuration object or string for the SDK
        """
        from azure.ai.voicelive.models import AzureStandardVoice
        
        voice_config = self.yaml_config.get('voice', {})
        voice_name = voice_config.get('name', 'en-US-Ava:DragonHDLatestNeural')
        voice_type = voice_config.get('type', 'azure-standard')
        
        if voice_type in ['azure-standard', 'azure-custom']:
            config = AzureStandardVoice(name=voice_name)
            # Add optional parameters
            if 'temperature' in voice_config:
                config.temperature = voice_config['temperature']
            if 'rate' in voice_config:
                config.rate = voice_config['rate']
            if 'locale' in voice_config:
                config.locale = voice_config['locale']
            if 'prefer_locales' in voice_config:
                config.prefer_locales = voice_config['prefer_locales']
            if 'style' in voice_config:
                config.style = voice_config['style']
            if 'pitch' in voice_config:
                config.pitch = voice_config['pitch']
            if 'volume' in voice_config:
                config.volume = voice_config['volume']
            return config
        else:
            # OpenAI voice (alloy, echo, fable, onyx, nova, shimmer)
            return voice_name
    
    def _construct_turn_detection_config(self):
        """Construct turn detection configuration from YAML config.
        
        Returns:
            Turn detection configuration object for the SDK
        """
        from azure.ai.voicelive.models import (
            ServerVad, 
            AzureSemanticVad,
            AzureSemanticVadMultilingual,
            EouDetection,
            EouThresholdLevel
        )
        
        td_config = self.yaml_config.get('turn_detection', {})
        td_type = td_config.get('type')
        
        if td_type is None or td_type == 'disabled':
            return None  # Push-to-talk mode
        
        # Common parameters
        threshold = td_config.get('threshold', 0.5)
        prefix_padding_ms = td_config.get('prefix_padding_ms', 300)
        silence_duration_ms = td_config.get('silence_duration_ms', 500)
        create_response = td_config.get('create_response', True)
        
        if td_type == 'server_vad':
            return ServerVad(
                threshold=threshold,
                prefix_padding_ms=prefix_padding_ms,
                silence_duration_ms=silence_duration_ms,
            )
        
        elif td_type == 'azure_semantic_vad':
            config = AzureSemanticVad(
                threshold=threshold,
                prefix_padding_ms=prefix_padding_ms,
                silence_duration_ms=silence_duration_ms,
            )
            
            # Add optional parameters
            if td_config.get('remove_filler_words'):
                config.remove_filler_words = True
            if td_config.get('interrupt_response'):
                config.interrupt_response = True
            
            # End of utterance detection
            eou_config = td_config.get('end_of_utterance_detection', {})
            if eou_config:
                # Map threshold_level string to enum
                threshold_level_str = eou_config.get('threshold_level', 'default')
                if threshold_level_str == 'default':
                    threshold_level = EouThresholdLevel.DEFAULT
                elif threshold_level_str == 'conservative':
                    threshold_level = EouThresholdLevel.CONSERVATIVE
                elif threshold_level_str == 'fast':
                    threshold_level = EouThresholdLevel.FAST
                else:
                    threshold_level = EouThresholdLevel.DEFAULT
                
                config.end_of_utterance_detection = EouDetection(
                    model=eou_config.get('model', 'semantic_detection_v1'),
                    threshold_level=threshold_level,
                    timeout_ms=eou_config.get('timeout_ms', 1000)
                )
            
            return config
        
        elif td_type == 'azure_semantic_vad_multilingual':
            config = AzureSemanticVadMultilingual(
                threshold=threshold,
                prefix_padding_ms=prefix_padding_ms,
                silence_duration_ms=silence_duration_ms,
            )
            
            if 'languages' in td_config:
                config.languages = td_config['languages']
            
            return config
        
        # Default to server VAD
        return ServerVad(
            threshold=threshold,
            prefix_padding_ms=prefix_padding_ms,
            silence_duration_ms=silence_duration_ms,
        )
    
    def _construct_function_tools(self):
        """Construct function tools for the session.
        
        Returns:
            List of FunctionTool objects
        """
        from azure.ai.voicelive.models import FunctionTool
        
        return [
            FunctionTool(
                name="get_destination_info",
                description="Retrieves vacation information, packages, hotels, and activities for a specific destination. IMPORTANT: Call this function EVERY TIME the customer mentions a new destination OR a new type of vacation for any destination.",
                parameters={
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": "The destination name (e.g., 'Paris', 'Bali', 'Tokyo', 'Prague')"
                        },
                        "vacation_type": {
                            "type": "string",
                            "description": "Type of vacation the customer is interested in (e.g., 'beach vacation', 'city sightseeing', 'romantic getaway')"
                        },
                        "conversation_summary": {
                            "type": "string",
                            "description": "Brief summary of what the customer has shared so far"
                        },
                        "user_mood": {
                            "type": "string",
                            "description": "Describe the customer's emotional state based on the conversation"
                        },
                        "happiness_score": {
                            "type": "number",
                            "description": "Customer's emotional state score: -1.0 (angry) to 1.0 (happy), 0.0 is neutral",
                            "minimum": -1.0,
                            "maximum": 1.0
                        }
                    },
                    "required": ["destination", "vacation_type", "conversation_summary", "user_mood", "happiness_score"]
                }
            )
        ]
    
    def _construct_transcription_config(self):
        """Construct transcription configuration from YAML config.
        
        Returns:
            AudioInputTranscriptionOptions or None if transcription is disabled
        """
        from azure.ai.voicelive.models import AudioInputTranscriptionOptions
        
        transcription_config = self.yaml_config.get('transcription', {})
        
        if not transcription_config.get('enabled', False):
            return None
        
        model = transcription_config.get('model', 'whisper-1')
        language = transcription_config.get('language', None)
        
        # Create transcription options
        options = AudioInputTranscriptionOptions(model=model)
        
        # Add optional language parameter if specified
        if language:
            options.language = language
        
        logger.info(f"Transcription enabled with model: {model}, language: {language or 'auto-detect'}")
        return options
    
    async def _setup_session(self):
        """Configure the Voice Live session for audio conversation with function tools."""
        from azure.ai.voicelive.models import (
            RequestSession,
            Modality,
            InputAudioFormat,
            OutputAudioFormat,
            AudioEchoCancellation,
            AudioNoiseReduction,
            AudioInputTranscriptionOptions,
            ToolChoiceLiteral,
        )
        
        logger.info("Setting up Voice Live session...")
        
        # Build session configuration
        voice_config = self._construct_voice_config()
        turn_detection = self._construct_turn_detection_config()
        function_tools = self._construct_function_tools()
        
        # Get audio processing options from config
        audio_config = self.yaml_config.get('audio', {})
        use_echo_cancellation = audio_config.get('echo_cancellation', True)
        use_noise_reduction = audio_config.get('noise_reduction', True)
        noise_reduction_type = audio_config.get('noise_reduction_type', 'azure_deep_noise_suppression')
        
        # Build session configuration
        session_config = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=self._system_prompt,
            voice=voice_config,
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            tools=function_tools,
            tool_choice=ToolChoiceLiteral.AUTO,
            input_audio_transcription=self._construct_transcription_config(),
        )
        
        # Add turn detection if not push-to-talk
        if turn_detection:
            session_config.turn_detection = turn_detection
        
        # Add echo cancellation if enabled
        if use_echo_cancellation:
            session_config.input_audio_echo_cancellation = AudioEchoCancellation()
        
        # Add noise reduction if enabled
        if use_noise_reduction:
            session_config.input_audio_noise_reduction = AudioNoiseReduction(type=noise_reduction_type)
        
        # Update session
        await self.connection.session.update(session=session_config)
        logger.info("Session configuration sent")
    
    async def _process_events(self):
        """Process events from the Voice Live connection."""
        from azure.ai.voicelive.models import ServerEventType, ItemType
        
        try:
            async for event in self.connection:
                if self._quit_event.is_set():
                    break
                
                await self._handle_event(event)
                
        except Exception:
            logger.exception("Error processing events")
            raise
    
    async def _handle_event(self, event):
        """Handle different types of events from Voice Live.
        
        Args:
            event: Server event from the Voice Live connection
        """
        from azure.ai.voicelive.models import ServerEventType, ItemType
        
        logger.debug(f"Received event: {event.type}")
        
        if event.type == ServerEventType.SESSION_UPDATED:
            logger.info(f"Session ready: {event.session.id}")
            self._session_ready = True
            
            # Proactive greeting if enabled
            if not self._conversation_started and self.yaml_config.get('proactive_greeting', True):
                self._conversation_started = True
                logger.info("Sending proactive greeting request")
                try:
                    await self.connection.response.create()
                except Exception:
                    logger.exception("Failed to send proactive greeting request")
            
            # Start audio capture once session is ready
            if self.audio_processor:
                self.audio_processor.start_capture()
                if self._is_push_to_talk:
                    self.audio_processor.stop_capture()  # Wait for spacebar
                    self._display_message("mic_ready", "Press SPACE to talk, 'q' to quit")
                else:
                    self._display_message("mic_ready", "Voice activity detection enabled - speak naturally...")
        
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
            logger.info("User started speaking - stopping playback")
            self._display_message("system_event", "🎤 Listening...")
            
            if self.audio_processor:
                self.audio_processor.skip_pending_audio()
            
            # Cancel active response if any
            if self._active_response and not self._response_api_done:
                try:
                    await self.connection.response.cancel()
                    logger.debug("Cancelled in-progress response due to barge-in")
                except Exception as e:
                    if "no active response" in str(e).lower():
                        logger.debug("Cancel ignored - response already completed")
                    else:
                        logger.warning(f"Cancel failed: {e}")
        
        elif event.type == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STOPPED:
            logger.info("User stopped speaking")
            self._display_message("system_event", "🤔 Processing...")
        
        elif event.type == ServerEventType.RESPONSE_CREATED:
            logger.info("Assistant response created")
            self._active_response = True
            self._response_api_done = False
            self._current_response_text = ""  # Reset text buffer
        
        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
            # Display user input transcription
            if hasattr(event, 'transcript') and event.transcript:
                self._display_message("user_transcription", f"[User said: {event.transcript}]")
                logger.info(f"User audio transcription: {event.transcript}")
        
        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA:
            # Handle streaming transcription (partial results)
            if hasattr(event, 'delta') and event.delta:
                logger.debug(f"Transcription delta: {event.delta}")
        
        elif event.type == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED:
            # Handle transcription failure
            error_msg = "Unknown error"
            if hasattr(event, 'error'):
                if hasattr(event.error, 'message'):
                    error_msg = event.error.message
                elif isinstance(event.error, dict):
                    error_msg = event.error.get('message', str(event.error))
            logger.error(f"Transcription failed: {error_msg}")
            self._display_message("status_error", f"[Transcription failed: {error_msg}]")
        
        elif event.type == ServerEventType.RESPONSE_AUDIO_DELTA:
            logger.debug("Received audio delta")
            if self.audio_processor and event.delta:
                self.audio_processor.queue_audio(event.delta)
        
        elif event.type == ServerEventType.RESPONSE_AUDIO_DONE:
            logger.info("Assistant finished speaking")
            self._display_message("ai_resp_audio", "Audio response complete")
        
        elif event.type == ServerEventType.RESPONSE_DONE:
            logger.info("Response complete")
            self._active_response = False
            self._response_api_done = True
            self._display_message("status_ok", "--- End of AI Response ---")
            
            # Execute pending function call if arguments are ready
            if self._pending_function_call and "arguments" in self._pending_function_call:
                await self._execute_function_call(self._pending_function_call)
                self._pending_function_call = None
        
        elif event.type == ServerEventType.ERROR:
            msg = event.error.message if hasattr(event.error, 'message') else str(event.error)
            if "no active response" in msg.lower():
                logger.debug(f"Benign error: {msg}")
            else:
                logger.error(f"Voice Live error: {msg}")
                self._display_message("status_error", f"Error: {msg}")
        
        elif event.type == ServerEventType.CONVERSATION_ITEM_CREATED:
            logger.debug(f"Conversation item created: {event.item.id}")
            
            if event.item.type == ItemType.FUNCTION_CALL:
                function_call_item = event.item
                self._pending_function_call = {
                    "name": function_call_item.name,
                    "call_id": function_call_item.call_id,
                    "previous_item_id": function_call_item.id
                }
                self._display_message("system_event", f"📞 Function call: {function_call_item.name}")
                logger.info(f"Function call detected: {function_call_item.name}")
        
        elif event.type == ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE:
            if self._pending_function_call and event.call_id == self._pending_function_call.get("call_id"):
                logger.info(f"Function arguments received: {event.arguments}")
                self._pending_function_call["arguments"] = event.arguments
                
                # Parse and display arguments
                try:
                    args = json.loads(event.arguments) if isinstance(event.arguments, str) else event.arguments
                    self._display_message("system_event", f"   Destination: {args.get('destination', 'N/A')}")
                    self._display_message("system_event", f"   Vacation Type: {args.get('vacation_type', 'N/A')}")
                    self._display_message("system_event", f"   Summary: {args.get('conversation_summary', 'N/A')}")
                    self._display_message("system_event", f"   Mood: {args.get('user_mood', 'N/A')}")
                    self._display_message("system_event", f"   Happiness: {args.get('happiness_score', 0):.2f}")
                except Exception as e:
                    logger.warning(f"Could not parse function arguments: {e}")
        
        elif event.type == ServerEventType.RESPONSE_TEXT_DELTA:
            # Text delta from response
            if hasattr(event, 'delta') and event.delta:
                # Print prefix on first delta
                if not self._current_response_text:
                    self._display_message("ai_resp_text", "")
                    print("💬 RESP: ", end="", flush=True)
                self._current_response_text += event.delta
                print(event.delta, end="", flush=True)
        
        elif event.type == ServerEventType.RESPONSE_TEXT_DONE:
            if self._current_response_text:
                print()  # New line after text response
                logger.debug(f"Text response: {self._current_response_text}")
        
        else:
            logger.debug(f"Unhandled event type: {event.type}")
    
    async def _execute_function_call(self, function_call_info: Dict[str, Any]):
        """Execute a function call with async job simulation matching original implementation.
        
        Args:
            function_call_info: Dictionary with function call details
        """
        from azure.ai.voicelive.models import FunctionCallOutputItem
        
        function_name = function_call_info["name"]
        call_id = function_call_info["call_id"]
        previous_item_id = function_call_info["previous_item_id"]
        arguments = function_call_info["arguments"]
        
        try:
            # Parse arguments
            if isinstance(arguments, str):
                args = json.loads(arguments)
            else:
                args = arguments if isinstance(arguments, dict) else {}
            
            logger.info(f"Executing function: {function_name} with args: {args}")
            
            if function_name == "get_destination_info":
                # Increment attempt counter for round-robin
                self._function_call_attempt_counter += 1
                attempt_num = self._function_call_attempt_counter
                is_odd_attempt = attempt_num % 2 == 1
                
                destination = args.get('destination', 'Unknown')
                vacation_type = args.get('vacation_type', 'general vacation')
                happiness = args.get('happiness_score', 0.0)
                
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
                    
                    # Send function result back
                    function_output = FunctionCallOutputItem(
                        call_id=call_id,
                        output=json.dumps(result, ensure_ascii=False)
                    )
                    await self.connection.conversation.item.create(
                        previous_item_id=previous_item_id,
                        item=function_output
                    )
                    
                    # Request new response
                    await self.connection.response.create()
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
                    
                    # Send pending result back
                    function_output = FunctionCallOutputItem(
                        call_id=call_id,
                        output=json.dumps(pending_result, ensure_ascii=False)
                    )
                    await self.connection.conversation.item.create(
                        previous_item_id=previous_item_id,
                        item=function_output
                    )
                    
                    # Trigger response so model can acknowledge
                    await self.connection.response.create()
                    
                    self._display_message("system_event", f"⏳ Function call returned PENDING status (job_id: {job_id})")
                    self._display_message("system_event", f"📋 Starting background async job (will complete in {self._async_completion_delay}s)")
                    
                    # Start background task to simulate async completion
                    asyncio.create_task(
                        self._handle_async_job_completion(job_id, destination, vacation_type, happiness)
                    )
            else:
                logger.warning(f"Unknown function: {function_name}")
        
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
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
            if self._active_response:
                self._display_message("system_event", "⚡ Cancelling active response to deliver async result")
                try:
                    await self.connection.response.cancel()
                    # Flush audio playback buffer to stop filler speech immediately
                    if self.audio_processor:
                        self.audio_processor.skip_pending_audio()
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
            from azure.ai.voicelive.models import SystemMessageItem, InputTextContentPart
            
            system_message = SystemMessageItem(
                content=[
                    InputTextContentPart(
                        text=f"ASYNC JOB COMPLETED: Job {job_id} has finished. The destination information is now available: {json.dumps(result, ensure_ascii=False)}"
                    )
                ]
            )
            
            await self.connection.conversation.item.create(item=system_message)
            
            # Trigger response generation so model can speak the result
            await self.connection.response.create()
            
            self._display_message("system_event", f"📢 Triggered response for async result (job_id: {job_id})")
            
        except Exception as e:
            logger.error(f"Error in async job completion handler: {e}")
            self._display_message("status_error", f"Async job error: {e}")
    
    async def _handle_push_to_talk_completion(self):
        """Handle push-to-talk recording completion."""
        while not self._quit_event.is_set() and self.connection:
            await self._space_released_event.wait()
            self._space_released_event.clear()
            
            if self._quit_event.is_set() or not self.connection:
                break
            
            # Commit audio buffer and request response
            self._display_message("audio_send", "Submitting audio...")
            
            try:
                await self.connection.input_audio_buffer.commit()
                logger.debug("Committed input audio buffer")
                
                await self.connection.response.create()
                logger.debug("Created response request")
                
                self._display_message("ai_thinking", "Waiting for AI response...")
            except Exception as e:
                logger.error(f"Error handling recording completion: {e}")
                self._display_message("status_error", f"Error: {e}")
    
    async def run(self):
        """Run the Voice Live client."""
        from azure.ai.voicelive.aio import connect
        
        self._display_message("status_info", "Starting Voice Live Client...")
        
        try:
            # Set up credential
            if self._use_token_credential:
                credential = DefaultAzureCredential()
                self._display_message("status_info", "Using Azure token credential")
            else:
                if not self._api_key:
                    raise VoiceLiveClientError("No API key provided. Set AZURE_VOICELIVE_API_KEY or use token credential.")
                credential = AzureKeyCredential(self._api_key)
                self._display_message("status_info", "Using API key credential")
            
            self._display_message("status_connect", f"Connecting to Voice Live API...")
            self._display_message("status_info", f"Endpoint: {self._endpoint}")
            self._display_message("status_info", f"Model: {self._model}")
            
            # Connect to Voice Live
            async with connect(
                endpoint=self._endpoint,
                credential=credential,
                model=self._model,
            ) as connection:
                self.connection = connection
                self._display_message("status_ok", "Connected to Voice Live API!")
                
                # Get event loop for audio processor
                loop = asyncio.get_event_loop()
                
                # Initialize audio processor
                self.audio_processor = AudioProcessor(connection, loop)
                self.audio_processor.start_playback()
                
                # Configure session
                await self._setup_session()
                
                # Start keyboard listener
                self._keyboard_listener = keyboard.Listener(
                    on_press=self._on_keyboard_press,
                    on_release=self._on_keyboard_release
                )
                self._keyboard_listener.start()
                
                # Create tasks
                tasks = [
                    asyncio.create_task(self._process_events()),
                ]
                
                # Add push-to-talk handler if needed
                if self._is_push_to_talk:
                    tasks.append(asyncio.create_task(self._handle_push_to_talk_completion()))
                
                self._display_message("status_ok", "Voice assistant ready!")
                print("\n" + "=" * 60)
                print("🎤 VOICE LIVE ASSISTANT READY")
                if self._is_push_to_talk:
                    print("  Press SPACE to record, release to send")
                else:
                    print("  Speak naturally - VAD will detect speech")
                print("  Press 'q' to quit")
                print("=" * 60 + "\n")
                
                # Wait for tasks
                try:
                    await asyncio.gather(*tasks)
                except asyncio.CancelledError:
                    pass
                
        except Exception as e:
            logger.exception("Error in Voice Live client")
            self._display_message("status_error", f"Fatal error: {e}")
            raise
        finally:
            # Cleanup
            if self.audio_processor:
                self.audio_processor.shutdown()
            if self._keyboard_listener:
                self._keyboard_listener.stop()
            self._display_message("quit", "Voice Live client stopped. Goodbye!")


async def main():
    """Main application function with command line argument support."""
    
    parser = argparse.ArgumentParser(
        description='Azure Voice Live API Client with configurable VAD modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                                        # Use default configuration
  python main.py -f configs/server-vad.yaml            # Use server VAD
  python main.py -f configs/azure-semantic-vad.yaml    # Use Azure semantic VAD
  python main.py -f configs/push-to-talk.yaml          # Use push-to-talk mode
  python main.py --use-token-credential                # Use Azure AD authentication

Available configurations:
  - configs/push-to-talk.yaml: Manual push-to-talk mode (spacebar control)
  - configs/server-vad.yaml: Server-side voice activity detection
  - configs/azure-semantic-vad.yaml: Azure semantic voice activity detection
  - configs/azure-semantic-vad-multilingual.yaml: Multilingual semantic VAD
  - configs/gpt-realtime.yaml: GPT Realtime model with HD voice
        '''
    )
    parser.add_argument(
        '-f', '--config-file',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--use-token-credential',
        action='store_true',
        help='Use Azure AD token credential instead of API key'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Override model from config (e.g., gpt-4o, gpt-realtime)'
    )
    parser.add_argument(
        '--voice',
        type=str,
        help='Override voice name from config'
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load YAML configuration
    yaml_config = {}
    if args.config_file:
        config_path = Path(args.config_file)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}")
    
    # Apply command-line overrides
    if args.model:
        yaml_config['model'] = args.model
    if args.voice:
        if 'voice' not in yaml_config:
            yaml_config['voice'] = {}
        yaml_config['voice']['name'] = args.voice
    
    # Build environment configuration
    app_config = {
        "AZURE_VOICELIVE_ENDPOINT": os.environ.get("AZURE_VOICELIVE_ENDPOINT"),
        "AZURE_VOICELIVE_MODEL": yaml_config.get('model', os.environ.get("AZURE_VOICELIVE_MODEL", "gpt-4o")),
        "AZURE_VOICELIVE_API_KEY": os.environ.get("AZURE_VOICELIVE_API_KEY"),
        "USE_TOKEN_CREDENTIAL": args.use_token_credential or os.environ.get("USE_TOKEN_CREDENTIAL", "").lower() == "true",
    }
    
    # Validate required configuration
    if not app_config["AZURE_VOICELIVE_ENDPOINT"]:
        print("❌ Error: AZURE_VOICELIVE_ENDPOINT environment variable is required")
        print("Set it in .env file or environment")
        return
    
    if not app_config["USE_TOKEN_CREDENTIAL"] and not app_config["AZURE_VOICELIVE_API_KEY"]:
        print("❌ Error: Either AZURE_VOICELIVE_API_KEY or --use-token-credential is required")
        return
    
    # Create and run client
    client = AzureVoiceLiveClient(config=app_config, yaml_config=yaml_config)
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    except Exception as e:
        logger.exception("Fatal error")
        print(f"❌ Fatal error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
