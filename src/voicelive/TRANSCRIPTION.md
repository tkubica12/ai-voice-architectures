# Input Audio Transcription

The Voice Live API client now supports **input audio transcription** to convert user speech to text. This feature runs asynchronously and provides real-time transcriptions of what users say.

## Features

- **Multiple transcription models** supported:
  - `whisper-1` - OpenAI Whisper model (default)
  - `gpt-4o-transcribe` - GPT-4o transcription
  - `gpt-4o-mini-transcribe` - GPT-4o mini transcription
  - `gpt-4o-transcribe-diarize` - GPT-4o with speaker diarization
  - `azure-speech` - Azure Speech Services (for non-realtime models)

- **Language support**:
  - Single language (BCP-47 format): `en-US`, `cs-CZ`
  - ISO-639-1 format: `en`, `cs`
  - Multi-language auto-detection: `en,cs,de,es`

- **Real-time display**: Transcriptions are displayed in cyan with 🎤 USER: prefix

## Configuration

Add the transcription section to your YAML configuration file:

```yaml
# Input audio transcription configuration
transcription:
  enabled: true
  model: "whisper-1"  # or gpt-4o-transcribe, azure-speech, etc.
  language: "cs"      # Optional: language code or multi-language list
```

### Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `enabled` | boolean | Enable/disable transcription | `false` |
| `model` | string | Transcription model to use | `"whisper-1"` |
| `language` | string | Language code (optional) | auto-detect |

### Language Codes

- **BCP-47 format**: `en-US`, `cs-CZ`, `de-DE`, `es-ES`
- **ISO-639-1 format**: `en`, `cs`, `de`, `es`
- **Multi-language**: `"en,cs,de"` (auto-detects among specified languages)

## Events

The implementation handles three transcription events:

### 1. CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
Received when transcription is complete for user audio:
```python
{
  "type": "conversation.item.input_audio_transcription.completed",
  "item_id": "item_ABC123",
  "content_index": 0,
  "transcript": "Hello, I want to book a vacation"
}
```

### 2. CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA
Streaming partial transcription results (logged but not displayed):
```python
{
  "type": "conversation.item.input_audio_transcription.delta",
  "item_id": "item_ABC123",
  "content_index": 0,
  "delta": "Hello, I want"
}
```

### 3. CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED
Transcription failed (displays error):
```python
{
  "type": "conversation.item.input_audio_transcription.failed",
  "item_id": "item_ABC123",
  "content_index": 0,
  "error": {
    "code": "transcription_error",
    "message": "Audio quality too low"
  }
}
```

## Example Output

When transcription is enabled, you'll see user speech transcribed:

```
🎤 USER: [User said: Hello, I would like to book a vacation to Prague]
💬 RESP: Great! I'd be happy to help you plan a trip to Prague...
```

## Implementation Details

### Code Location

- **Configuration method**: `_construct_transcription_config()` in [main.py](main.py#L560)
- **Event handlers**: Lines 707-729 in [main.py](main.py#L707-L729)
- **Session setup**: [main.py](main.py#L589)

### How It Works

1. **Configuration**: Transcription settings are loaded from YAML config
2. **Session initialization**: `AudioInputTranscriptionOptions` is passed to session.update
3. **Audio processing**: When audio buffer is committed, transcription runs asynchronously
4. **Event delivery**: Server sends transcription events as they become available
5. **Display**: Completed transcriptions are displayed in cyan

### Notes

- Transcription runs **asynchronously** with response generation
- The model consumes audio **natively**, so transcription is supplementary
- Transcription may diverge slightly from model's interpretation
- Treat transcripts as rough guidance, not exact model input

## Example Configurations

### Czech Language Transcription
```yaml
transcription:
  enabled: true
  model: "whisper-1"
  language: "cs"
```

### Multilingual Auto-Detection
```yaml
transcription:
  enabled: true
  model: "whisper-1"
  language: "en,cs,de,es"  # Detects among these languages
```

### GPT-4o Transcription with Diarization
```yaml
transcription:
  enabled: true
  model: "gpt-4o-transcribe-diarize"
  language: "en-US"
```

### Azure Speech (for non-realtime models)
```yaml
transcription:
  enabled: true
  model: "azure-speech"
  language: "cs-CZ"
```

## Disable Transcription

To disable transcription:

```yaml
transcription:
  enabled: false
```

Or simply omit the transcription section from your config file.
