# Azure Voice Live API Client

Real-time voice assistant using Microsoft's Voice Live API with configurable VAD modes, Azure Neural voices, and function calling capabilities.

## Overview

This implementation provides a voice-first conversational AI experience using Azure's Voice Live API. It mirrors the functionality of the Azure OpenAI Realtime API client but leverages Voice Live's enhanced conversational features:

- **Azure Semantic VAD**: Advanced end-of-utterance detection using AI
- **Noise Suppression**: Azure Deep Noise Suppression for cleaner audio
- **Echo Cancellation**: Server-side echo cancellation
- **Azure Neural Voices**: High-quality Azure TTS voices including HD variants
- **Multiple Model Options**: GPT-4o, GPT-Realtime, GPT-4.1, and more

## Prerequisites

- Python 3.12+
- Azure subscription with Microsoft Foundry resource
- Working microphone and speakers/headphones
- PortAudio library (for PyAudio)

### Windows
```powershell
# PyAudio should install directly via pip
```

### Linux/macOS
```bash
# Debian/Ubuntu
sudo apt-get install -y portaudio19-dev libasound2-dev

# macOS (Homebrew)
brew install portaudio
```

## Installation

```bash
cd src/voicelive
uv sync
```

Or with pip:
```bash
pip install -e .
```

## Configuration

### Environment Variables

Create a `.env` file in the `src/voicelive` directory:

```bash
# Required
AZURE_VOICELIVE_ENDPOINT=https://your-resource.services.ai.azure.com/

# Authentication (choose one)
AZURE_VOICELIVE_API_KEY=your-api-key
# OR use --use-token-credential flag for Azure AD

# Optional
AZURE_VOICELIVE_MODEL=gpt-4o
LOG_LEVEL=INFO
```

### YAML Configuration Files

Configuration files in the `configs/` folder define different modes:

| Config File | Description |
|-------------|-------------|
| `push-to-talk.yaml` | Manual spacebar control |
| `server-vad.yaml` | Server-side voice activity detection |
| `azure-semantic-vad.yaml` | Azure semantic VAD with end-of-utterance detection |
| `azure-semantic-vad-multilingual.yaml` | Multilingual semantic VAD |
| `gpt-realtime.yaml` | GPT Realtime model configuration |
| `gpt-4-1.yaml` | GPT-4.1 model configuration |
| `server-vad-sensitive.yaml` | High sensitivity for quiet environments |
| `server-vad-robust.yaml` | Low sensitivity for noisy environments |

## Usage

### Basic Usage

```bash
# Use default configuration
uv run python main.py

# Use specific configuration
uv run python main.py -f configs/azure-semantic-vad.yaml

# Use Azure AD authentication
uv run python main.py --use-token-credential -f configs/server-vad.yaml
```

### Command Line Options

```
Options:
  -f, --config-file PATH    Path to YAML configuration file
  --use-token-credential    Use Azure AD token instead of API key
  --model MODEL             Override model (e.g., gpt-4o, gpt-realtime)
  --voice VOICE             Override voice name
```

### Interactive Controls

- **SPACE**: Start/stop recording (push-to-talk mode)
- **q**: Quit the application

## Voice Activity Detection Modes

### Push-to-Talk
Manual control using spacebar. Best for noisy environments or when you want precise control.

### Server VAD
Automatic speech detection based on audio volume/silence. Simple and reliable.

### Azure Semantic VAD
AI-powered detection that understands context and natural pauses. Features:
- End-of-utterance detection
- Filler word removal
- Barge-in support

### Azure Semantic VAD Multilingual
Supports multiple languages: English, Spanish, French, Italian, German, Japanese, Portuguese, Chinese, Korean, Hindi.

## Supported Models

| Model | Description |
|-------|-------------|
| `gpt-realtime` | Native audio model, lowest latency |
| `gpt-realtime-mini` | Smaller realtime model |
| `gpt-4o` | GPT-4o with Azure Speech |
| `gpt-4o-mini` | Smaller GPT-4o |
| `gpt-4.1` | Premium GPT-4.1 model |
| `gpt-5` | Latest GPT-5 model |
| `phi4-mm-realtime` | Phi-4 multimodal |

## Voice Options

### Azure Neural Voices
- `en-US-Ava:DragonHDLatestNeural` (HD voice)
- `en-US-AvaNeural`
- `en-US-JennyNeural`
- `en-US-GuyNeural`

### OpenAI Voices
- `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Voice Live Client                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Audio     │  │   Event     │  │    Function         │ │
│  │  Processor  │  │   Handler   │  │    Calling          │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          │                                   │
│                    ┌─────┴─────┐                            │
│                    │ WebSocket │                            │
│                    │Connection │                            │
│                    └─────┬─────┘                            │
└──────────────────────────┼──────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    │ Voice Live  │
                    │     API     │
                    └─────────────┘
```

## Comparison with Azure OpenAI Realtime

| Feature | Voice Live API | OpenAI Realtime API |
|---------|---------------|---------------------|
| Azure Semantic VAD | ✅ | ❌ |
| End-of-utterance detection | ✅ | ❌ |
| Azure Deep Noise Suppression | ✅ | Limited |
| Server Echo Cancellation | ✅ | ❌ |
| Azure Neural Voices | ✅ | ❌ |
| Model flexibility | ✅ Multiple | Limited |
| Fully managed | ✅ | ❌ |

## Troubleshooting

### Connection Issues
- Verify `AZURE_VOICELIVE_ENDPOINT` is correct
- Check network/firewall settings
- Ensure your Azure identity has proper roles (Cognitive Services User, Azure AI User)

### Audio Issues
- Check microphone/speaker connections
- Verify PyAudio installation
- Try different audio devices

### Authentication
- For API key: check `AZURE_VOICELIVE_API_KEY`
- For Azure AD: ensure proper role assignments

### Enable Debug Logging
```bash
LOG_LEVEL=DEBUG uv run python main.py
```

## Related Resources

- [Voice Live API Documentation](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live)
- [Voice Live How-to Guide](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/voice-live-how-to)
- [Azure AI VoiceLive Python SDK](https://learn.microsoft.com/en-us/python/api/overview/azure/ai-voicelive-readme)
- [Function Calling in Voice Live](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-function-calling)
