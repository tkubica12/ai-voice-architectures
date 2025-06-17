# Azure OpenAI Realtime Voice Chat Client

A comprehensive voice chat client for Azure OpenAI's Realtime API with support for multiple Voice Activity Detection (VAD) modes and configurable parameters.

## Features

- **Multiple VAD Modes**: Push-to-talk, Server VAD, and Semantic VAD
- **YAML Configuration**: Easily configurable via YAML files
- **Azure Integration**: Uses Azure Managed Identity for authentication
- **High-Quality Audio**: 24kHz PCM audio processing
- **Real-time Streaming**: Continuous audio streaming with minimal latency

## Prerequisites

- Python 3.12+
- Azure OpenAI service with Realtime API access
- Microphone and speakers
- UV package manager

## Installation

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI configuration
```

## Environment Variables

Create a `.env` file with the following variables:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
OPENAI_API_VERSION=2024-10-01-preview
LOG_LEVEL=INFO
```

## Usage

### Basic Usage (Default Push-to-Talk)

```bash
uv run python main.py
```

### Using Configuration Files

```bash
# Server VAD mode
uv run python main.py -f configs/server-vad.yaml

# Semantic VAD mode
uv run python main.py -f configs/semantic-vad.yaml

# High sensitivity server VAD
uv run python main.py -f configs/server-vad-sensitive.yaml

# Low sensitivity server VAD for noisy environments
uv run python main.py -f configs/server-vad-robust.yaml
```

## VAD Modes

### 1. Push-to-Talk Mode (`realtime-push-to-talk`)

- **Control**: Manual control using spacebar
- **Best for**: Precise control, avoiding false triggers
- **Usage**: Press and hold spacebar to record, release to send

### 2. Server VAD Mode (`realtime-server-vad`)

- **Control**: Server-side voice activity detection
- **Best for**: Natural conversation flow
- **Features**: Automatic speech detection based on silence periods
- **Configurable Parameters**:
  - `threshold`: Sensitivity (0.0-1.0, higher = less sensitive)
  - `prefix_padding_ms`: Audio to include before speech (ms)
  - `silence_duration_ms`: Silence duration to detect speech end (ms)
  - `create_response`: Automatically create responses
  - `interrupt_response`: Allow interrupting AI responses

### 3. Semantic VAD Mode (`realtime-semantic-vad`)

- **Control**: AI-powered semantic detection
- **Best for**: Complex conversations, avoiding mid-sentence interruptions
- **Features**: Uses semantic analysis to determine when user has finished speaking

## Configuration Files

Configuration files are located in the `configs/` directory:

- `push-to-talk.yaml`: Manual control mode
- `server-vad.yaml`: Standard server VAD
- `semantic-vad.yaml`: Semantic VAD mode
- `server-vad-sensitive.yaml`: High sensitivity for quiet environments
- `server-vad-robust.yaml`: Low sensitivity for noisy environments

### Example Configuration

```yaml
scenario: "realtime-server-vad"
description: "Server-side voice activity detection"

openai:
  model: "gpt-4o-realtime-preview"
  voice: "alloy"
  temperature: 0.8
  instructions: "You are a helpful AI assistant."

turn_detection:
  type: "server_vad"
  threshold: 0.5
  prefix_padding_ms: 300
  silence_duration_ms: 500
  create_response: true
  interrupt_response: true

audio:
  sample_rate: 24000
  channels: 1
  chunk_size: 1024
  format: "pcm16"

ui:
  show_audio_messages: false
  show_system_messages: true
  show_debug_messages: false
```

## Available Voices

- `alloy`: Balanced, neutral voice
- `ash`: Clear, articulate voice
- `ballad`: Warm, conversational voice
- `coral`: Friendly, engaging voice
- `echo`: Deep, resonant voice
- `sage`: Wise, authoritative voice
- `shimmer`: Bright, energetic voice
- `verse`: Smooth, melodic voice

## Controls

### Push-to-Talk Mode
- **Spacebar**: Press and hold to record, release to send
- **Q**: Quit application

### VAD Modes
- **Speak naturally**: Voice is detected automatically
- **Spacebar**: Manually trigger response (optional)
- **Q**: Quit application

## Troubleshooting

### Audio Issues
- Ensure microphone permissions are granted
- Check audio device settings
- Try different buffer sizes for better performance

### Authentication Issues
- Verify Azure credentials are properly configured
- Ensure Managed Identity has appropriate permissions
- Check Azure OpenAI resource access

### Configuration Issues
- Validate YAML syntax
- Ensure all required fields are present
- Check file paths are correct

## Advanced Configuration

### Custom VAD Parameters

For noisy environments, increase threshold and silence duration:
```yaml
turn_detection:
  type: "server_vad"
  threshold: 0.7  # Less sensitive
  silence_duration_ms: 800  # Longer silence before cutoff
```

For quiet environments or soft-spoken users:
```yaml
turn_detection:
  type: "server_vad"
  threshold: 0.3  # More sensitive
  silence_duration_ms: 300  # Quicker response
```

### Logging Configuration

Set the `LOG_LEVEL` environment variable:
- `DEBUG`: Detailed logging
- `INFO`: General information (default)
- `WARNING`: Warnings only
- `ERROR`: Errors only

## Contributing

1. Follow the Python instructions in `.github/instructions/python.instructions.md`
2. Use `uv` for dependency management
3. Add docstrings for all public methods
4. Test with different VAD configurations

## License

This project is licensed under the MIT License - see the LICENSE file for details.
