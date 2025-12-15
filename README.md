# NeMo Speaker Diarization

A complete speaker diarization pipeline using NVIDIA NeMo (via WSL2) or pyannote.audio with optional Whisper transcription.

## Features

- **Speaker Diarization**: Identify "who spoke when" in audio files
- **Transcription**: Optional Whisper-based transcription with speaker labels  
- **Dual Backend**: 
  - WSL2 + NeMo for advanced diarization (Linux)
  - pyannote.audio fallback for Windows
- **Custom Whisper Models**: Support for cached/finetuned models
- **Multi-language**: Support for English, Persian, Arabic, and more

## Quick Start

```python
from nemo_diarization import diarize_and_transcribe

result = diarize_and_transcribe(
    meeting_audio_path="meeting.wav",
    expected_language="en",
    output_transcriptions=True,
    transcriptor_model_path=None  # or path to your Whisper model
)

print(f"Speakers: {result['num_speakers']}")
print(f"Transcription: {result['transcription']}")
```

## Installation

### Windows
```bash
pip install -r requirements_windows.txt
```

### WSL2 (for NeMo)
```bash
wsl -d Ubuntu bash -c "cd /mnt/d/Git_repos/Nemo-diarization/venv_nemo_wsl && bin/pip install -r /mnt/d/Git_repos/Nemo-diarization/requirements_wsl2.txt"
```

## Documentation

See [NEMO_README.md](NEMO_README.md) for complete documentation.

## Project Structure

```
├── nemo_diarization.py          # Main diarization function
├── nemo_diarization_wsl.py      # WSL2 NeMo backend script
├── test_nemo_diarization.ipynb  # Interactive test notebook
├── example_usage.py             # Usage examples
├── requirements_windows.txt     # Windows dependencies
├── requirements_wsl2.txt        # WSL2 dependencies
├── venv_nemo_wsl/              # WSL2 virtual environment
└── outputs/                     # Output files directory
```

## License

MIT
