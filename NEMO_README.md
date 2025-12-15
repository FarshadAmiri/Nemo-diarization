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

## Installation

### Windows (pyannote.audio backend)

```bash
# Use your existing venv
D:\venvs\venv_nemo\Scripts\activate
pip install pyannote.audio torch torchaudio openai-whisper
```

### WSL2 (NeMo backend) - Already set up!

The WSL2 environment is already configured with NeMo installed in:
```
D:\Git_repos\Nemo-diarization\venv_nemo_wsl
```

## Quick Start

### Simple Usage

```python
from nemo_diarization import diarize_and_transcribe

result = diarize_and_transcribe(
    meeting_audio_path="meeting.wav",
    expected_language="en",
    output_transcriptions=True,
    transcriptor_model_path=None  # or path to your cached Whisper model
)

print(f"Speakers: {result['num_speakers']}")
print(f"Transcription: {result['transcription']}")
```

### Advanced Usage

```python
from nemo_diarization import process_audio_with_nemo

result = process_audio_with_nemo(
    meeting_audio_path="meeting.wav",
    voice_embeddings_database_path="speakers_db.json",
    expected_language="fa",  # Persian
    output_transcriptions=True,
    transcriptor_model_path=r"D:\models\whisper_persian_finetuned.pt",
    num_speakers=3,  # Expected number of speakers
    use_wsl=True  # Use NeMo in WSL2 (True) or pyannote on Windows (False)
)
```

### Using the Test Notebook

Open `test_nemo_diarization.ipynb` in Jupyter and follow the cells to:
1. Configure paths and parameters
2. Run diarization
3. View results
4. Export to different formats

## Function Parameters

### `process_audio_with_nemo()`

- **meeting_audio_path** (str): Path to your audio file (WAV, MP3, etc.)
- **voice_embeddings_database_path** (str): Path to speaker database JSON (can be empty)
- **expected_language** (str, optional): Language code ('en', 'fa', 'ar', etc.) or None for auto-detect
- **output_transcriptions** (bool): Whether to generate transcriptions (default: False)
- **transcriptor_model_path** (str, optional): Path to your cached Whisper model
- **num_speakers** (int, optional): Expected number of speakers (auto-detect if None)
- **output_dir** (str, optional): Where to save results
- **use_wsl** (bool): Use WSL2 NeMo (True) or Windows pyannote (False)

## Output Format

```python
{
    "segments": [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01"},
        ...
    ],
    "num_speakers": 2,
    "transcription": "Full transcribed text...",
    "speaker_segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "speaker": "SPEAKER_00",
            "text": "Hello, how are you?"
        },
        ...
    ],
    "detected_language": "en",
    "output_files": {
        "diarization": "path/to/diarization_result.json",
        "transcription": "path/to/transcription_with_speakers.json"
    }
}
```

## Whisper Models

You can use your cached Whisper models:

```python
# Small model
transcriptor_model_path = r"D:\models\whisper_small.pt"

# Medium model
transcriptor_model_path = r"D:\models\whisper_medium.pt"

# Persian finetuned
transcriptor_model_path = r"D:\models\whisper_persian_finetuned.pt"
```

Or use default models by setting `transcriptor_model_path=None`.

## Python Version

Python 3.12 is fully supported and recommended. Both backends work with 3.12.

## Troubleshooting

### HuggingFace Token for pyannote

If using pyannote.audio, you may need a HuggingFace token:

```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

### WSL2 Issues

If WSL2 NeMo fails, the system automatically falls back to Windows pyannote.audio.

## Files

- `nemo_diarization.py`: Main function and Windows bridge
- `nemo_diarization_wsl.py`: WSL2 NeMo script
- `test_nemo_diarization.ipynb`: Interactive test notebook
- `venv_nemo_wsl/`: WSL2 Python virtual environment (pre-configured)

## Example Workflow

1. **Prepare your audio**: `meeting.wav`
2. **Run diarization**:
   ```python
   from nemo_diarization import diarize_and_transcribe
   
   result = diarize_and_transcribe(
       meeting_audio_path="meeting.wav",
       expected_language="en",
       output_transcriptions=True
   )
   ```
3. **View results**: Check `result['speaker_segments']` for timestamped transcription with speaker labels

## License

MIT
