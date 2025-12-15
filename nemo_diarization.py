"""
NVIDIA NeMo Speaker Diarization with Whisper Transcription
Provides a unified function for speaker diarization and optional transcription
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Union
import platform


def process_audio_with_nemo(
    meeting_audio_path: str,
    voice_embeddings_database_path: str,
    expected_language: Optional[str] = None,
    output_transcriptions: bool = False,
    transcriptor_model_path: Optional[str] = None,
    num_speakers: Optional[int] = None,
    output_dir: Optional[str] = None,
    use_wsl: bool = True
) -> Dict:
    """
    Complete pipeline for speaker diarization using NeMo and optional Whisper transcription
    
    Args:
        meeting_audio_path: Path to the meeting/audio file to process
        voice_embeddings_database_path: Path to speaker embeddings database (JSON file)
        expected_language: Language code ('en', 'fa', 'ar', etc.) or None for auto-detect
        output_transcriptions: Whether to generate transcriptions
        transcriptor_model_path: Path to custom Whisper model (optional)
        num_speakers: Expected number of speakers (optional, will auto-detect if None)
        output_dir: Directory to save output files (optional)
        use_wsl: Use WSL2 for NeMo (True) or fallback to pyannote on Windows (False)
    
    Returns:
        Dictionary containing:
            - segments: List of diarization segments with speaker labels
            - transcription: Full transcribed text if output_transcriptions=True
            - num_speakers: Number of detected speakers
            - speaker_segments: List of segments with speaker and text
            - output_files: Paths to generated output files
    
    Example:
        result = process_audio_with_nemo(
            meeting_audio_path="meeting.wav",
            voice_embeddings_database_path="speakers_db.json",
            expected_language="en",
            output_transcriptions=True,
            transcriptor_model_path="whisper_medium.pt"
        )
    """
    
    # Validate inputs
    if not os.path.exists(meeting_audio_path):
        raise FileNotFoundError(f"Audio file not found: {meeting_audio_path}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(meeting_audio_path).parent / "nemo_output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("NVIDIA NeMo SPEAKER DIARIZATION PIPELINE")
    print("="*70)
    print(f"Audio file: {meeting_audio_path}")
    print(f"Voice database: {voice_embeddings_database_path}")
    print(f"Language: {expected_language or 'auto-detect'}")
    print(f"Transcription: {'enabled' if output_transcriptions else 'disabled'}")
    print(f"Mode: {'WSL2 NeMo' if use_wsl else 'Windows pyannote'}")
    print("="*70)
    
    # Choose processing backend
    if use_wsl and platform.system() == "Windows":
        result = _process_with_nemo_wsl(
            meeting_audio_path=meeting_audio_path,
            voice_embeddings_database_path=voice_embeddings_database_path,
            expected_language=expected_language,
            output_transcriptions=output_transcriptions,
            transcriptor_model_path=transcriptor_model_path,
            num_speakers=num_speakers,
            output_dir=output_dir
        )
    else:
        result = _process_with_pyannote(
            meeting_audio_path=meeting_audio_path,
            voice_embeddings_database_path=voice_embeddings_database_path,
            expected_language=expected_language,
            output_transcriptions=output_transcriptions,
            transcriptor_model_path=transcriptor_model_path,
            num_speakers=num_speakers,
            output_dir=output_dir
        )
    
    return result


def _process_with_nemo_wsl(
    meeting_audio_path: str,
    voice_embeddings_database_path: str,
    expected_language: Optional[str],
    output_transcriptions: bool,
    transcriptor_model_path: Optional[str],
    num_speakers: Optional[int],
    output_dir: Path
) -> Dict:
    """Process audio using NeMo in WSL2"""
    
    print("\n[WSL2 Mode] Running NeMo diarization in WSL2 Ubuntu...")
    
    # Convert Windows paths to WSL paths
    audio_path_wsl = _windows_to_wsl_path(meeting_audio_path)
    output_dir_wsl = _windows_to_wsl_path(str(output_dir))
    
    # Create WSL command to run NeMo diarization script
    wsl_script_path = Path(__file__).parent / "nemo_diarization_wsl.py"
    wsl_script_path_wsl = _windows_to_wsl_path(str(wsl_script_path))
    
    venv_python = "/mnt/d/Git_repos/Nemo-diarization/venv_nemo_wsl/bin/python"
    
    cmd_args = [
        "wsl", "-d", "Ubuntu",
        venv_python,
        wsl_script_path_wsl,
        "--audio", audio_path_wsl,
        "--output-dir", output_dir_wsl
    ]
    
    if num_speakers:
        cmd_args.extend(["--num-speakers", str(num_speakers)])
    
    if output_transcriptions:
        cmd_args.append("--transcribe")
        if expected_language:
            cmd_args.extend(["--language", expected_language])
        if transcriptor_model_path:
            model_path_wsl = _windows_to_wsl_path(transcriptor_model_path)
            cmd_args.extend(["--model-path", model_path_wsl])
    
    # Run WSL command
    print(f"Executing: {' '.join(cmd_args)}")
    result = subprocess.run(cmd_args, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError(f"NeMo diarization failed: {result.stderr}")
    
    # Load results from output file
    result_file = output_dir / "diarization_result.json"
    with open(result_file, 'r') as f:
        diarization_result = json.load(f)
    
    return diarization_result


def _process_with_pyannote(
    meeting_audio_path: str,
    voice_embeddings_database_path: str,
    expected_language: Optional[str],
    output_transcriptions: bool,
    transcriptor_model_path: Optional[str],
    num_speakers: Optional[int],
    output_dir: Path
) -> Dict:
    """Fallback processing using pyannote.audio on Windows"""
    
    print("\n[Windows Mode] Running pyannote.audio diarization...")
    
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        raise ImportError(
            "pyannote.audio not installed. Please install with:\n"
            "pip install pyannote.audio torch torchaudio"
        )
    
    # Load HF token
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    # Initialize pipeline
    print("Loading pyannote diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    
    # Run diarization
    print("Running diarization...")
    if num_speakers:
        diarization = pipeline(meeting_audio_path, num_speakers=num_speakers)
    else:
        diarization = pipeline(meeting_audio_path)
    
    # Convert to segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    print(f"✓ Detected {len(set(s['speaker'] for s in segments))} speakers")
    print(f"✓ Total segments: {len(segments)}")
    
    result = {
        'segments': segments,
        'num_speakers': len(set(s['speaker'] for s in segments)),
        'output_files': {}
    }
    
    # Save diarization results
    diarization_file = output_dir / "diarization_result.json"
    with open(diarization_file, 'w') as f:
        json.dump(result, f, indent=2)
    result['output_files']['diarization'] = str(diarization_file)
    
    # Transcription if requested
    if output_transcriptions:
        result = _add_transcription(
            result,
            meeting_audio_path,
            expected_language,
            transcriptor_model_path,
            output_dir
        )
    
    return result


def _add_transcription(
    result: Dict,
    audio_path: str,
    language: Optional[str],
    model_path: Optional[str],
    output_dir: Path
) -> Dict:
    """Add Whisper transcription to diarization results"""
    
    print("\n[Transcription] Running Whisper...")
    
    try:
        import whisper
    except ImportError:
        raise ImportError("openai-whisper not installed. Install with: pip install openai-whisper")
    
    # Load Whisper model
    if model_path and os.path.exists(model_path):
        print(f"Loading custom Whisper model from: {model_path}")
        model = whisper.load_model(model_path)
    else:
        print("Loading Whisper base model...")
        model = whisper.load_model("base")
    
    # Transcribe
    transcription = model.transcribe(
        audio_path,
        language=language,
        task="transcribe",
        word_timestamps=True,
        verbose=False
    )
    
    # Align transcription with diarization
    speaker_segments = _align_transcription_with_diarization(
        result['segments'],
        transcription['segments']
    )
    
    result['transcription'] = transcription['text']
    result['speaker_segments'] = speaker_segments
    result['detected_language'] = transcription.get('language', 'unknown')
    
    # Save transcription results
    transcript_file = output_dir / "transcription_with_speakers.json"
    with open(transcript_file, 'w') as f:
        json.dump(speaker_segments, f, indent=2, ensure_ascii=False)
    result['output_files']['transcription'] = str(transcript_file)
    
    print(f"✓ Transcription complete ({result['detected_language']})")
    
    return result


def _align_transcription_with_diarization(
    diarization_segments: List[Dict],
    transcription_segments: List[Dict]
) -> List[Dict]:
    """Align Whisper transcription segments with speaker diarization"""
    
    aligned = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg['start']
        trans_end = trans_seg['end']
        trans_mid = (trans_start + trans_end) / 2
        
        # Find overlapping speaker segment
        speaker = "UNKNOWN"
        for dia_seg in diarization_segments:
            if dia_seg['start'] <= trans_mid <= dia_seg['end']:
                speaker = dia_seg['speaker']
                break
        
        aligned.append({
            'start': trans_start,
            'end': trans_end,
            'speaker': speaker,
            'text': trans_seg['text'].strip()
        })
    
    return aligned


def _windows_to_wsl_path(windows_path: str) -> str:
    """Convert Windows path to WSL path"""
    path = Path(windows_path).absolute()
    
    # Convert drive letter (e.g., D:\ to /mnt/d/)
    path_str = str(path)
    if len(path_str) > 1 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    
    return path_str.replace('\\', '/')


# Convenience function with simpler name
def diarize_and_transcribe(
    meeting_audio_path: str,
    voice_embeddings_database_path: str = "",
    expected_language: Optional[str] = None,
    output_transcriptions: bool = True,
    transcriptor_model_path: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Simplified wrapper function for diarization with transcription
    
    Args:
        meeting_audio_path: Path to audio file
        voice_embeddings_database_path: Path to speaker database (can be empty)
        expected_language: Language code or None
        output_transcriptions: Whether to transcribe
        transcriptor_model_path: Path to Whisper model
        **kwargs: Additional arguments for process_audio_with_nemo
    
    Returns:
        Diarization and transcription results
    """
    return process_audio_with_nemo(
        meeting_audio_path=meeting_audio_path,
        voice_embeddings_database_path=voice_embeddings_database_path,
        expected_language=expected_language,
        output_transcriptions=output_transcriptions,
        transcriptor_model_path=transcriptor_model_path,
        **kwargs
    )
