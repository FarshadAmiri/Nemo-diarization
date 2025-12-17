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


def diarize_with_nemo(
    meeting_audio_path: str,
    voice_embeddings_database_path: str = "",
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
    print("NVIDIA NeMo SPEAKER DIARIZATION")
    print("="*70)
    print(f"Audio file: {meeting_audio_path}")
    if voice_embeddings_database_path:
        print(f"Voice database: {voice_embeddings_database_path}")
    print(f"Mode: {'WSL2 NeMo GPU' if use_wsl else 'Windows pyannote'}")
    print("="*70)
    
    # Choose processing backend
    if use_wsl and platform.system() == "Windows":
        result = _process_with_nemo_wsl(
            meeting_audio_path=meeting_audio_path,
            voice_embeddings_database_path=voice_embeddings_database_path,
            num_speakers=num_speakers,
            output_dir=output_dir
        )
    else:
        result = _process_with_pyannote(
            meeting_audio_path=meeting_audio_path,
            voice_embeddings_database_path=voice_embeddings_database_path,
            num_speakers=num_speakers,
            output_dir=output_dir
        )
    
    # Store audio path for transcription
    result['audio_path'] = str(meeting_audio_path)
    
    return result


def _process_with_nemo_wsl(
    meeting_audio_path: str,
    voice_embeddings_database_path: str,
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
    
    # Use native WSL venv (not on mounted drive) for fast imports
    venv_python = "/home/user_1/nemo_venv/bin/python"
    
    cmd_args = [
        "wsl", "-d", "Ubuntu",
        venv_python,
        wsl_script_path_wsl,
        "--audio", audio_path_wsl,
        "--output-dir", output_dir_wsl
    ]
    
    if num_speakers:
        cmd_args.extend(["--num-speakers", str(num_speakers)])
    
    # Run WSL command - don't capture output, let it print directly
    print(f"Executing NeMo diarization in WSL...")
    print("Note: First run may take longer while models download\n")
    
    # Run without capturing output to avoid buffering/blocking issues
    result = subprocess.run(cmd_args, check=False)
    
    if result.returncode != 0:
        print(f"\nError: WSL process failed with code {result.returncode}")
        raise RuntimeError(f"NeMo diarization failed")
    
    print("\n✓ NeMo diarization completed")
    
    # Load results from output file
    result_file = output_dir / "diarization_result.json"
    with open(result_file, 'r') as f:
        diarization_result = json.load(f)

    # Speaker identification: match anonymous speakers to known voices
    if voice_embeddings_database_path and os.path.exists(voice_embeddings_database_path):
        try:
            print("\n[Speaker Identification] Matching speakers to database...")
            diarization_result = _identify_speakers_windows(
                diarization_result,
                meeting_audio_path,
                voice_embeddings_database_path
            )
        except Exception as e:
            print(f"Warning: speaker identification failed: {e}")
    
    # Merge consecutive segments from the same speaker
    if 'segments' in diarization_result:
        original_count = len(diarization_result['segments'])
        diarization_result['segments'] = _merge_consecutive_segments(diarization_result['segments'])
        merged_count = len(diarization_result['segments'])
        if merged_count < original_count:
            print(f"✓ Merged {original_count} segments → {merged_count} segments")

    return diarization_result


def _process_with_pyannote(
    meeting_audio_path: str,
    voice_embeddings_database_path: str,
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


def _identify_speakers_windows(diarization_result, audio_path, embeddings_db_path):
    """Identify clustered speakers by comparing Resemblyzer embeddings.

    This runs on the Windows side (where the Resemblyzer DB was created).
    It extracts each diarized segment to a temporary WAV (16k mono) using ffmpeg,
    computes Resemblyzer embeddings, and matches against the supplied DB with
    cosine similarity.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except Exception:
        raise ImportError("Resemblyzer is required for speaker identification. Install with: pip install resemblyzer")

    import numpy as np
    from tempfile import NamedTemporaryFile

    # Load known embeddings DB
    with open(embeddings_db_path, 'r', encoding='utf-8') as f:
        known_db = json.load(f)

    # Convert known embeddings to numpy arrays
    known_names = list(known_db.keys())
    known_embeddings = [np.array(known_db[n]) for n in known_names]

    encoder = VoiceEncoder()

    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # Map diarizer speaker labels to known names
    speaker_map = {}

    segments = diarization_result.get('segments', [])
    import tempfile
    import subprocess
    
    for seg in segments:
        start = seg['start']
        end = seg['end']
        speaker_label = seg.get('speaker', 'unknown')

        # extract segment to temp wav using ffmpeg
        # Use manual cleanup instead of with statement for better Windows compatibility
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        try:
            os.close(tmp_fd)  # Close file descriptor immediately
            
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-ss', str(start), '-to', str(end), '-i', str(audio_path),
                '-ar', '16000', '-ac', '1', tmp_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Warning: ffmpeg failed for segment {start}-{end}: {result.stderr}")
                continue

            # compute embedding
            wav = preprocess_wav(tmp_path)
            emb = encoder.embed_utterance(wav)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

        # compare to known embeddings
        best_name = None
        best_score = -1.0
        for name, k_emb in zip(known_names, known_embeddings):
            score = _cosine(emb, k_emb)
            if score > best_score:
                best_score = score
                best_name = name

        # threshold for acceptance
        if best_score >= 0.65:
            identified = best_name
        else:
            identified = speaker_label

        # store mapping for this speaker label (first seen wins if high confidence)
        if speaker_label not in speaker_map:
            speaker_map[speaker_label] = identified

        seg['identified_speaker'] = identified
        seg['match_score'] = round(best_score, 3)

    # Optionally rename speakers in output to identified names when available
    for seg in segments:
        mapped = speaker_map.get(seg.get('speaker'))
        if mapped and mapped != seg.get('speaker'):
            seg['speaker'] = mapped

    diarization_result['segments'] = segments
    diarization_result['speaker_map'] = speaker_map

    return diarization_result


def _merge_consecutive_segments(segments):
    """Merge consecutive segments from the same speaker.
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'speaker' keys
    
    Returns:
        List of merged segments
    """
    if not segments:
        return segments
    
    # Sort by start time
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    
    merged = []
    current = sorted_segments[0].copy()
    
    for seg in sorted_segments[1:]:
        # If same speaker and segments are close enough (within 1 second gap), merge
        if seg['speaker'] == current['speaker'] and (seg['start'] - current['end']) < 1.0:
            # Extend the current segment
            current['end'] = seg['end']
            # Preserve other fields if they exist
            if 'match_score' in seg and 'match_score' in current:
                # Average the match scores
                current['match_score'] = round((current['match_score'] + seg['match_score']) / 2, 3)
        else:
            # Different speaker or too far apart, save current and start new
            merged.append(current)
            current = seg.copy()
    
    # Don't forget the last segment
    merged.append(current)
    
    return merged


def add_transcription_to_segments(
    diarization_result: Dict,
    expected_language: Optional[str] = None,
    model_name: str = "base"
) -> Dict:
    """
    Add Whisper transcription to diarization segments (runs on Windows).
    
    Args:
        diarization_result: Result from diarize_with_nemo()
        expected_language: Language code ('en', 'fa', etc.) or None for auto-detect
        model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    
    Returns:
        Updated diarization result with transcription fields added
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Whisper not installed. Install with: pip install openai-whisper")
    
    audio_path = diarization_result.get('audio_path')
    if not audio_path:
        raise ValueError("audio_path not found in diarization_result")
    
    print(f"\n[Transcription] Loading Whisper '{model_name}' model...")
    model = whisper.load_model(model_name)
    
    print("[Transcription] Transcribing audio...")
    transcription = model.transcribe(
        audio_path,
        language=expected_language,
        word_timestamps=True,
        verbose=False
    )
    
    # Align transcription with speaker segments
    print("[Transcription] Aligning with speaker segments...")
    speaker_segments = []
    segments = diarization_result.get('segments', [])
    
    for trans_seg in transcription['segments']:
        trans_mid = (trans_seg['start'] + trans_seg['end']) / 2
        
        # Find the speaker for this segment
        speaker = "UNKNOWN"
        for dia_seg in segments:
            if dia_seg['start'] <= trans_mid <= dia_seg['end']:
                speaker = dia_seg['speaker']
                break
        
        speaker_segments.append({
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'speaker': speaker,
            'text': trans_seg['text'].strip()
        })
    
    # Update result with transcription
    diarization_result['transcription'] = transcription['text']
    diarization_result['speaker_segments'] = speaker_segments
    diarization_result['detected_language'] = transcription.get('language')
    
    print(f"✓ Transcription complete")
    print(f"✓ Detected language: {diarization_result['detected_language']}")
    print(f"✓ Transcribed {len(speaker_segments)} segments")
    
    return diarization_result


# Convenience function with simpler name
def diarize_and_transcribe(
    meeting_audio_path: str,
    voice_embeddings_database_path: str = "",
    expected_language: Optional[str] = None,
    transcriptor_model_name: str = "base",
    num_speakers: Optional[int] = None,
    use_wsl: bool = True
) -> Dict:
    """
    Complete pipeline: diarization + transcription.
    
    Args:
        meeting_audio_path: Path to audio file
        voice_embeddings_database_path: Path to speaker database (optional)
        expected_language: Language code or None for auto-detect
        transcriptor_model_name: Whisper model ('tiny', 'base', 'small', 'medium', 'large')
        num_speakers: Expected number of speakers (optional)
        use_wsl: Use WSL2 NeMo (True) or Windows pyannote (False)
    
    Returns:
        Complete diarization and transcription results
    """
    # Step 1: Diarization
    result = diarize_with_nemo(
        meeting_audio_path=meeting_audio_path,
        voice_embeddings_database_path=voice_embeddings_database_path,
        num_speakers=num_speakers,
        use_wsl=use_wsl
    )
    
    # Step 2: Add transcription
    result = add_transcription_to_segments(
        diarization_result=result,
        expected_language=expected_language,
        model_name=transcriptor_model_name
    )
    
    return result
