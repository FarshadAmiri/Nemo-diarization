#!/usr/bin/env python3
"""
Pyannote Diarization Script for WSL2
This script runs inside WSL2 Ubuntu and performs speaker diarization using pyannote.audio
(NeMo has installation issues on WSL2, so we use pyannote instead)
"""

import argparse
import json
import os
from pathlib import Path


def run_nemo_diarization(
    audio_path: str,
    output_dir: str,
    num_speakers: int = None,
    transcribe: bool = False,
    language: str = None,
    model_path: str = None
):
    """Run pyannote.audio speaker diarization in WSL2"""
    
    print("Initializing pyannote.audio diarization (WSL2 mode)...")
    
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as e:
        print(f"Error importing pyannote.audio: {e}")
        print("Installing pyannote.audio...")
        import subprocess
        subprocess.run([
            "pip", "install",
            "pyannote.audio"
        ], check=True)
        from pyannote.audio import Pipeline
        import torch
    
    # Load pyannote pipeline
    print("Loading pyannote diarization pipeline...")
    
    # Try to get HF token from environment
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Use CPU (WSL2)
    device = torch.device("cpu")
    pipeline.to(device)
    
    # Run diarization
    print(f"Running diarization on: {audio_path}")
    if num_speakers:
        diarization = pipeline(audio_path, num_speakers=num_speakers)
    else:
        diarization = pipeline(audio_path)
    
    # Convert to segments
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    print(f"✓ Diarization complete: {len(segments)} segments")
    print(f"✓ Detected {len(set(s['speaker'] for s in segments))} speakers")
    
    result = {
        'segments': segments,
        'num_speakers': len(set(s['speaker'] for s in segments)),
        'output_files': {}
    }
    
    # Transcription if requested
    if transcribe:
        result = add_whisper_transcription(
            result, audio_path, language, model_path, output_dir
        )
    
    # Save result
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_file = Path(output_dir) / "diarization_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}")
    
    return result


def add_whisper_transcription(result, audio_path, language, model_path, output_dir):
    """Add Whisper transcription"""
    
    print("\nRunning Whisper transcription...")
    
    try:
        import whisper
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "openai-whisper"], check=True)
        import whisper
    
    # Load model
    if model_path and os.path.exists(model_path):
        model = whisper.load_model(model_path)
    else:
        model = whisper.load_model("base")
    
    # Transcribe
    transcription = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )
    
    # Align with diarization
    speaker_segments = []
    for trans_seg in transcription['segments']:
        trans_mid = (trans_seg['start'] + trans_seg['end']) / 2
        
        speaker = "UNKNOWN"
        for dia_seg in result['segments']:
            if dia_seg['start'] <= trans_mid <= dia_seg['end']:
                speaker = dia_seg['speaker']
                break
        
        speaker_segments.append({
            'start': trans_seg['start'],
            'end': trans_seg['end'],
            'speaker': speaker,
            'text': trans_seg['text'].strip()
        })
    
    result['transcription'] = transcription['text']
    result['speaker_segments'] = speaker_segments
    result['detected_language'] = transcription.get('language')
    
    # Save
    transcript_file = Path(output_dir) / "transcription_with_speakers.json"
    with open(transcript_file, 'w') as f:
        json.dump(speaker_segments, f, indent=2, ensure_ascii=False)
    
    result['output_files']['transcription'] = str(transcript_file)
    
    print(f"✓ Transcription complete")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeMo Speaker Diarization")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-speakers", type=int, help="Expected number of speakers")
    parser.add_argument("--transcribe", action="store_true", help="Enable transcription")
    parser.add_argument("--language", help="Language code for transcription")
    parser.add_argument("--model-path", help="Path to custom Whisper model")
    
    args = parser.parse_args()
    
    run_nemo_diarization(
        audio_path=args.audio,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        transcribe=args.transcribe,
        language=args.language,
        model_path=args.model_path
    )
