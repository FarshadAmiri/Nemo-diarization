#!/usr/bin/env python3
"""
NeMo Diarization Script for WSL2
This script runs inside WSL2 Ubuntu and performs speaker diarization using NeMo toolkit.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def run_nemo_diarization(
    audio_path: str,
    output_dir: str,
    num_speakers: int = None,
    transcribe: bool = False,
    language: str = None,
    model_path: str = None
):
    """Run NeMo speaker diarization in WSL2"""
    
    print("Initializing NeMo diarization (WSL2 mode)...")
    
    try:
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import OmegaConf
    except ImportError as e:
        print(f"Error importing NeMo: {e}")
        print("NeMo toolkit should already be installed in the WSL2 environment")
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create manifest file for NeMo
    manifest_path = Path(output_dir) / "input_manifest.json"
    audio_absolute_path = str(Path(audio_path).absolute())
    
    manifest_entry = {
        "audio_filepath": audio_absolute_path,
        "offset": 0,
        "duration": None,  # Will be auto-detected
        "label": "infer",
        "uniq_id": Path(audio_path).stem
    }
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest_entry, f)
        f.write('\n')
    
    print(f"Created manifest: {manifest_path}")
    
    # Configure NeMo diarization
    config = {
        'device': 'cpu',  # Use CPU for WSL2
        'num_workers': 1,  # Number of workers for data loading
        'sample_rate': 16000,  # Audio sample rate
        'batch_size': 64,  # Batch size for processing
        'verbose': True,  # Show progress
        'diarizer': {
            'manifest_filepath': str(manifest_path),
            'out_dir': str(Path(output_dir)),
            'oracle_vad': False,  # Use Voice Activity Detection
            'collar': 0.25,  # Collar for scoring (in seconds)
            'ignore_overlap': True,  # Ignore overlapping speech
            'clustering': {
                'parameters': {
                    'oracle_num_speakers': num_speakers if num_speakers else False,
                    'max_num_speakers': num_speakers if num_speakers else 8,
                    'enhanced_count_thres': 80,
                    'max_rp_threshold': 0.25,
                    'sparse_search_volume': 30
                }
            },
            'vad': {
                'model_path': 'vad_multilingual_marblenet',  # Pretrained VAD model
                'parameters': {
                    'window_length_in_sec': 0.15,
                    'shift_length_in_sec': 0.01,
                    'smoothing': 'median',
                    'overlap': 0.5,
                    'onset': 0.5,
                    'offset': 0.3,
                    'pad_onset': 0.1,
                    'pad_offset': 0.1,
                    'min_duration_on': 0.2,
                    'min_duration_off': 0.2
                }
            },
            'speaker_embeddings': {
                'model_path': 'titanet_large',  # Pretrained speaker embedding model
                'parameters': {
                    'window_length_in_sec': [1.5, 1.25, 1.0, 0.75, 0.5],
                    'shift_length_in_sec': [0.75, 0.625, 0.5, 0.375, 0.25],
                    'multiscale_weights': [1, 1, 1, 1, 1],
                    'save_embeddings': False  # Don't save intermediate embeddings
                }
            }
        }
    }
    
    cfg = OmegaConf.create(config)
    
    # Load and run diarization
    print("Loading NeMo clustering diarizer...")
    from nemo.collections.asr.models import ClusteringDiarizer
    
    diarizer = ClusteringDiarizer(cfg=cfg)
    
    print(f"Running diarization on: {audio_path}")
    diarizer.diarize()
    
    # Parse RTTM output
    rttm_path = Path(output_dir) / "pred_rttms" / f"{Path(audio_path).stem}.rttm"
    
    segments = []
    speakers = set()
    
    if rttm_path.exists():
        print(f"Reading RTTM file: {rttm_path}")
        with open(rttm_path, 'r') as f:
            for line in f:
                if line.startswith('SPEAKER'):
                    parts = line.strip().split()
                    speaker = parts[7]
                    start = float(parts[3])
                    duration = float(parts[4])
                    end = start + duration
                    
                    segments.append({
                        'start': start,
                        'end': end,
                        'speaker': speaker
                    })
                    speakers.add(speaker)
    else:
        print(f"Warning: RTTM file not found: {rttm_path}")
    
    print(f"✓ Diarization complete: {len(segments)} segments")
    print(f"✓ Detected {len(speakers)} speakers: {sorted(speakers)}")
    
    result = {
        'segments': segments,
        'num_speakers': len(speakers),
        'speakers': sorted(list(speakers)),
        'output_files': {
            'rttm': str(rttm_path) if rttm_path.exists() else None
        }
    }
    
    # Transcription if requested
    if transcribe:
        result = add_whisper_transcription(
            result, audio_path, language, model_path, output_dir
        )
    
    # Save result
    result_file = Path(output_dir) / "diarization_result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Results saved to: {result_file}")
    
    return result


def add_whisper_transcription(result, audio_path, language, model_path, output_dir):
    """Add Whisper transcription aligned with speaker segments"""
    
    print("\nRunning Whisper transcription...")
    
    try:
        import whisper
    except ImportError:
        import subprocess
        print("Installing openai-whisper...")
        subprocess.run(["pip", "install", "openai-whisper"], check=True)
        import whisper
    
    # Load model
    if model_path and os.path.exists(model_path):
        print(f"Loading custom Whisper model: {model_path}")
        model = whisper.load_model(model_path)
    else:
        print("Loading Whisper base model...")
        model = whisper.load_model("base")
    
    # Transcribe
    print("Transcribing audio...")
    transcription = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )
    
    # Align transcription with speaker segments
    speaker_segments = []
    for trans_seg in transcription['segments']:
        trans_mid = (trans_seg['start'] + trans_seg['end']) / 2
        
        # Find the speaker for this segment
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
    
    # Save transcription with speakers
    transcript_file = Path(output_dir) / "transcription_with_speakers.json"
    with open(transcript_file, 'w') as f:
        json.dump(speaker_segments, f, indent=2, ensure_ascii=False)
    
    result['output_files']['transcription'] = str(transcript_file)
    
    print(f"✓ Transcription complete")
    print(f"✓ Detected language: {result['detected_language']}")
    
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
    
    result = run_nemo_diarization(
        audio_path=args.audio,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
        transcribe=args.transcribe,
        language=args.language,
        model_path=args.model_path
    )
    
    # Print summary
    print("\n" + "="*50)
    print("DIARIZATION SUMMARY")
    print("="*50)
    print(f"Audio: {args.audio}")
    print(f"Speakers detected: {result['num_speakers']}")
    print(f"Total segments: {len(result['segments'])}")
    if 'transcription' in result:
        print(f"Transcription: Yes ({result['detected_language']})")
    print("="*50)
