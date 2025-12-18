import argparse
import json
from pathlib import Path

from nemo_diarization import add_transcription_to_segments

parser = argparse.ArgumentParser(description='Transcribe audio using Whisper (GPU) via nemo_diarization.add_transcription_to_segments')
parser.add_argument('--audio', required=True, help='Path to audio file')
parser.add_argument('--model', default='medium', help='Whisper model size (tiny, base, small, medium, large)')
parser.add_argument('--language', default=None, help='Expected language code or None for auto')
parser.add_argument('--output', default='transcription_result.json', help='Output JSON file')
args = parser.parse_args()

# Create minimal diarization_result with audio_path only
diarization_result = {'audio_path': args.audio}

print(f"Transcribing {args.audio} with Whisper model {args.model} (GPU if available)...")
result = add_transcription_to_segments(diarization_result, expected_language=args.language, model_name=args.model)

# Save result
out_path = Path(args.output)
with out_path.open('w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"\n✓ Transcription finished. Saved to {out_path.absolute()}")
print(f"Detected language: {result.get('detected_language')}")
print(f"Merged speech blocks: {len(result.get('merged_speeches', []))}")
