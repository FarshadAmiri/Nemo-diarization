"""
Example: Using NeMo Diarization
Simple script showing how to use the diarization function
"""

from nemo_diarization import diarize_and_transcribe

# Example 1: Basic diarization only
print("=" * 70)
print("Example 1: Basic Diarization (No Transcription)")
print("=" * 70)

result = diarize_and_transcribe(
    meeting_audio_path="your_audio.wav",  # Replace with your audio file
    expected_language=None,  # Auto-detect
    output_transcriptions=False  # Just diarization
)

print(f"\nDetected {result['num_speakers']} speakers")
print(f"Found {len(result['segments'])} segments")
print("\nFirst 5 segments:")
for i, seg in enumerate(result['segments'][:5], 1):
    print(f"  {i}. [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}")


# Example 2: Diarization + Transcription with default Whisper model
print("\n" + "=" * 70)
print("Example 2: Diarization + Transcription (English)")
print("=" * 70)

result = diarize_and_transcribe(
    meeting_audio_path="your_audio.wav",
    expected_language="en",
    output_transcriptions=True
)

print(f"\nDetected language: {result.get('detected_language', 'N/A')}")
print("\nFirst 3 transcribed segments with speakers:")
for seg in result['speaker_segments'][:3]:
    print(f"\n[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}:")
    print(f"  \"{seg['text']}\"")


# Example 3: With custom Whisper model
print("\n" + "=" * 70)
print("Example 3: Using Custom Whisper Model (Persian)")
print("=" * 70)

result = diarize_and_transcribe(
    meeting_audio_path="persian_audio.wav",
    expected_language="fa",
    output_transcriptions=True,
    transcriptor_model_path=r"D:\models\whisper_persian_finetuned.pt"  # Your cached model
)

print(f"\nFull transcription:\n{result['transcription']}")


# Example 4: Advanced options
print("\n" + "=" * 70)
print("Example 4: Advanced Configuration")
print("=" * 70)

from nemo_diarization import process_audio_with_nemo

result = process_audio_with_nemo(
    meeting_audio_path="meeting.wav",
    voice_embeddings_database_path="speakers_db.json",
    expected_language="en",
    output_transcriptions=True,
    transcriptor_model_path=r"D:\models\whisper_medium.pt",
    num_speakers=3,  # If you know there are 3 speakers
    use_wsl=False,  # Use Windows pyannote instead of WSL2 NeMo
    output_dir="my_output_folder"
)

print(f"Results saved to: {result['output_files']}")
