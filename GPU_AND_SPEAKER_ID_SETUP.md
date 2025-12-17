# GPU Acceleration & Speaker Identification Setup

## Status: In Progress

### ✅ Completed
1. **Code Modifications**
   - [nemo_diarization_wsl.py](nemo_diarization_wsl.py): Auto-detects CUDA and uses GPU when available
   - [nemo_diarization.py](nemo_diarization.py): Added `_identify_speakers_windows()` for post-processing speaker matching
   - Increased batch_size to 128 and num_workers to 4 for better GPU utilization

2. **WSL2 Setup**
   - RTX 4090 is accessible from WSL2 (verified with `nvidia-smi`)
   - CUDA 12.6 driver available

### 🔄 In Progress
1. **Installing PyTorch with CUDA** in WSL2 venv
   - Command: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`
   - Status: Downloading ~664MB (currently at ~60MB)
   - This will enable GPU acceleration in NeMo

### 📋 Remaining Tasks
1. **Verify CUDA Works in WSL Python**
   ```bash
   wsl -d Ubuntu bash -c "cd /mnt/d/Git_repos/Nemo-diarization/venv_nemo_wsl && ./bin/python3 -c 'import torch; print(\"CUDA:\", torch.cuda.is_available())'"
   ```

2. **Install Windows Dependencies** (for speaker identification)
   - ffmpeg: Required for extracting audio segments
   - resemblyzer: Already installed (used in your notebook)
   
   Check ffmpeg:
   ```powershell
   ffmpeg -version
   ```
   
   If missing, install via:
   ```powershell
   winget install ffmpeg
   # or download from https://ffmpeg.org/download.html
   ```

3. **Test Full Pipeline**
   - Run your notebook cell 7 again
   - Expected improvements:
     * **Speed**: ~10-20x faster with GPU (should process faster than real-time)
     * **Speaker Names**: Should show sp3000, sp777, sp422, sp1993 instead of speaker_0, speaker_1, etc.

## How Speaker Identification Works

After NeMo clusters speakers, the code:
1. Extracts each segment to a temporary WAV file (using ffmpeg)
2. Computes Resemblyzer embeddings for each segment
3. Compares against your database (`speakers_db.json`) using cosine similarity
4. Maps cluster labels (speaker_0) → known names (sp3000) when confidence >= 0.65
5. Adds `identified_speaker` and `match_score` to each segment

## Expected Results

### Before (CPU, no identification):
```
1. [0.54s - 9.71s] speaker_1
2. [10.22s - 15.47s] speaker_1
3. [21.18s - 33.31s] speaker_2
```
Processing time: ~2x audio length

### After (GPU + identification):
```
1. [0.54s - 9.71s] sp3000 (match: 0.82)
2. [10.22s - 15.47s] sp3000 (match: 0.81)
3. [21.18s - 33.31s] sp777 (match: 0.74)
```
Processing time: <0.5x audio length (faster than real-time)

## Configuration Details

### GPU Settings (nemo_diarization_wsl.py)
- `device`: Auto-detects 'cuda' when available
- `batch_size`: 128 (increased for GPU)
- `num_workers`: 4 (parallel data loading)

### Speaker Matching Threshold
- Minimum cosine similarity: 0.65
- Adjust in `nemo_diarization.py` line ~180 if needed:
  ```python
  if best_score >= 0.65:  # Change this threshold
      identified = best_name
  ```

## Troubleshooting

### If GPU not used:
```bash
# Check CUDA in WSL
wsl -d Ubuntu bash -c "cd /mnt/d/Git_repos/Nemo-diarization/venv_nemo_wsl && ./bin/python3 -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'"
```

### If speaker names not showing:
- Verify ffmpeg is installed: `ffmpeg -version`
- Check database exists: `outputs\db\speakers_db.json`
- Lower threshold in code (line ~180) if match scores are low

### If extraction fails:
- Install ffmpeg from https://ffmpeg.org/download.html
- Add to PATH
- Restart VS Code

## Next Steps
1. Wait for PyTorch install to complete (~2-5 minutes)
2. Verify CUDA works
3. Install ffmpeg if missing
4. Re-run notebook cell 7
5. Check results show speaker names and faster processing
