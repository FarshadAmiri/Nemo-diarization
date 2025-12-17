#!/bin/bash
cd /mnt/d/Git_repos/Nemo-diarization/venv_nemo_wsl
./bin/pip install --default-timeout=600 torch torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "PyTorch installation complete"
./bin/python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
