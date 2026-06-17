
# Seam Tracking Code

This repository provides the implementation for the following paper:

**A Robotic Laser Welding Seam Tracking Method for Complex 3-D Weld Trajectories Based on Geometric Structure Analysis**  
Zhaoqi Chu, Xiangrong Liu, Xuhui Que, Bo Yu, Yawei Hu and Juan Liu
*IEEE Transactions on Industrial Informatics*, 2026.  
DOI: [10.1109/TII.2026.3661009](https://doi.org/10.1109/TII.2026.3661009)

# Installation

```
# Create and activate conda environment
conda create -n st_dl python=3.11
conda activate st_dl

# Set project root. <ROOT> refers to this repository root directory.
export SEAMTRACKING_ROOT=<ROOT>

# Build DCNv2 module
cd <ROOT>/models/backbones
git clone git@github.com:lucasjinreal/DCNv2_latest.git
mv DCNv2_latest DCNv2
cd <ROOT>/models/backbones/DCNv2
./make.sh

# Install dependencies
cd <ROOT>
pip install -r requirements.txt

```

# Dataset and Pretrained Models

The dataset and pretrained weights are hosted on Hugging Face:

- **Dataset**: [xmuczq/SeamTracking](https://huggingface.co/datasets/xmuczq/SeamTracking)
- **Pretrained weights**: [xmuczq/SeamTracking](https://huggingface.co/xmuczq/SeamTracking)

**Option 1 — Hugging Face CLI** (recommended)

```bash
pip install huggingface_hub

# Download dataset → data/
hf download xmuczq/SeamTracking \
    --repo-type dataset --local-dir data

# Download pretrained weights → log/
hf download xmuczq/SeamTracking \
    --repo-type model --local-dir log
```

**Option 2 — Python**

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="xmuczq/SeamTracking", repo_type="dataset", local_dir="data")
snapshot_download(repo_id="xmuczq/SeamTracking", repo_type="model",   local_dir="log")
```

After downloading, the directory structure should be:

```
SeamTracking/
├── data/
│   ├── <dataset_name_1>/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── <dataset_name_2>/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── <dataset_name_3>/
│       ├── train/
│       ├── val/
│       └── test/
└── log/
    └── <timestamp> centernet_<task>/
        ├── best.pth
        ├── checkpoint.pth
        └── runs/
```

# Training


```
cd <ROOT>
python train.py
```

# Test
```
cd <ROOT>
python test.py
```
Results will be saved to:
```
"<ROOT>/results"
```

