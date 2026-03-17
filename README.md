# OmniSight — AV2 Scene Explanation with Alpamayo-R1-10B

OmniSight loads [Argoverse 2](https://argoverse.github.io/user-guide/datasets/sensor.html) sensor scenes (LiDAR + cameras + 3D annotations) and uses [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) to generate natural language explanations of autonomous driving scenarios.

## Features

- Load any AV2 scene (LiDAR sweeps, ring cameras, 3D object annotations)
- Four explanation tasks: **scene summary**, **object behavior**, **lidar analysis**, **custom Q&A**
- Streaming and batch inference modes
- One-command DGX deployment via `deploy_dgx.sh`
- OpenAI-compatible API — drop-in replacement for other LLM clients

## Project Structure

```
OmniSight-av2-alpa/
├── load_scene.py            # AV2 scene loader → returns dict of sweeps/cameras/annotations
├── alpamayo/
│   ├── server.py            # Start vLLM inference server
│   ├── client.py            # AlpamayoClient — wraps vLLM OpenAI API
│   ├── inference.py         # SceneInference — full pipeline + CLI
│   └── prompts.py           # Prompt templates for each task
├── examples/
│   └── explain_scene.py     # Run all 4 tasks on a scene
├── scripts/
│   └── deploy_dgx.sh        # One-command DGX setup
└── requirements.txt
```

## Quick Start

### 1. Clone on DGX

```bash
git clone https://github.com/your-org/OmniSight-av2-alpa.git
cd OmniSight-av2-alpa
```

### 2. One-command deploy

```bash
bash scripts/deploy_dgx.sh
```

This will:
- Create a `omnisight` conda environment
- Install all dependencies
- Download `nvidia/Alpamayo-R1-10B` from HuggingFace (~20 GB)
- Start the vLLM server in a `tmux` session

### 3. Run inference

```bash
# Scene summary
python -m alpamayo.inference \
    --data-dir /raid/av2/sensor/val \
    --task summary

# Object behavior analysis (vehicles only)
python -m alpamayo.inference \
    --data-dir /raid/av2/sensor/val \
    --task behavior \
    --category VEHICLE

# LiDAR point cloud analysis
python -m alpamayo.inference \
    --data-dir /raid/av2/sensor/val \
    --task lidar

# Custom question
python -m alpamayo.inference \
    --data-dir /raid/av2/sensor/val \
    --task custom \
    --question "Why is this scene challenging for autonomous driving?"

# Stream response token by token
python -m alpamayo.inference \
    --data-dir /raid/av2/sensor/val \
    --task summary \
    --stream
```

### 4. Python API

```python
from alpamayo import SceneInference

inf = SceneInference(server_url="http://localhost:8000/v1")

# Load + explain in one call
result = inf.run(
    data_dir="/raid/av2/sensor/val",
    log_id="scene-001",          # omit to use first scene
    task="summary",
)
print(result.explanation)
```

### 5. Run all tasks on one scene

```bash
python examples/explain_scene.py --data-dir /raid/av2/sensor/val
```

## Manual Setup (without deploy script)

```bash
# 1. Create environment
conda create -n omnisight python=3.11 -y
conda activate omnisight
pip install -r requirements.txt

# 2. Download model
huggingface-cli download nvidia/Alpamayo-R1-10B \
    --local-dir ~/models/Alpamayo-R1-10B \
    --local-dir-use-symlinks False

# 3. Start server
python -m alpamayo.server --model-path ~/models/Alpamayo-R1-10B

# 4. Run inference (separate terminal)
python -m alpamayo.inference --data-dir /raid/av2/sensor/val --task summary
```

## Multi-GPU (tensor parallelism)

```bash
# Use 4 GPUs
python -m alpamayo.server --tensor-parallel 4

# Or via deploy script
bash scripts/deploy_dgx.sh --tensor-parallel 4
```

## Access from your laptop (SSH port forward)

```bash
# On your laptop — forward DGX port 8000 to localhost
ssh -L 8000:localhost:8000 user@your-dgx-ip

# Then point the client to localhost
inf = SceneInference(server_url="http://localhost:8000/v1")
```

## Requirements

- NVIDIA DGX (A100 or H100) with CUDA 12+
- ~20 GB disk space for the model
- [Argoverse 2 Sensor Dataset](https://argoverse.github.io/user-guide/datasets/sensor.html)
- `tmux` (for background server in deploy script)

## License

Apache 2.0
