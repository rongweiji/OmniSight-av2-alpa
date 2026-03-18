# OmniSight — AV2 Scene Viewer with Alpamayo-R1-10B Trajectory Inference

OmniSight loads [Argoverse 2](https://argoverse.github.io/user-guide/datasets/sensor.html) sensor scenes (LiDAR + 7 ring cameras + 3D annotations) and runs [nvidia/Alpamayo-R1-10B](https://huggingface.co/nvidia/Alpamayo-R1-10B) on an **NVIDIA DGX Spark (GB10 Grace Blackwell)** to predict 6.4-second future trajectories with Chain-of-Causation driving reasoning — all visualised in a real-time 3D web viewer.

## Demo

[![OmniSight Demo](https://vumbnail.com/1174572929.jpg)](https://vimeo.com/1174572929)

## Performance on NVIDIA DGX Spark (GB10)

Measured on Dell Pro Max with GB10 Grace Blackwell (CUDA 13.0, 128 GB unified memory):

| Metric | Value |
|---|---|
| Model size | 10B parameters (22 GB bfloat16) |
| Model load time | ~2 min (first run, PTX JIT for sm_121) |
| Model load time (cached) | ~2 s |
| Trajectory inference | **3.8 s / frame** |
| Scene description (VLM) | **+4 s / frame** |
| GPU memory allocated | 22.2 GB |
| Trajectory horizon | 64 waypoints × 0.1s = **6.4 seconds** |
| Input | 4 cameras × 4 frames + 16-step ego history |
| Output | 64 (x,y,z) waypoints + CoC reasoning text |

## Features

- **Real-time 3D viewer** — interactive LiDAR point cloud with orbit/zoom/pan
- **7 ring cameras** — full surround view, preloaded for smooth playback
- **3D annotation boxes** — per-category color-coded bounding boxes
- **Trajectory overlay** — predicted 6.4s path rendered as green→red gradient in 3D
- **AI inference panel** — Chain-of-Causation decision + rich scene description per frame
- **Batch inference** — process all scenes once, results served instantly by the API
- **Auto-play + loop** — demo-ready, starts playing automatically

## Project Structure

```
OmniSight-av2-alpa/
├── load_scene.py            # AV2 scene loader → returns dict of sweeps/cameras/annotations
├── viewer.py                # FastAPI web viewer — camera gallery + BEV LiDAR (port 7860)
├── alpamayo/
│   ├── server.py            # Start vLLM inference server (port 8000)
│   ├── client.py            # AlpamayoClient — wraps vLLM OpenAI API
│   ├── inference.py         # SceneInference — full pipeline + CLI
│   └── prompts.py           # Prompt templates for each task
├── examples/
│   └── explain_scene.py     # Run all 4 tasks on a scene
├── scripts/
│   └── deploy_dgx.sh        # One-command DGX setup (env + model + data + both services)
└── requirements.txt
```

## Quick Start

### 1. Clone on DGX

```bash
git clone https://github.com/rongweiji/OmniSight-av2-alpa.git
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

## Web Data Viewer

Preview loaded scenes in your browser — camera images, bird's-eye-view LiDAR, annotation stats.

```bash
# Start the viewer (runs on port 7860 by default)
python viewer.py --data-dir /raid/av2/sensor/val --port 7860

# On your laptop, SSH tunnel and open browser
ssh -L 7860:localhost:7860 user@<dgx-ip>
# → http://localhost:7860
```

Viewer endpoints:
- `/` — list all scenes
- `/scene/<log-id>` — camera gallery, BEV LiDAR, annotation table
- `/scene/<log-id>/info` — JSON metadata
- `/scene/<log-id>/lidar` — per-sweep JSON stats

## Access from your laptop (SSH port forward)

```bash
# Forward both ports in one command
ssh -L 7860:localhost:7860 -L 8000:localhost:8000 user@<dgx-ip>

# Web viewer:   http://localhost:7860
# Inference API: http://localhost:8000/v1/models
```

## Requirements

- NVIDIA DGX (A100 or H100) with CUDA 12+
- ~20 GB disk space for the model
- [Argoverse 2 Sensor Dataset](https://argoverse.github.io/user-guide/datasets/sensor.html)
- `tmux` (for background server in deploy script)

## License

Apache 2.0
