"""
Example: explain an AV2 scene with Alpamayo-R1-10B.

Prerequisites:
    1. vLLM server running:
       python -m alpamayo.server --model-path nvidia/Alpamayo-R1-10B

    2. AV2 dataset available at a local path.

Run:
    python examples/explain_scene.py \
        --data-dir /raid/av2/sensor/val \
        --log-id <scene_id>
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from alpamayo import SceneInference
from alpamayo.client import GenerationConfig
from load_scene import load_scene


def run_all_tasks(data_dir: str, log_id: str | None, server_url: str):
    config = GenerationConfig(temperature=0.6, max_tokens=2048)
    inf = SceneInference(server_url=server_url, config=config)

    if not inf.check_server():
        print(
            f"[error] Cannot reach server at {server_url}\n"
            "        Start it with: python -m alpamayo.server"
        )
        sys.exit(1)

    # Load once, reuse for all tasks
    print("[step 1/4] Loading AV2 scene...")
    scene = load_scene(data_dir, log_id)

    tasks = [
        ("summary",  None,  None),
        ("behavior", None,  "VEHICLE"),
        ("lidar",    None,  None),
        ("custom",   "What makes this scene particularly challenging for autonomous driving?", None),
    ]

    for task, question, category in tasks:
        print(f"\n[step] Running task: {task}")
        result = inf.run_on_scene(
            scene=scene,
            task=task,
            question=question,
            category=category,
        )
        result.display()


def main():
    parser = argparse.ArgumentParser(description="Explain an AV2 scene with Alpamayo")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--log-id", default=None)
    parser.add_argument("--server-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    run_all_tasks(args.data_dir, args.log_id, args.server_url)


if __name__ == "__main__":
    main()
