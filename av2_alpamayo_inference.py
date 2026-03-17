#!/usr/bin/env python3
"""
av2_alpamayo_inference.py
Load an AV2 scene and run Alpamayo-R1 trajectory inference on it.

Run from ~/alpamayo-nvlabs with the ar1_venv activated:

    cd ~/alpamayo-nvlabs
    source ar1_venv/bin/activate
    python ~/OmniSight-av2-alpa/av2_alpamayo_inference.py \
        --data-dir /raid/av2/sensor/val \
        --model-path ~/models/Alpamayo-R1-10B \
        --frame-idx 20 \
        --output ~/inference_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── Import alpamayo_r1 from the NVlabs repo ───────────────────────────────────
# Assumes this script is run from ~/alpamayo-nvlabs or src/ is on the path.
NVLABS_DIR = Path.home() / "alpamayo-nvlabs"
if (NVLABS_DIR / "src").exists():
    sys.path.insert(0, str(NVLABS_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# ── Camera mapping: AV2 → Alpamayo 4-camera format ───────────────────────────
# Model was trained on: front-wide, front-tele, cross-left, cross-right
# We use the closest AV2 equivalents:
AV2_CAM_MAP = [
    "ring_front_center",   # → front-wide
    "ring_front_center",   # → front-tele (AV2 has no telephoto, reuse front)
    "ring_side_left",      # → cross-left
    "ring_side_right",     # → cross-right
]

MODEL_H = 320    # model expected image height
MODEL_W = 576    # model expected image width
N_CAM_FRAMES  = 4   # 4 consecutive camera frames (0.4 s history @ 10 Hz)
N_EGO_HISTORY = 16  # 16 ego pose history steps (1.6 s @ 10 Hz)


# ── Helpers ───────────────────────────────────────────────────────────────────

def quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def nearest_image(cam_dir: Path, target_ts: int) -> Path:
    """Return the image file in cam_dir whose timestamp is nearest to target_ts."""
    jpgs = sorted(cam_dir.glob("*.jpg"))
    pngs = sorted(cam_dir.glob("*.png")) if not jpgs else []
    files = jpgs or pngs
    if not files:
        raise FileNotFoundError(f"No images in {cam_dir}")
    ts_list = [int(f.stem) for f in files]
    nearest = min(ts_list, key=lambda t: abs(t - target_ts))
    ext = ".jpg" if jpgs else ".png"
    return cam_dir / f"{nearest}{ext}"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_av2_for_alpamayo(
    data_dir: str,
    log_id: str | None = None,
    frame_idx: int = 20,
) -> dict:
    """
    Load one AV2 scene and convert to Alpamayo model input format.

    Returns dict with:
        image_frames      list[PIL.Image]  — 16 images (4 frames × 4 cams)
        ego_history_xyz   Tensor [1, 16, 3]
        ego_history_rot   Tensor [1, 16, 3, 3]
        log_id            str
        current_ts        int  nanoseconds
    """
    root = Path(data_dir)

    # Pick scene
    if log_id is None:
        scenes = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        if not scenes:
            raise ValueError(f"No scenes in {data_dir}")
        log_id = scenes[0]
        print(f"[av2] No log_id given — using: {log_id}")

    scene_dir = root / log_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene not found: {scene_dir}")
    print(f"[av2] Scene: {scene_dir}")

    # ── 1. Pick N_CAM_FRAMES consecutive timestamps ───────────────────────────
    front_dir = scene_dir / "sensors" / "cameras" / "ring_front_center"
    all_ts = sorted(int(f.stem) for f in front_dir.glob("*.jpg"))
    if not all_ts:
        all_ts = sorted(int(f.stem) for f in front_dir.glob("*.png"))
    print(f"[av2] {len(all_ts)} front-center frames available")

    frame_idx = min(frame_idx, len(all_ts) - 1)
    start = max(0, frame_idx - N_CAM_FRAMES + 1)
    selected_ts = all_ts[start : frame_idx + 1]
    while len(selected_ts) < N_CAM_FRAMES:          # pad if at start of scene
        selected_ts = [selected_ts[0]] + selected_ts

    print(f"[av2] Using frames {start}–{frame_idx} (ts {selected_ts[0]} → {selected_ts[-1]})")

    # ── 2. Load camera images ─────────────────────────────────────────────────
    image_frames: list = []
    for ts in selected_ts:
        for cam in AV2_CAM_MAP:
            cam_dir = scene_dir / "sensors" / "cameras" / cam
            img_path = nearest_image(cam_dir, ts)
            img = Image.open(img_path).convert("RGB").resize(
                (MODEL_W, MODEL_H), Image.LANCZOS
            )
            image_frames.append(img)

    print(f"[av2] Loaded {len(image_frames)} images ({N_CAM_FRAMES} frames × {len(AV2_CAM_MAP)} cams, {MODEL_W}×{MODEL_H})")

    # ── 3. Load ego-vehicle poses ─────────────────────────────────────────────
    pose_path = scene_dir / "city_SE3_egovehicle.feather"
    ego_df = pd.read_feather(pose_path).sort_values("timestamp_ns").reset_index(drop=True)
    print(f"[av2] Ego pose columns: {list(ego_df.columns)}")
    has_quat = all(c in ego_df.columns for c in ["qw", "qx", "qy", "qz"])
    if not has_quat:
        print("[av2] WARNING: no quaternion columns found — using identity rotation")

    # Find ego pose nearest to current camera timestamp
    current_ts = selected_ts[-1]
    pose_idx = int((ego_df["timestamp_ns"] - current_ts).abs().idxmin())
    hist_start = max(0, pose_idx - N_EGO_HISTORY + 1)
    hist_df = ego_df.iloc[hist_start : pose_idx + 1].reset_index(drop=True)
    while len(hist_df) < N_EGO_HISTORY:             # pad if at start of scene
        hist_df = pd.concat([hist_df.iloc[[0]], hist_df]).reset_index(drop=True)
    hist_df = hist_df.tail(N_EGO_HISTORY).reset_index(drop=True)

    # Reference = last (current) pose
    ref = hist_df.iloc[-1]
    ref_xyz = np.array([ref["tx_m"], ref["ty_m"], ref["tz_m"]])
    ref_R   = quat_to_rot(ref["qw"], ref["qx"], ref["qy"], ref["qz"]) if has_quat else np.eye(3)

    # Convert each pose to ego-relative frame
    ego_history_xyz_list: list[np.ndarray] = []
    ego_history_rot_list: list[np.ndarray] = []
    for _, row in hist_df.iterrows():
        xyz = np.array([row["tx_m"], row["ty_m"], row["tz_m"]])
        R   = quat_to_rot(row["qw"], row["qx"], row["qy"], row["qz"]) if has_quat else np.eye(3)
        ego_history_xyz_list.append(ref_R.T @ (xyz - ref_xyz))
        ego_history_rot_list.append(ref_R.T @ R)

    # Model expects [B, n_traj_group, N, 3] and [B, n_traj_group, N, 3, 3]
    ego_history_xyz = torch.tensor(
        np.array(ego_history_xyz_list, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)   # [1, 1, 16, 3]
    ego_history_rot = torch.tensor(
        np.array(ego_history_rot_list, dtype=np.float32)
    ).unsqueeze(0).unsqueeze(0)   # [1, 1, 16, 3, 3]

    print(f"[av2] ego_history_xyz: {ego_history_xyz.shape}  ego_history_rot: {ego_history_rot.shape}")

    return {
        "image_frames":    image_frames,
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
        "log_id":          log_id,
        "current_ts":      current_ts,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(data: dict, model_path: str, output_path: str | None = None) -> dict:
    t_start = time.perf_counter()

    print(f"\n[model] Loading Alpamayo-R1 from {model_path} ...")
    t0 = time.perf_counter()
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"[model] Model loaded on GPU  ({t_load:.1f}s)")

    # ── GPU memory after load ──────────────────────────────────────────────────
    mem_alloc = torch.cuda.memory_allocated() / 1e9
    mem_res   = torch.cuda.memory_reserved()  / 1e9
    print(f"[gpu]   allocated={mem_alloc:.2f} GB  reserved={mem_res:.2f} GB")

    processor = helper.get_processor(model.tokenizer)

    # create_message expects Tensor [N, C, H, W], not a list of PIL images
    import torchvision.transforms.functional as TF
    frames_tensor = torch.stack([TF.to_tensor(img) for img in data["image_frames"]])
    messages  = helper.create_message(frames_tensor)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = {
        "tokenized_data":  inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    # ── Warm-up sync so timing is accurate ────────────────────────────────────
    torch.cuda.synchronize()

    print("[model] Running trajectory inference ...")
    t0 = time.perf_counter()
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    # cot may be a numpy array of strings — extract the first element as str
    cot_raw = extra["cot"]
    cot = str(cot_raw[0]) if hasattr(cot_raw, "__len__") else str(cot_raw)

    # pred_xyz shape is [B, n_traj_group, num_traj, T, 3] — take first sample
    wp_np = pred_xyz.cpu().float().numpy()
    print(f"[model] pred_xyz shape: {wp_np.shape}")
    waypoints = wp_np.reshape(-1, 3)  # flatten all leading dims → [T, 3]

    # ── Trajectory stats ──────────────────────────────────────────────────────
    import math
    steps = len(waypoints) - 1
    seg_dists = [
        math.sqrt(sum((waypoints[i+1][k] - waypoints[i][k])**2 for k in range(3)))
        for i in range(steps)
    ]
    total_dist = sum(seg_dists)
    final_wp   = waypoints[-1]
    peak_speed = max(seg_dists, default=0.0) / 0.1

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Scene  : {data['log_id']}")
    print(f"  Frame  : ts={data['current_ts']}")
    print()
    print("  Chain-of-Causation reasoning:")
    print("  " + cot.replace("\n", "\n  "))
    print()
    print(f"  Predicted trajectory — {len(waypoints)} waypoints over {len(waypoints)*0.1:.1f}s:")
    for i in range(0, min(10, len(waypoints))):
        wp = waypoints[i]
        print(f"    t={i*0.1:.1f}s   x={wp[0]:+.3f}  y={wp[1]:+.3f}  z={wp[2]:+.3f}")
    if len(waypoints) > 10:
        print(f"    ... ({len(waypoints)-10} more waypoints)")
    print()
    print("  Trajectory summary:")
    print(f"    Total path length : {total_dist:.2f} m")
    print(f"    Final position    : x={final_wp[0]:+.2f}  y={final_wp[1]:+.2f}  z={final_wp[2]:+.2f} m")
    print(f"    Peak speed        : {peak_speed:.2f} m/s  ({peak_speed*3.6:.1f} km/h)")
    print()
    print("  Performance:")
    print(f"    Model load time   : {t_load:.2f} s")
    print(f"    Inference time    : {t_infer:.2f} s")
    print(f"    Total time        : {t_total:.2f} s")
    print(f"    GPU mem allocated : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"    GPU mem reserved  : {torch.cuda.memory_reserved()/1e9:.2f} GB")
    print("="*60)

    result = {
        "log_id":        data["log_id"],
        "current_ts":    data["current_ts"],
        "cot":           cot,
        "waypoints_xyz": waypoints.tolist(),   # 64 × 3
        "metrics": {
            "model_load_s":      round(t_load, 3),
            "inference_s":       round(t_infer, 3),
            "total_s":           round(t_total, 3),
            "total_path_m":      round(total_dist, 3),
            "peak_speed_ms":     round(peak_speed, 3),
            "gpu_mem_alloc_gb":  round(torch.cuda.memory_allocated()/1e9, 3),
            "gpu_mem_reserved_gb": round(torch.cuda.memory_reserved()/1e9, 3),
        },
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"\n[saved] {output_path}")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Alpamayo-R1 inference on AV2 data")
    parser.add_argument("--data-dir",   default="/raid/av2/sensor/val",
                        help="AV2 val directory")
    parser.add_argument("--model-path", default=str(Path.home() / "models/Alpamayo-R1-10B"),
                        help="Local Alpamayo model path")
    parser.add_argument("--log-id",     default=None,
                        help="Scene log ID (default: first scene found)")
    parser.add_argument("--frame-idx",  type=int, default=20,
                        help="Which camera frame to use as 'now' (default: 20)")
    parser.add_argument("--output",     default=None,
                        help="Save JSON result to this path (default: {data_dir}/{log_id}/inference/{ts}.json)")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = load_av2_for_alpamayo(args.data_dir, args.log_id, args.frame_idx)
    print(f"[av2]  Data loading: {time.perf_counter()-t0:.2f}s")

    # Default output: save into the scene directory so the API can serve it
    output = args.output
    if output is None:
        infer_dir = Path(args.data_dir) / data["log_id"] / "inference"
        infer_dir.mkdir(parents=True, exist_ok=True)
        output = str(infer_dir / f"{data['current_ts']}.json")
        print(f"[av2]  Auto output path: {output}")

    run_inference(data, args.model_path, output)
