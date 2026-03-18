#!/usr/bin/env python3
"""
av2_alpamayo_batch.py
Run Alpamayo-R1 inference across all AV2 scenes in one shot.

Loads the model ONCE and loops over every scene + sampled frames.
Results are saved to {data_dir}/{log_id}/inference/{timestamp_ns}.json
so the OmniSight web viewer can read them automatically.

Run:
    cd ~/alpamayo-nvlabs && source ar1_venv/bin/activate
    python ~/OmniSight-av2-alpa/av2_alpamayo_batch.py \\
        --data-dir /raid/av2/sensor/val \\
        --model-path ~/models/Alpamayo-R1-10B \\
        --stride 30 \\
        --describe

Options:
    --stride N      Run inference every N LiDAR frames (default 30 = every 3s at 10Hz)
    --scenes X,Y,Z  Comma-separated scene IDs to process (default: all)
    --describe      Also generate rich scene descriptions via VLM backbone
    --skip-existing Skip frames that already have a saved result (resume support)
    --max-frames N  Max frames per scene (default: unlimited)
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

NVLABS_DIR = Path.home() / "alpamayo-nvlabs"
if (NVLABS_DIR / "src").exists():
    sys.path.insert(0, str(NVLABS_DIR / "src"))

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper

# Re-use helpers from the single-frame script
AV2_CAM_MAP = [
    "ring_front_center",
    "ring_front_center",
    "ring_side_left",
    "ring_side_right",
]

MODEL_H = 320
MODEL_W = 576
N_CAM_FRAMES  = 4
N_EGO_HISTORY = 16

DESCRIBE_PROMPT = (
    "You are an expert autonomous driving analyst. "
    "These images are from a moving vehicle's surround-view cameras. "
    "Describe the driving scene in detail covering:\n"
    "1. Road type and layout (highway, intersection, urban street, etc.)\n"
    "2. Traffic density and nearby vehicles — position, speed, behaviour\n"
    "3. Pedestrians or cyclists visible\n"
    "4. Road markings, signs, traffic lights\n"
    "5. Weather and lighting conditions\n"
    "6. Any hazards, unusual events, or safety-relevant observations\n"
    "Be specific and factual."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),  1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx**2+qy**2)],
    ], dtype=np.float64)


def nearest_image(cam_dir: Path, target_ts: int) -> Path:
    jpgs = sorted(cam_dir.glob("*.jpg"))
    pngs = sorted(cam_dir.glob("*.png")) if not jpgs else []
    files = jpgs or pngs
    if not files:
        raise FileNotFoundError(f"No images in {cam_dir}")
    ts_list = [int(f.stem) for f in files]
    nearest = min(ts_list, key=lambda t: abs(t - target_ts))
    ext = ".jpg" if jpgs else ".png"
    return cam_dir / f"{nearest}{ext}"


# ── Per-frame data loading ────────────────────────────────────────────────────

def load_frame(scene_dir: Path, ego_df: pd.DataFrame,
               all_ts: list[int], frame_idx: int) -> dict | None:
    """Load one frame's images + ego history. Returns None on error."""
    try:
        frame_idx = min(frame_idx, len(all_ts) - 1)
        start = max(0, frame_idx - N_CAM_FRAMES + 1)
        selected_ts = all_ts[start: frame_idx + 1]
        while len(selected_ts) < N_CAM_FRAMES:
            selected_ts = [selected_ts[0]] + selected_ts

        image_frames = []
        for ts in selected_ts:
            for cam in AV2_CAM_MAP:
                cam_dir = scene_dir / "sensors" / "cameras" / cam
                img_path = nearest_image(cam_dir, ts)
                img = Image.open(img_path).convert("RGB").resize(
                    (MODEL_W, MODEL_H), Image.LANCZOS
                )
                image_frames.append(img)

        current_ts = selected_ts[-1]
        has_quat = all(c in ego_df.columns for c in ["qw", "qx", "qy", "qz"])
        pose_idx = int((ego_df["timestamp_ns"] - current_ts).abs().idxmin())
        hist_start = max(0, pose_idx - N_EGO_HISTORY + 1)
        hist_df = ego_df.iloc[hist_start: pose_idx + 1].reset_index(drop=True)
        while len(hist_df) < N_EGO_HISTORY:
            hist_df = pd.concat([hist_df.iloc[[0]], hist_df]).reset_index(drop=True)
        hist_df = hist_df.tail(N_EGO_HISTORY).reset_index(drop=True)

        ref = hist_df.iloc[-1]
        ref_xyz = np.array([ref["tx_m"], ref["ty_m"], ref["tz_m"]])
        ref_R   = quat_to_rot(ref["qw"], ref["qx"], ref["qy"], ref["qz"]) if has_quat else np.eye(3)

        xyz_list, rot_list = [], []
        for _, row in hist_df.iterrows():
            xyz = np.array([row["tx_m"], row["ty_m"], row["tz_m"]])
            R   = quat_to_rot(row["qw"], row["qx"], row["qy"], row["qz"]) if has_quat else np.eye(3)
            xyz_list.append(ref_R.T @ (xyz - ref_xyz))
            rot_list.append(ref_R.T @ R)

        ego_xyz = torch.tensor(np.array(xyz_list, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
        ego_rot = torch.tensor(np.array(rot_list, dtype=np.float32)).unsqueeze(0).unsqueeze(0)

        return {
            "image_frames":    image_frames,
            "ego_history_xyz": ego_xyz,
            "ego_history_rot": ego_rot,
            "current_ts":      current_ts,
        }
    except Exception as e:
        print(f"    [warn] Frame {frame_idx} load error: {e}")
        return None


# ── Per-frame inference ───────────────────────────────────────────────────────

import math
import torchvision.transforms.functional as TF


def infer_frame(model, processor, frame_data: dict,
                with_description: bool) -> dict:
    import torch

    frames_tensor = torch.stack([TF.to_tensor(img) for img in frame_data["image_frames"]])
    messages = helper.create_message(frames_tensor)

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
        "ego_history_xyz": frame_data["ego_history_xyz"],
        "ego_history_rot": frame_data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, "cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )
    torch.cuda.synchronize()
    t_infer = time.perf_counter() - t0

    # Extract CoC
    cot = extra["cot"]
    while not isinstance(cot, str) and hasattr(cot, "__len__"):
        cot = cot[0]
    cot = str(cot)

    # Extract waypoints
    waypoints = pred_xyz.cpu().float().numpy().reshape(-1, 3)

    # Trajectory stats
    seg_dists = [
        math.sqrt(sum((waypoints[i+1][k] - waypoints[i][k])**2 for k in range(3)))
        for i in range(len(waypoints) - 1)
    ]
    total_dist  = sum(seg_dists)
    peak_speed  = max(seg_dists, default=0.0) / 0.1

    # Optional scene description — use the VLM backbone inside AlpamayoR1
    scene_description = None
    if with_description:
        # AlpamayoR1 wraps a VLM; find whichever sub-module has generate()
        gen_model = None
        for attr in ["vlm", "language_model", "model", "transformer", "base_model"]:
            candidate = getattr(model, attr, None)
            if candidate is not None and hasattr(candidate, "generate"):
                gen_model = candidate
                break
        # Also check the model itself as a last resort
        if gen_model is None and hasattr(model, "generate"):
            gen_model = model

        if gen_model is None:
            print("    [warn] Could not find generate() — skipping scene description")
        else:
            try:
                desc_messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": f} for f in frames_tensor]
                        + [{"type": "text", "text": DESCRIBE_PROMPT}],
                    }
                ]
                desc_inputs = processor.apply_chat_template(
                    desc_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                input_len = desc_inputs["input_ids"].shape[1]
                desc_inputs_cuda = {k: v.to("cuda") if hasattr(v, "to") else v
                                    for k, v in desc_inputs.items()}
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    out_ids = gen_model.generate(
                        **desc_inputs_cuda,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                    )
                scene_description = processor.tokenizer.decode(
                    out_ids[0][input_len:], skip_special_tokens=True
                ).strip()
            except Exception as e:
                print(f"    [warn] Scene description failed: {e}")

    return {
        "current_ts":        frame_data["current_ts"],
        "cot":               cot,
        "scene_description": scene_description,
        "waypoints_xyz":     waypoints.tolist(),
        "metrics": {
            "inference_s":       round(t_infer, 3),
            "total_path_m":      round(total_dist, 3),
            "peak_speed_ms":     round(peak_speed, 3),
            "gpu_mem_alloc_gb":  round(torch.cuda.memory_allocated()/1e9, 3),
        },
    }


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(data_dir: str, model_path: str, scene_ids: list[str],
              stride: int, with_description: bool,
              skip_existing: bool, max_frames: int | None):

    root = Path(data_dir)

    # ── Load model once ────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Loading Alpamayo-R1 from {model_path} ...")
    t0 = time.perf_counter()
    model = AlpamayoR1.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    model.eval()
    t_load = time.perf_counter() - t0
    processor = helper.get_processor(model.tokenizer)
    print(f"  Model ready  ({t_load:.1f}s)  |  "
          f"GPU {torch.cuda.memory_allocated()/1e9:.1f} GB allocated")
    print("=" * 60)

    total_scenes = len(scene_ids)
    total_saved  = 0
    total_skipped = 0
    total_errors  = 0
    t_batch_start = time.perf_counter()

    for scene_idx, log_id in enumerate(scene_ids, 1):
        scene_dir = root / log_id
        if not scene_dir.is_dir():
            print(f"[{scene_idx}/{total_scenes}] SKIP {log_id} — not a directory")
            continue

        # Front camera timestamps = master clock
        front_dir = scene_dir / "sensors" / "cameras" / "ring_front_center"
        all_ts = sorted(int(f.stem) for f in front_dir.glob("*.jpg"))
        if not all_ts:
            all_ts = sorted(int(f.stem) for f in front_dir.glob("*.png"))
        if not all_ts:
            print(f"[{scene_idx}/{total_scenes}] SKIP {log_id} — no front camera images")
            continue

        # Ego poses
        pose_path = scene_dir / "city_SE3_egovehicle.feather"
        if not pose_path.exists():
            print(f"[{scene_idx}/{total_scenes}] SKIP {log_id} — no ego pose file")
            continue
        ego_df = pd.read_feather(pose_path).sort_values("timestamp_ns").reset_index(drop=True)

        # Output directory
        infer_dir = scene_dir / "inference"
        infer_dir.mkdir(parents=True, exist_ok=True)

        # Select frame indices
        frame_indices = list(range(N_CAM_FRAMES - 1, len(all_ts), stride))
        if max_frames:
            frame_indices = frame_indices[:max_frames]

        scene_saved = scene_skipped = scene_errors = 0
        print(f"\n[{scene_idx}/{total_scenes}] {log_id}")
        print(f"  {len(all_ts)} frames total → {len(frame_indices)} to process (stride={stride})")

        for fi, frame_idx in enumerate(frame_indices):
            # Compute what timestamp this frame will produce
            ts_for_frame = all_ts[frame_idx]
            out_path = infer_dir / f"{ts_for_frame}.json"

            if skip_existing and out_path.exists():
                scene_skipped += 1
                continue

            frame_data = load_frame(scene_dir, ego_df, all_ts, frame_idx)
            if frame_data is None:
                scene_errors += 1
                continue

            try:
                result = infer_frame(model, processor, frame_data, with_description)
                result["log_id"] = log_id
                out_path.write_text(json.dumps(result, indent=2))
                scene_saved += 1

                # ETA estimate
                elapsed = time.perf_counter() - t_batch_start
                done_total = total_saved + scene_saved
                remaining_this_scene = len(frame_indices) - fi - 1
                remaining_scenes = total_scenes - scene_idx
                # rough frames remaining
                frames_per_scene_avg = len(frame_indices)
                frames_left = remaining_this_scene + remaining_scenes * frames_per_scene_avg
                speed = done_total / elapsed if elapsed > 0 else 0
                eta_s = frames_left / speed if speed > 0 else 0
                eta_str = f"{int(eta_s//3600)}h{int((eta_s%3600)//60)}m" if eta_s > 60 else f"{eta_s:.0f}s"

                print(f"  [{fi+1:3d}/{len(frame_indices)}] ts={ts_for_frame}  "
                      f"cot={result['cot'][:50]!r}  "
                      f"path={result['metrics']['total_path_m']:.1f}m  "
                      f"infer={result['metrics']['inference_s']:.2f}s  "
                      f"ETA={eta_str}")

            except Exception as e:
                print(f"  [{fi+1:3d}/{len(frame_indices)}] ERROR frame {frame_idx}: {e}")
                scene_errors += 1

        total_saved   += scene_saved
        total_skipped += scene_skipped
        total_errors  += scene_errors
        print(f"  → saved={scene_saved}  skipped={scene_skipped}  errors={scene_errors}")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_batch_start
    print("\n" + "=" * 60)
    print("  Batch complete")
    print(f"  Scenes   : {total_scenes}")
    print(f"  Saved    : {total_saved}")
    print(f"  Skipped  : {total_skipped}")
    print(f"  Errors   : {total_errors}")
    print(f"  Total time: {elapsed/3600:.1f}h")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Alpamayo-R1 inference over all AV2 scenes")
    parser.add_argument("--data-dir",   default="/raid/av2/sensor/val")
    parser.add_argument("--model-path", default=str(Path.home() / "models/Alpamayo-R1-10B"))
    parser.add_argument("--scenes",     default=None,
                        help="Comma-separated scene IDs (default: all)")
    parser.add_argument("--stride",     type=int, default=30,
                        help="Run inference every N frames (default: 30 = every 3s at 10Hz)")
    parser.add_argument("--describe",   action="store_true",
                        help="Also generate rich scene description via VLM backbone")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames that already have a saved result (resume)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per scene (default: unlimited)")
    args = parser.parse_args()

    root = Path(args.data_dir)
    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",")]
    else:
        scene_ids = sorted(p.name for p in root.iterdir()
                           if p.is_dir() and not p.name.startswith("."))

    print(f"Scenes to process: {len(scene_ids)}")
    print(f"Stride: every {args.stride} frames (~{args.stride/10:.1f}s)")
    print(f"Describe: {args.describe}")
    print(f"Skip existing: {args.skip_existing}")

    run_batch(
        data_dir=args.data_dir,
        model_path=args.model_path,
        scene_ids=scene_ids,
        stride=args.stride,
        with_description=args.describe,
        skip_existing=args.skip_existing,
        max_frames=args.max_frames,
    )
