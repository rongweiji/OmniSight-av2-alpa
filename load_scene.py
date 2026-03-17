"""
AV2 Sensor Dataset scene loader.

Loads one scene (log) from the Argoverse 2 sensor dataset into a plain dict
that the rest of the pipeline (prompts, viewer, inference) can consume.

AV2 on-disk layout expected:
    <data_dir>/
        <log_id>/
            sensors/
                cameras/
                    ring_front_center/  *.jpg
                    ring_front_left/    *.jpg
                    ring_front_right/   *.jpg
                    ring_rear_left/     *.jpg
                    ring_rear_right/    *.jpg
                    ring_side_left/     *.jpg
                    ring_side_right/    *.jpg
                lidar/                  *.feather
            annotations.feather
            city_SE3_egovehicle.feather
            calibration/

Usage:
    from load_scene import load_scene
    scene = load_scene("/raid/av2/sensor/val", log_id=None)  # picks first scene
    scene = load_scene("/raid/av2/sensor/val", log_id="abc123...")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# AV2 LiDAR column names (feather schema)
LIDAR_COLS = ["x", "y", "z"]
LIDAR_INTENSITY_COL = "intensity"  # optional


def _list_scenes(data_dir: Path) -> list[str]:
    """Return sorted list of log IDs (subdirectory names) in data_dir."""
    return sorted(
        p.name
        for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def _load_annotations(log_path: Path) -> list[dict]:
    """
    Parse annotations.feather → list of annotation dicts.

    Each dict has:
        category    (str)
        track_uuid  (str)
        xyz_m       (np.ndarray shape [3])
        timestamp_ns (int)
        length_m, width_m, height_m  (float, box dimensions)
        yaw_rad     (float, heading)
    """
    ann_path = log_path / "annotations.feather"
    if not ann_path.exists():
        return []

    df = pd.read_feather(ann_path)
    annotations = []
    for row in df.itertuples(index=False):
        ann = {
            "category": getattr(row, "category", "unknown"),
            "track_uuid": str(getattr(row, "track_uuid", "")),
            "xyz_m": np.array(
                [
                    float(getattr(row, "tx_m", 0.0)),
                    float(getattr(row, "ty_m", 0.0)),
                    float(getattr(row, "tz_m", 0.0)),
                ],
                dtype=np.float32,
            ),
            "timestamp_ns": int(getattr(row, "timestamp_ns", 0)),
            "length_m": float(getattr(row, "length_m", 0.0)),
            "width_m": float(getattr(row, "width_m", 0.0)),
            "height_m": float(getattr(row, "height_m", 0.0)),
            "yaw_rad": float(getattr(row, "qw", 0.0)),  # AV2 uses qw/qx/qy/qz
        }
        annotations.append(ann)
    return annotations


def _load_sweeps(lidar_dir: Path, max_sweeps: int) -> list[dict]:
    """
    Load LiDAR feather files → list of sweep dicts.

    Each sweep dict:
        timestamp_ns  (int)
        xyz           (np.ndarray float32, shape [N, 3])
        intensity     (np.ndarray float32, shape [N]) — if present
        path          (str)
    """
    if not lidar_dir.exists():
        return []

    files = sorted(lidar_dir.glob("*.feather"))[:max_sweeps]
    sweeps = []
    for f in files:
        df = pd.read_feather(f)
        # Ensure required columns exist
        missing = [c for c in LIDAR_COLS if c not in df.columns]
        if missing:
            continue
        sweep: dict = {
            "timestamp_ns": int(f.stem),
            "xyz": df[LIDAR_COLS].to_numpy(dtype=np.float32),
            "path": str(f),
        }
        if LIDAR_INTENSITY_COL in df.columns:
            sweep["intensity"] = df[LIDAR_INTENSITY_COL].to_numpy(dtype=np.float32)
        sweeps.append(sweep)
    return sweeps


def _load_cameras(cameras_dir: Path, max_frames: int) -> dict[str, dict]:
    """
    Discover camera image files → dict of camera_name → camera info dict.

    Camera info dict:
        count   (int) total images
        paths   (list[str]) paths to first max_frames images, sorted by timestamp
        timestamps_ns (list[int])
    """
    if not cameras_dir.exists():
        return {}

    cameras: dict[str, dict] = {}
    for cam_dir in sorted(cameras_dir.iterdir()):
        if not cam_dir.is_dir():
            continue
        images = sorted(cam_dir.glob("*.jpg"))
        # Also support PNG
        if not images:
            images = sorted(cam_dir.glob("*.png"))
        cameras[cam_dir.name] = {
            "count": len(images),
            "paths": [str(p) for p in images[:max_frames]],
            "timestamps_ns": [int(p.stem) for p in images[:max_frames]],
        }
    return cameras


def _load_ego_poses(log_path: Path) -> list[dict]:
    """
    Load ego-vehicle poses from city_SE3_egovehicle.feather.

    Returns list of dicts with timestamp_ns and translation (x, y, z).
    """
    pose_path = log_path / "city_SE3_egovehicle.feather"
    if not pose_path.exists():
        return []

    df = pd.read_feather(pose_path)
    poses = []
    for row in df.itertuples(index=False):
        poses.append(
            {
                "timestamp_ns": int(getattr(row, "timestamp_ns", 0)),
                "tx_m": float(getattr(row, "tx_m", 0.0)),
                "ty_m": float(getattr(row, "ty_m", 0.0)),
                "tz_m": float(getattr(row, "tz_m", 0.0)),
            }
        )
    return poses


def _infer_city_name(log_path: Path) -> str:
    """
    Try to extract city name from the log directory.
    AV2 cities: ATX, DTW, MIA, PAO, PIT, WDC.
    """
    # Some AV2 releases include city in a metadata file
    meta_path = log_path / "city_name.txt"
    if meta_path.exists():
        return meta_path.read_text().strip()
    # Fall back: check if log_id encodes city
    parts = log_path.name.split("_")
    known_cities = {"ATX", "DTW", "MIA", "PAO", "PIT", "WDC"}
    for part in parts:
        if part.upper() in known_cities:
            return part.upper()
    return "unknown"


def load_scene(
    data_dir: str,
    log_id: Optional[str] = None,
    max_sweeps: int = 30,
    max_frames: int = 30,
) -> dict:
    """
    Load one AV2 sensor scene into a plain Python dict.

    Args:
        data_dir:    Path to the AV2 split directory
                     (e.g. /raid/av2/sensor/val).
        log_id:      Scene log ID. If None, the first scene alphabetically
                     is used.
        max_sweeps:  Maximum number of LiDAR sweeps to load (default 30 ≈ 3s).
        max_frames:  Maximum camera frames to index per camera (default 30).

    Returns:
        dict with keys:
            log_id         str
            city_name      str
            log_path       str
            annotations    list[dict]
            sweeps         list[dict]   — LiDAR
            cameras        dict[str, dict]
            ego_poses      list[dict]
            stats          dict  — quick summary counts
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"AV2 data directory not found: {data_dir}")

    # Resolve log_id
    if log_id is None:
        scenes = _list_scenes(data_path)
        if not scenes:
            raise ValueError(f"No scenes (subdirectories) found in {data_dir}")
        log_id = scenes[0]
        print(f"[load_scene] No log_id specified — using first scene: {log_id}")

    log_path = data_path / log_id
    if not log_path.exists():
        raise FileNotFoundError(f"Scene not found: {log_path}")

    print(f"[load_scene] Loading scene: {log_id}")

    annotations = _load_annotations(log_path)
    sweeps = _load_sweeps(log_path / "sensors" / "lidar", max_sweeps)
    cameras = _load_cameras(log_path / "sensors" / "cameras", max_frames)
    ego_poses = _load_ego_poses(log_path)
    city_name = _infer_city_name(log_path)

    # Quick category counts for prompts
    category_counts: dict[str, int] = {}
    for ann in annotations:
        cat = ann["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    avg_pts = (
        int(np.mean([s["xyz"].shape[0] for s in sweeps])) if sweeps else 0
    )

    stats = {
        "n_annotations": len(annotations),
        "n_sweeps_loaded": len(sweeps),
        "n_cameras": len(cameras),
        "n_ego_poses": len(ego_poses),
        "avg_lidar_points": avg_pts,
        "category_counts": category_counts,
        "duration_s": round(len(sweeps) * 0.1, 1),  # ~10 Hz LiDAR
    }

    scene = {
        "log_id": log_id,
        "city_name": city_name,
        "log_path": str(log_path),
        "annotations": annotations,
        "sweeps": sweeps,
        "cameras": cameras,
        "ego_poses": ego_poses,
        "stats": stats,
    }

    print(
        f"[load_scene] Loaded: {len(sweeps)} LiDAR sweeps, "
        f"{len(cameras)} cameras, {len(annotations)} annotations, "
        f"city={city_name}"
    )
    return scene


def list_scenes(data_dir: str) -> list[str]:
    """Return all scene log IDs in a data directory."""
    return _list_scenes(Path(data_dir))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Load and inspect an AV2 scene")
    parser.add_argument("--data-dir", required=True, help="AV2 split directory")
    parser.add_argument("--log-id", default=None)
    parser.add_argument("--max-sweeps", type=int, default=30)
    parser.add_argument(
        "--list", action="store_true", help="List available scenes and exit"
    )
    args = parser.parse_args()

    if args.list:
        scenes = list_scenes(args.data_dir)
        print(f"Found {len(scenes)} scenes in {args.data_dir}:")
        for s in scenes:
            print(f"  {s}")
    else:
        scene = load_scene(args.data_dir, args.log_id, args.max_sweeps)
        # Print summary (JSON-serialisable subset)
        summary = {
            "log_id": scene["log_id"],
            "city_name": scene["city_name"],
            "stats": scene["stats"],
            "cameras": {k: v["count"] for k, v in scene["cameras"].items()},
        }
        print(json.dumps(summary, indent=2))
