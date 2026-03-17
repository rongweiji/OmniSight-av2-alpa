"""
OmniSight Data API — FastAPI backend serving AV2 scene data to the Next.js frontend.

Replaces viewer.py with a proper REST API.

Run:
    python -m api.server --data-dir /raid/av2/sensor/val --port 8080

Endpoints:
    GET /api/scenes                               — list all scene IDs
    GET /api/scenes/{log_id}                      — scene metadata + full timeline
    GET /api/scenes/{log_id}/lidar/{ts}           — LiDAR point cloud (JSON)
    GET /api/scenes/{log_id}/camera/{cam}/{ts}    — camera JPEG/PNG
    GET /api/scenes/{log_id}/annotations/{ts}     — 3D annotation boxes near ts
    GET /health
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="OmniSight API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR: str = os.getenv("AV2_DATA_DIR", "/raid/av2/sensor/val")
LIDAR_COLS = ["x", "y", "z"]

# Category → colour (hex) for annotation boxes in the frontend
CATEGORY_COLORS: dict[str, str] = {
    "VEHICLE": "#ff4444",
    "PEDESTRIAN": "#44ff44",
    "BICYCLIST": "#4444ff",
    "MOTORCYCLIST": "#ffaa00",
    "BUS": "#ff44ff",
    "TRUCK": "#ff8800",
    "CONSTRUCTION_VEHICLE": "#888800",
    "VEHICULAR_TRAILER": "#008888",
    "BOX_TRUCK": "#884400",
    "REGULAR_VEHICLE": "#ff6666",
    "LARGE_VEHICLE": "#cc4400",
}


def _root() -> Path:
    return Path(DATA_DIR)


def _log(log_id: str) -> Path:
    p = _root() / log_id
    if not p.exists():
        raise HTTPException(404, f"Scene not found: {log_id}")
    return p


def _nearest_ts(timestamps: list[int], target: int) -> int:
    """Return the timestamp in the list closest to target."""
    if not timestamps:
        return target
    return min(timestamps, key=lambda t: abs(t - target))


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/scenes")
def list_scenes():
    root = _root()
    if not root.exists():
        raise HTTPException(404, f"Data directory not found: {DATA_DIR}")
    scenes = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    return {"scenes": scenes, "count": len(scenes), "data_dir": DATA_DIR}


@app.get("/api/scenes/{log_id}")
def scene_info(log_id: str):
    """
    Full scene metadata including the complete sorted list of LiDAR timestamps
    (used as the master playback timeline) and camera timestamps per camera.
    """
    log_path = _log(log_id)

    # LiDAR timestamps — master clock
    lidar_dir = log_path / "sensors" / "lidar"
    lidar_ts: list[int] = []
    if lidar_dir.exists():
        lidar_ts = sorted(int(f.stem) for f in lidar_dir.glob("*.feather"))

    # Camera timestamps per camera
    cams_dir = log_path / "sensors" / "cameras"
    camera_ts: dict[str, list[int]] = {}
    if cams_dir.exists():
        for cam_dir in sorted(cams_dir.iterdir()):
            if not cam_dir.is_dir():
                continue
            ts = sorted(int(f.stem) for f in cam_dir.glob("*.jpg"))
            if not ts:
                ts = sorted(int(f.stem) for f in cam_dir.glob("*.png"))
            if ts:
                camera_ts[cam_dir.name] = ts

    # City name
    city_name = "unknown"
    city_file = log_path / "city_name.txt"
    if city_file.exists():
        city_name = city_file.read_text().strip()

    # Annotation count
    ann_path = log_path / "annotations.feather"
    n_annotations = 0
    if ann_path.exists():
        n_annotations = len(pd.read_feather(ann_path))

    duration_s = 0.0
    if len(lidar_ts) > 1:
        duration_s = round((lidar_ts[-1] - lidar_ts[0]) / 1e9, 2)

    return {
        "log_id": log_id,
        "city_name": city_name,
        "lidar_timestamps": lidar_ts,
        "camera_timestamps": camera_ts,
        "camera_names": list(camera_ts.keys()),
        "n_lidar_frames": len(lidar_ts),
        "n_annotations": n_annotations,
        "duration_s": duration_s,
    }


@app.get("/api/scenes/{log_id}/lidar/{timestamp_ns}")
def get_lidar(log_id: str, timestamp_ns: int, max_points: int = 25000):
    """
    Return a LiDAR sweep as a JSON array of {x, y, z} points coloured by height.
    Points are downsampled to max_points for fast transfer.
    Coordinate system: AV2 (x=forward, y=left, z=up) — the frontend converts.
    """
    f = _log(log_id) / "sensors" / "lidar" / f"{timestamp_ns}.feather"
    if not f.exists():
        raise HTTPException(404, f"LiDAR frame not found: {timestamp_ns}")

    df = pd.read_feather(f)
    missing = [c for c in LIDAR_COLS if c not in df.columns]
    if missing:
        raise HTTPException(500, f"Missing LiDAR columns: {missing}")

    xyz = df[LIDAR_COLS].to_numpy(dtype=np.float32)

    # Intensity (optional)
    intensity: np.ndarray | None = None
    if "intensity" in df.columns:
        intensity = df["intensity"].to_numpy(dtype=np.float32)

    # Downsample
    if len(xyz) > max_points:
        idx = np.random.choice(len(xyz), max_points, replace=False)
        xyz = xyz[idx]
        if intensity is not None:
            intensity = intensity[idx]

    return JSONResponse({
        "timestamp_ns": timestamp_ns,
        "n_points": len(xyz),
        "points": xyz.tolist(),
        "intensity": intensity.tolist() if intensity is not None else None,
        "bounds": {
            "x": [float(xyz[:, 0].min()), float(xyz[:, 0].max())],
            "y": [float(xyz[:, 1].min()), float(xyz[:, 1].max())],
            "z": [float(xyz[:, 2].min()), float(xyz[:, 2].max())],
        },
    })


@app.get("/api/scenes/{log_id}/camera/{camera_name}/{timestamp_ns}")
def get_camera(log_id: str, camera_name: str, timestamp_ns: int):
    """Serve the exact camera frame for the given timestamp."""
    base = _log(log_id) / "sensors" / "cameras" / camera_name
    jpg = base / f"{timestamp_ns}.jpg"
    png = base / f"{timestamp_ns}.png"
    if jpg.exists():
        return Response(jpg.read_bytes(), media_type="image/jpeg")
    if png.exists():
        return Response(png.read_bytes(), media_type="image/png")
    raise HTTPException(404, f"Camera image not found: {camera_name}/{timestamp_ns}")


@app.get("/api/scenes/{log_id}/annotations/{timestamp_ns}")
def get_annotations(log_id: str, timestamp_ns: int, window_ns: int = 55_000_000):
    """
    Return 3D annotation boxes within window_ns of the given timestamp.
    Used by the LiDAR viewer to draw bounding boxes.
    """
    ann_path = _log(log_id) / "annotations.feather"
    if not ann_path.exists():
        return {"annotations": []}

    df = pd.read_feather(ann_path)
    if "timestamp_ns" in df.columns:
        df = df[abs(df["timestamp_ns"] - timestamp_ns) <= window_ns]

    result = []
    for row in df.itertuples(index=False):
        cat = getattr(row, "category", "unknown")
        result.append({
            "category": cat,
            "color": CATEGORY_COLORS.get(cat, "#ffff00"),
            "track_uuid": str(getattr(row, "track_uuid", ""))[:8],
            "x": float(getattr(row, "tx_m", 0)),
            "y": float(getattr(row, "ty_m", 0)),
            "z": float(getattr(row, "tz_m", 0)),
            "length": float(getattr(row, "length_m", 1)),
            "width": float(getattr(row, "width_m", 1)),
            "height": float(getattr(row, "height_m", 1)),
        })
    return {"timestamp_ns": timestamp_ns, "annotations": result}


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": DATA_DIR}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="OmniSight Data API")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AV2_DATA_DIR", "/raid/av2/sensor/val"),
        help="AV2 split directory (default: /raid/av2/sensor/val)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    print("=" * 55)
    print("  OmniSight Data API")
    print("=" * 55)
    print(f"  Data dir : {DATA_DIR}")
    print(f"  API URL  : http://{args.host}:{args.port}")
    print(f"  Docs     : http://{args.host}:{args.port}/docs")
    print("=" * 55)

    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
