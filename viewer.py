"""
OmniSight Data Viewer — FastAPI web app to preview AV2 scenes on the DGX.

Serves:
    GET /                          — scene list
    GET /scene/{log_id}            — scene overview (stats + camera gallery)
    GET /scene/{log_id}/image/{camera}/{filename}  — serve a raw JPEG
    GET /scene/{log_id}/lidar      — LiDAR stats (JSON)
    GET /scene/{log_id}/bev        — bird's-eye-view PNG of one LiDAR sweep
    GET /scenes                    — JSON list of all scene IDs
    GET /scene/{log_id}/info       — JSON scene summary

Usage on DGX:
    python viewer.py --data-dir /raid/av2/sensor/val --port 7860

Then open http://<dgx-ip>:7860 in your browser (or SSH tunnel:
    ssh -L 7860:localhost:7860 user@dgx  then  http://localhost:7860)
"""

from __future__ import annotations

import argparse
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
    import uvicorn
except ImportError as e:
    raise ImportError("Run: pip install fastapi uvicorn") from e

try:
    from PIL import Image, ImageDraw
except ImportError as e:
    raise ImportError("Run: pip install Pillow") from e

from load_scene import list_scenes, load_scene

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(title="OmniSight AV2 Viewer", version="1.0")

# Set via CLI args / env vars
DATA_DIR: str = os.getenv("AV2_DATA_DIR", "/raid/av2/sensor/val")


# ── Scene cache (keeps last 4 scenes in memory) ──────────────────────────────

@lru_cache(maxsize=4)
def _cached_scene(data_dir: str, log_id: str) -> dict:
    return load_scene(data_dir, log_id)


def get_scene(log_id: str) -> dict:
    try:
        return _cached_scene(DATA_DIR, log_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} — OmniSight</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #e6edf3; }}
    nav {{ background: #161b22; padding: 12px 24px; border-bottom: 1px solid #30363d;
           display: flex; align-items: center; gap: 16px; }}
    nav a {{ color: #58a6ff; text-decoration: none; font-weight: 600; }}
    nav span {{ color: #8b949e; font-size: 14px; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 24px; margin-bottom: 8px; }}
    h2 {{ font-size: 18px; margin: 24px 0 12px; color: #58a6ff; }}
    p, li {{ color: #8b949e; font-size: 14px; line-height: 1.6; }}
    .badge {{ display: inline-block; background: #21262d; border: 1px solid #30363d;
              border-radius: 6px; padding: 2px 8px; font-size: 12px; color: #79c0ff; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
             padding: 16px; margin-bottom: 16px; }}
    .scene-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                   gap: 16px; }}
    .scene-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                   padding: 16px; transition: border-color .2s; }}
    .scene-card:hover {{ border-color: #58a6ff; }}
    .scene-card a {{ color: #58a6ff; text-decoration: none; font-size: 13px; font-family: monospace; }}
    .cam-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
                 gap: 12px; }}
    .cam-block {{ background: #0d1117; border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }}
    .cam-block img {{ width: 100%; display: block; }}
    .cam-label {{ padding: 6px 10px; font-size: 12px; color: #8b949e; }}
    .stat-row {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }}
    .stat {{ background: #21262d; border-radius: 6px; padding: 10px 16px; }}
    .stat-val {{ font-size: 22px; font-weight: 700; color: #f0f6fc; }}
    .stat-lbl {{ font-size: 11px; color: #8b949e; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border-bottom: 1px solid #21262d; text-align: left; }}
    th {{ color: #8b949e; font-weight: 500; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px;
            font-family: monospace; font-size: 13px; }}
    .bev-img {{ border-radius: 6px; border: 1px solid #30363d; }}
  </style>
</head>
<body>
  <nav>
    <a href="/">OmniSight</a>
    <span>AV2 Data Viewer</span>
  </nav>
  <div class="container">
    {body}
  </div>
</body>
</html>"""


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    """Scene list page."""
    try:
        scenes = list_scenes(DATA_DIR)
    except Exception as e:
        body = f"<h1>Error</h1><p>Could not read data directory: <code>{DATA_DIR}</code><br>{e}</p>"
        return HTMLResponse(_html_page("Error", body))

    if not scenes:
        body = f"<h1>No scenes found</h1><p>Directory: <code>{DATA_DIR}</code></p>"
        return HTMLResponse(_html_page("No scenes", body))

    cards = ""
    for s in scenes:
        cards += f"""
        <div class="scene-card">
          <a href="/scene/{s}">{s[:40]}</a>
          <p style="margin-top:8px;font-size:12px">
            <a href="/scene/{s}" style="color:#3fb950">▶ View scene</a>
          </p>
        </div>"""

    body = f"""
    <h1>AV2 Scenes</h1>
    <p style="margin-bottom:16px">Data directory: <code>{DATA_DIR}</code> &nbsp;
       <span class="badge">{len(scenes)} scenes</span></p>
    <div class="scene-grid">{cards}</div>"""
    return HTMLResponse(_html_page("Scenes", body))


@app.get("/scene/{log_id}", response_class=HTMLResponse)
def scene_page(log_id: str, sweep_idx: int = 0):
    """Main scene viewer: stats + camera gallery + BEV."""
    scene = get_scene(log_id)
    stats = scene["stats"]
    cameras = scene["cameras"]

    # ── Stats bar ────────────────────────────────────────────────────────
    stat_items = [
        (stats["n_sweeps_loaded"], "LiDAR sweeps"),
        (stats["avg_lidar_points"], "avg pts/sweep"),
        (stats["n_annotations"], "annotations"),
        (stats["n_cameras"], "cameras"),
        (f"{stats['duration_s']}s", "duration"),
        (scene["city_name"], "city"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="stat-val">{v}</div>'
        f'<div class="stat-lbl">{l}</div></div>'
        for v, l in stat_items
    )

    # ── Category table ───────────────────────────────────────────────────
    cc = stats.get("category_counts", {})
    rows = "".join(
        f"<tr><td>{cat}</td><td>{cnt}</td></tr>"
        for cat, cnt in sorted(cc.items(), key=lambda x: -x[1])
    )
    cat_table = f"""
    <table>
      <thead><tr><th>Category</th><th>Count</th></tr></thead>
      <tbody>{rows if rows else '<tr><td colspan="2">No annotations</td></tr>'}</tbody>
    </table>"""

    # ── Camera gallery ────────────────────────────────────────────────────
    cam_blocks = ""
    for cam_name, cam_info in cameras.items():
        paths = cam_info.get("paths", [])
        if not paths:
            continue
        # Show middle frame
        mid = paths[len(paths) // 2]
        filename = Path(mid).name
        img_url = f"/scene/{log_id}/image/{cam_name}/{filename}"
        cam_blocks += f"""
        <div class="cam-block">
          <img src="{img_url}" loading="lazy" alt="{cam_name}">
          <div class="cam-label">{cam_name} &nbsp;·&nbsp; {cam_info['count']} frames</div>
        </div>"""

    # ── BEV section ───────────────────────────────────────────────────────
    bev_html = ""
    if stats["n_sweeps_loaded"] > 0:
        bev_url = f"/scene/{log_id}/bev?sweep_idx={sweep_idx}"
        bev_html = f"""
        <h2>Bird's-Eye View (LiDAR sweep {sweep_idx})</h2>
        <img src="{bev_url}" class="bev-img" alt="BEV" style="max-width:600px">"""

    body = f"""
    <p style="margin-bottom:4px"><a href="/" style="color:#58a6ff">← All scenes</a></p>
    <h1>{log_id}</h1>
    <p style="margin-bottom:16px;font-size:13px">
      City: <strong>{scene['city_name']}</strong> &nbsp;
      <a href="/scene/{log_id}/info" style="color:#8b949e;font-size:12px">JSON ↗</a>
    </p>

    <div class="stat-row">{stat_html}</div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="card">
        <h2 style="margin-top:0">Object Categories</h2>
        {cat_table}
      </div>
      <div class="card">
        <h2 style="margin-top:0">LiDAR Summary</h2>
        <p>Sweeps loaded: {stats['n_sweeps_loaded']} of {stats['n_sweeps_loaded']}</p>
        <p>Avg points/sweep: {stats['avg_lidar_points']:,}</p>
        <p>Estimated duration: {stats['duration_s']}s @ 10 Hz</p>
        <p style="margin-top:8px">
          <a href="/scene/{log_id}/lidar" style="color:#58a6ff;font-size:13px">
            Full LiDAR JSON ↗
          </a>
        </p>
      </div>
    </div>

    <h2>Camera Images</h2>
    <div class="cam-grid">{cam_blocks if cam_blocks else '<p>No camera images found.</p>'}</div>

    {bev_html}"""

    return HTMLResponse(_html_page(log_id[:20], body))


@app.get("/scene/{log_id}/image/{camera}/{filename}")
def serve_image(log_id: str, camera: str, filename: str):
    """Serve a raw camera JPEG."""
    scene = get_scene(log_id)
    cam_info = scene["cameras"].get(camera)
    if cam_info is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera}' not found")

    # Find the path that matches this filename
    target = None
    for p in cam_info["paths"]:
        if Path(p).name == filename:
            target = Path(p)
            break

    if target is None or not target.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    data = target.read_bytes()
    media = "image/jpeg" if filename.endswith(".jpg") else "image/png"
    return Response(content=data, media_type=media)


@app.get("/scene/{log_id}/lidar")
def lidar_stats(log_id: str):
    """Return per-sweep LiDAR statistics as JSON."""
    scene = get_scene(log_id)
    sweeps = scene["sweeps"]
    result = []
    for s in sweeps:
        xyz = s["xyz"]
        result.append(
            {
                "timestamp_ns": s["timestamp_ns"],
                "n_points": int(xyz.shape[0]),
                "x_range": [round(float(xyz[:, 0].min()), 2), round(float(xyz[:, 0].max()), 2)],
                "y_range": [round(float(xyz[:, 1].min()), 2), round(float(xyz[:, 1].max()), 2)],
                "z_range": [round(float(xyz[:, 2].min()), 2), round(float(xyz[:, 2].max()), 2)],
            }
        )
    return JSONResponse({"log_id": log_id, "sweeps": result})


@app.get("/scene/{log_id}/bev")
def bird_eye_view(log_id: str, sweep_idx: int = 0, size: int = 600, range_m: float = 50.0):
    """
    Render a bird's-eye-view PNG of one LiDAR sweep.

    Projects (x, y) to image pixels, colours by height (z).
    """
    scene = get_scene(log_id)
    sweeps = scene["sweeps"]

    if not sweeps:
        raise HTTPException(status_code=404, detail="No LiDAR sweeps loaded")

    idx = min(sweep_idx, len(sweeps) - 1)
    xyz = sweeps[idx]["xyz"]

    # Filter to range_m radius
    dist = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    mask = dist < range_m
    xyz = xyz[mask]

    # Map to pixel coords
    scale = (size / 2) / range_m
    px = ((xyz[:, 0] * scale) + size / 2).astype(int)
    py = ((-xyz[:, 1] * scale) + size / 2).astype(int)

    # Clamp
    px = np.clip(px, 0, size - 1)
    py = np.clip(py, 0, size - 1)

    # Colour by z height (dark blue = low, bright green = high)
    z = xyz[:, 2]
    z_norm = np.clip((z - z.min()) / (z.max() - z.min() + 1e-6), 0, 1)

    img_arr = np.zeros((size, size, 3), dtype=np.uint8)
    img_arr[py, px, 0] = (z_norm * 80).astype(np.uint8)          # R
    img_arr[py, px, 1] = (z_norm * 200 + 55).astype(np.uint8)    # G
    img_arr[py, px, 2] = (255 - z_norm * 200).astype(np.uint8)   # B

    # Draw ego-vehicle marker
    img = Image.fromarray(img_arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(255, 80, 80))

    # Draw annotation boxes
    annotations = scene["annotations"]
    if annotations:
        sweep_ts = sweeps[idx]["timestamp_ns"]
        for ann in annotations:
            if abs(ann["timestamp_ns"] - sweep_ts) < 60_000_000:  # within 60ms
                ax = int(ann["xyz_m"][0] * scale + size / 2)
                ay = int(-ann["xyz_m"][1] * scale + size / 2)
                if 0 <= ax < size and 0 <= ay < size:
                    draw.rectangle([ax - 4, ay - 4, ax + 4, ay + 4],
                                   outline=(255, 200, 0), width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/scene/{log_id}/info")
def scene_info(log_id: str):
    """JSON summary of the scene (no large numpy arrays)."""
    scene = get_scene(log_id)
    return JSONResponse(
        {
            "log_id": scene["log_id"],
            "city_name": scene["city_name"],
            "log_path": scene["log_path"],
            "stats": scene["stats"],
            "cameras": {k: {"count": v["count"]} for k, v in scene["cameras"].items()},
        }
    )


@app.get("/scenes")
def scenes_json():
    """JSON list of all scene IDs."""
    try:
        scenes = list_scenes(DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return JSONResponse({"data_dir": DATA_DIR, "scenes": scenes})


@app.get("/health")
def health():
    return {"status": "ok", "data_dir": DATA_DIR}


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    global DATA_DIR

    parser = argparse.ArgumentParser(description="OmniSight AV2 Data Viewer")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("AV2_DATA_DIR", "/raid/av2/sensor/val"),
        help="Path to AV2 split directory (default: /raid/av2/sensor/val)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload")
    args = parser.parse_args()

    DATA_DIR = args.data_dir

    print("=" * 60)
    print("  OmniSight AV2 Data Viewer")
    print("=" * 60)
    print(f"  Data dir : {DATA_DIR}")
    print(f"  URL      : http://{args.host}:{args.port}")
    print()
    print("  SSH tunnel (from your laptop):")
    print(f"    ssh -L {args.port}:localhost:{args.port} user@<dgx-ip>")
    print(f"  Then open: http://localhost:{args.port}")
    print("=" * 60)

    uvicorn.run(
        "viewer:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
