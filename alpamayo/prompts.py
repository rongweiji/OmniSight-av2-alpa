"""
Prompt templates for Alpamayo-R1-10B scene explanation tasks.
"""

SYSTEM_PROMPT = """\
You are an autonomous driving perception expert with deep knowledge of 3D scene understanding.
You analyze sensor data from the Argoverse 2 dataset (lidar sweeps + camera images + 3D annotations)
and provide clear, structured explanations of what is happening in the scene.
Think step by step before answering.\
"""


def scene_summary_prompt(scene: dict) -> str:
    """
    Generate a prompt asking the model to summarize the overall scene.
    """
    anns = scene.get("annotations", [])
    sweeps = scene.get("sweeps", [])
    cameras = list(scene.get("cameras", {}).keys())
    city = scene.get("city_name", "unknown")
    log_id = scene.get("log_id", "unknown")

    # Count objects by category
    category_counts: dict[str, int] = {}
    for ann in anns:
        cat = ann.get("category", "unknown")
        category_counts[cat] = category_counts.get(cat, 0) + 1

    category_str = "\n".join(
        f"  - {cat}: {count}" for cat, count in sorted(category_counts.items())
    )

    return f"""\
Scene ID: {log_id}
City: {city}
LiDAR sweeps: {len(sweeps)} (temporal sequence)
Cameras: {', '.join(cameras)}
Total annotated objects: {len(anns)}

Object categories detected:
{category_str if category_str else "  (none)"}

Task: Provide a structured scene summary covering:
1. Overall scene type (intersection, highway, urban street, etc.)
2. Key objects and their significance for safe driving
3. Potential hazards or points of attention
4. Scene complexity assessment\
"""


def object_behavior_prompt(scene: dict, category: str | None = None) -> str:
    """
    Generate a prompt asking the model to explain object behaviors in the scene.
    """
    anns = scene.get("annotations", [])
    sweeps = scene.get("sweeps", [])

    if category:
        anns = [a for a in anns if a.get("category") == category]

    # Group by track to understand trajectories
    tracks: dict[str, list] = {}
    for ann in anns:
        tid = ann.get("track_uuid", "unknown")
        tracks.setdefault(tid, []).append(ann)

    track_summary = []
    for tid, track_anns in list(tracks.items())[:10]:  # limit to 10 tracks
        cat = track_anns[0].get("category", "unknown")
        n_frames = len(track_anns)
        positions = [a["xyz_m"].tolist() if hasattr(a.get("xyz_m", None), "tolist") else a.get("xyz_m", [0, 0, 0]) for a in track_anns]
        start = positions[0]
        end = positions[-1]
        track_summary.append(
            f"  track={tid[:8]}  category={cat}  frames={n_frames}"
            f"  start_xy=({start[0]:.1f}, {start[1]:.1f})"
            f"  end_xy=({end[0]:.1f}, {end[1]:.1f})"
        )

    tracks_str = "\n".join(track_summary) if track_summary else "  (no tracks)"
    filter_note = f" (filtered to: {category})" if category else ""

    return f"""\
Scene ID: {scene.get('log_id', 'unknown')} | City: {scene.get('city_name', 'unknown')}
Total sweeps: {len(sweeps)} | Duration: ~{len(sweeps) * 0.1:.1f}s
Object tracks{filter_note}:
{tracks_str}

Task: Analyze the object tracks above and explain:
1. Movement patterns — which objects are moving, stopped, or turning?
2. Interactions — are any objects interacting with each other?
3. Predicted intent — what are the objects likely to do next?
4. Safety relevance — which objects require the ego vehicle's attention?\
"""


def lidar_density_prompt(scene: dict) -> str:
    """
    Generate a prompt asking the model to interpret lidar point cloud statistics.
    """
    sweeps = scene.get("sweeps", [])
    if not sweeps:
        return "No lidar sweeps available."

    import numpy as np

    point_counts = [s["xyz"].shape[0] for s in sweeps]
    avg_pts = int(np.mean(point_counts))
    min_pts = min(point_counts)
    max_pts = max(point_counts)

    # Spatial extent from first sweep
    first_xyz = sweeps[0]["xyz"]
    x_range = (float(first_xyz[:, 0].min()), float(first_xyz[:, 0].max()))
    y_range = (float(first_xyz[:, 1].min()), float(first_xyz[:, 1].max()))
    z_range = (float(first_xyz[:, 2].min()), float(first_xyz[:, 2].max()))

    return f"""\
LiDAR point cloud statistics:
  Sweeps: {len(sweeps)}
  Points per sweep — avg: {avg_pts:,}  min: {min_pts:,}  max: {max_pts:,}
  Spatial extent (first sweep):
    X (forward): [{x_range[0]:.1f}, {x_range[1]:.1f}] m
    Y (lateral): [{y_range[0]:.1f}, {y_range[1]:.1f}] m
    Z (height):  [{z_range[0]:.1f}, {z_range[1]:.1f}] m

Task: Based on the lidar statistics above:
1. What does the point density suggest about the scene environment?
2. What is the likely sensor range and coverage?
3. Are there signs of occlusion or sparse regions?
4. What driving conditions might these statistics indicate?\
"""


def custom_prompt(scene: dict, question: str) -> str:
    """
    Build a prompt for a custom free-form question about the scene.
    """
    anns = scene.get("annotations", [])
    sweeps = scene.get("sweeps", [])
    cameras = list(scene.get("cameras", {}).keys())

    return f"""\
Scene context:
  Scene ID: {scene.get('log_id', 'unknown')}
  City: {scene.get('city_name', 'unknown')}
  LiDAR sweeps: {len(sweeps)}
  Cameras: {', '.join(cameras)}
  Annotated objects: {len(anns)}

Question: {question}\
"""
