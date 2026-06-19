"""
view_clusters.py

Interactive DBSCAN tuner with density control and config save/load.

Toggle at the top:
  USE_SAVED_CONFIG = True   → load dbscan_config.json and run directly (no sliders)
  USE_SAVED_CONFIG = False  → show sliders; saves to dbscan_config.json on exit

Sliders:
  Point Density   -3 to +3
                   0  = original cloud
                  -ve = sparser  (stride subsampling)
                  +ve = denser   (KNN edge interpolation per cluster)
  eps             neighbourhood radius (×0.001 m)
  Min Samples     minimum points per cluster
  Show Cluster    0=all, 1=cluster0, 2=cluster1 ...

NOTE: sliders are created ONCE at startup and never recreated.
      Points are cleared and redrawn using removeAllUserDebugItems()
      followed by re-adding sliders at their current positions,
      BUT only the point cloud changes — sliders keep their values
      because we pass the current values back as startValue.
"""

import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import pybullet as p
import json
import os
import time


# ══════════════════════════════════════════════════════════════
# TOGGLE + CONFIG
# ══════════════════════════════════════════════════════════════

USE_SAVED_CONFIG = False
CONFIG_FILE      = "dbscan_config.json"

DEFAULTS = {
    "density":      0,
    "eps":          0.082,
    "min_samples":  3,
    "show_cluster": 0,
}


# ══════════════════════════════════════════════════════════════
# CONFIG SAVE / LOAD
# ══════════════════════════════════════════════════════════════

def save_config(cfg, path=CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Config saved → {path}")
    print(f"       density={cfg['density']}  eps={cfg['eps']:.4f}m  "
          f"min_samples={cfg['min_samples']}")


def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded config from {path}")
        return cfg
    print(f"[INFO] No config found — using defaults")
    return None


# ══════════════════════════════════════════════════════════════
# PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
armId = p.loadURDF(
    "arm_3.urdf", [0,0,0], p.getQuaternionFromEuler([0,0,0]),
    useFixedBase=True,
    flags=(p.URDF_USE_INERTIA_FROM_FILE |
           p.URDF_USE_SELF_COLLISION     |
           p.URDF_USE_IMPLICIT_CYLINDER)
)


# ══════════════════════════════════════════════════════════════
# LOAD POINTCLOUD
# ══════════════════════════════════════════════════════════════

ALL_POINTS = np.load("points.npy")
print(f"[INFO] Loaded {len(ALL_POINTS)} points")


# ══════════════════════════════════════════════════════════════
# DENSITY
#
#  density = 0  → original cloud unchanged
#
#  density < 0  → subsample: stride = 1 + abs(density)
#                  -1 → every 2nd point  (~50%)
#                  -2 → every 3rd point  (~33%)
#                  -3 → every 4th point  (~25%)
#
#  density > 0  → interpolate new points along KNN edges per cluster
#                 For each point, find k nearest neighbours.
#                 Insert `density` evenly spaced midpoints on each edge.
#                 Stays local to the surface — no cross-gap points.
#                  +1 → 1 midpoint per edge   (~2× points)
#                  +2 → 2 midpoints per edge  (~3× points)
#                  +3 → 3 midpoints per edge  (~4× points)
# ══════════════════════════════════════════════════════════════

def interpolate_cluster(pts: np.ndarray, n_interp: int,
                        k_neighbours: int = 6) -> np.ndarray:
    if len(pts) < 2 or n_interp < 1:
        return pts

    k      = min(k_neighbours + 1, len(pts))
    tree   = cKDTree(pts)
    _, idx = tree.query(pts, k=k)

    new_pts    = [pts]
    seen_edges = set()

    for i in range(len(pts)):
        for j in idx[i, 1:]:
            edge = (min(i, j), max(i, j))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            for t in np.linspace(0, 1, n_interp + 2)[1:-1]:
                new_pts.append((pts[i] * (1-t) + pts[j] * t).reshape(1, 3))

    return np.vstack(new_pts)


def apply_density(all_points: np.ndarray, labels: np.ndarray,
                  density: int) -> tuple:
    density = int(round(density))

    if density == 0:
        return all_points.copy(), labels.copy()

    if density < 0:
        stride = 1 + abs(density)
        return all_points[::stride].copy(), labels[::stride].copy()

    # density > 0: interpolate per cluster, leave noise as-is
    out_pts, out_lbl = [], []
    for lbl in np.unique(labels):
        mask = labels == lbl
        cpts = all_points[mask]
        if lbl == -1:
            out_pts.append(cpts)
            out_lbl.append(np.full(len(cpts), -1, dtype=int))
        else:
            dense = interpolate_cluster(cpts, n_interp=density)
            out_pts.append(dense)
            out_lbl.append(np.full(len(dense), lbl, dtype=int))

    return np.vstack(out_pts), np.concatenate(out_lbl)


# ══════════════════════════════════════════════════════════════
# CLUSTERING
# ══════════════════════════════════════════════════════════════

MERGE_THRESHOLD = 100

def run_clustering(points: np.ndarray, eps: float,
                   min_samples: int) -> np.ndarray:
    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_.copy()

    unique_labels, counts = np.unique(labels, return_counts=True)
    small = [lbl for lbl, cnt in zip(unique_labels, counts)
             if lbl != -1 and cnt < MERGE_THRESHOLD]

    for small_lbl in small:
        small_mask   = labels == small_lbl
        small_points = points[small_mask]
        distances    = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1:
                continue
            if not np.any(labels == lbl):
                continue
            centroid = points[labels == lbl].mean(axis=0)
            distances.append((lbl, np.mean(
                np.linalg.norm(small_points - centroid, axis=1))))
        if distances:
            closest = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest

    return labels


# ══════════════════════════════════════════════════════════════
# COLOUR MAP
# ══════════════════════════════════════════════════════════════

CLUSTER_COLORS = [
    [1.0, 0.2, 0.0],   # 0 red-orange
    [0.1, 0.9, 0.1],   # 1 green
    [0.2, 0.5, 1.0],   # 2 blue
    [1.0, 0.9, 0.0],   # 3 yellow
    [0.0, 0.9, 0.9],   # 4 cyan
    [0.9, 0.2, 0.9],   # 5 magenta
]
NOISE_COLOR = [0.45, 0.45, 0.45]

def label_to_color(lbl):
    return NOISE_COLOR if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]


# ══════════════════════════════════════════════════════════════
# POINT RENDERING
#
#   We track point debug IDs separately from slider IDs.
#   On each update we only remove the OLD point items by ID,
#   leaving sliders completely untouched.
#   This avoids the labels-vanishing bug without recreating sliders.
# ══════════════════════════════════════════════════════════════

_point_ids = []

def clear_points():
    global _point_ids
    for did in _point_ids:
        try:
            p.removeUserDebugItem(did)
        except Exception:
            pass
    _point_ids = []


def render_clusters(pts: np.ndarray, labels: np.ndarray,
                    show_cluster: int, point_size: float = 4.0):
    global _point_ids
    clear_points()
    if len(pts) == 0:
        return
    for lbl in np.unique(labels):
        if show_cluster != 0 and lbl != (show_cluster - 1):
            continue
        mask  = labels == lbl
        cpts  = pts[mask]
        if len(cpts) == 0:
            continue
        color = label_to_color(lbl)
        did   = p.addUserDebugPoints(
            pointPositions=cpts.tolist(),
            pointColorsRGB=[color] * len(cpts),
            pointSize=point_size
        )
        _point_ids.append(did)


def print_summary(labels, eps, min_samples, density, n_pts):
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters = int(np.sum(unique >= 0))
    n_noise    = int(np.sum(labels == -1))
    print(f"\n{'─'*55}")
    print(f"  density={density:+d}  eps={eps:.4f}m  "
          f"min_samples={min_samples}  pts={n_pts}")
    print(f"  clusters={n_clusters}  noise={n_noise}")
    for lbl, cnt in zip(unique, counts):
        tag = "noise" if lbl == -1 else f"cluster {lbl}"
        print(f"    {tag}: {cnt} pts")
    print(f"{'─'*55}")


# ══════════════════════════════════════════════════════════════
# STATIC MODE  (USE_SAVED_CONFIG = True)
# ══════════════════════════════════════════════════════════════

saved = load_config()
cfg   = saved if (USE_SAVED_CONFIG and saved) else dict(DEFAULTS)

if USE_SAVED_CONFIG and saved:
    print("\n[INFO] USE_SAVED_CONFIG=True — running with saved config\n")
    base_labels      = run_clustering(ALL_POINTS, cfg["eps"], cfg["min_samples"])
    pts, lbl         = apply_density(ALL_POINTS, base_labels, cfg["density"])
    render_clusters(pts, lbl, cfg["show_cluster"])
    print_summary(lbl, cfg["eps"], cfg["min_samples"], cfg["density"], len(pts))

    for _ in range(100000):
        try:
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
        except Exception:
            break

    p.disconnect()
    exit()


# ══════════════════════════════════════════════════════════════
# SLIDERS  (created once, never recreated)
# ══════════════════════════════════════════════════════════════

init_density  = int(cfg["density"])
init_eps_raw  = int(round(cfg["eps"] * 1000))
init_min_samp = int(cfg["min_samples"])
init_show     = int(cfg["show_cluster"])

sid_density = p.addUserDebugParameter(
    "Point Density  (-3=sparse  0=original  +3=denser)",
    -3, 3, init_density
)
sid_eps = p.addUserDebugParameter(
    "Neighbourhood Radius eps  (x0.001m)",
    10, 200, init_eps_raw
)
sid_min_samples = p.addUserDebugParameter(
    "Min Samples per Cluster",
    1, 20, init_min_samp
)
sid_show = p.addUserDebugParameter(
    "Show Cluster  (0=all  1=cluster0  2=cluster1 ...)",
    0, 8, init_show
)

print("\n[INFO] Sliders ready — adjust in PyBullet GUI.")
print("       Close PyBullet window to save config and exit.\n")


# ══════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════

prev_density     = None
prev_eps         = None
prev_min_samples = None
prev_show        = None

# Cache base clustering to avoid re-running DBSCAN on density-only changes
cached_base_pts    = None
cached_base_labels = None

while True:
    try:
        density_raw  = p.readUserDebugParameter(sid_density)
        eps_raw      = p.readUserDebugParameter(sid_eps)
        min_samp_raw = p.readUserDebugParameter(sid_min_samples)
        show_raw     = p.readUserDebugParameter(sid_show)
    except Exception:
        print("[INFO] PyBullet window closed.")
        break

    density      = int(round(density_raw))
    eps          = round(eps_raw / 1000.0, 4)
    min_samples  = int(round(min_samp_raw))
    show_cluster = int(round(show_raw))

    cluster_changed = (eps != prev_eps or min_samples != prev_min_samples)
    any_changed     = (density      != prev_density     or
                       cluster_changed                   or
                       show_cluster != prev_show)

    if any_changed:
        if cluster_changed or cached_base_labels is None:
            cached_base_labels = run_clustering(ALL_POINTS, eps, min_samples)
            cached_base_pts    = ALL_POINTS

        pts, lbl = apply_density(cached_base_pts, cached_base_labels, density)
        render_clusters(pts, lbl, show_cluster)
        print_summary(lbl, eps, min_samples, density, len(pts))

        prev_density     = density
        prev_eps         = eps
        prev_min_samples = min_samples
        prev_show        = show_cluster

    p.stepSimulation()
    time.sleep(1.0 / 240.0)


# ══════════════════════════════════════════════════════════════
# SAVE CONFIG + UPDATED POINTCLOUD ON EXIT
# ══════════════════════════════════════════════════════════════

final_cfg = {
    "density":      prev_density     if prev_density     is not None else DEFAULTS["density"],
    "eps":          prev_eps         if prev_eps         is not None else DEFAULTS["eps"],
    "min_samples":  prev_min_samples if prev_min_samples is not None else DEFAULTS["min_samples"],
    "show_cluster": prev_show        if prev_show        is not None else DEFAULTS["show_cluster"],
}
save_config(final_cfg)

# Save the final (possibly interpolated/subsampled) pointcloud
# pts and lbl are the last rendered arrays from the loop
try:
    if pts is not None and len(pts) > 0:
        np.save("points.npy", pts)
        print(f"[INFO] points.npy updated → {len(pts)} points saved")
        if len(pts) != len(ALL_POINTS):
            diff = len(pts) - len(ALL_POINTS)
            print(f"       ({'+' if diff > 0 else ''}{diff} vs original {len(ALL_POINTS)})")
    else:
        print("[WARN] No points to save — points.npy unchanged")
except NameError:
    print("[WARN] Loop never ran — points.npy unchanged")

p.disconnect()
print("[INFO] Done.")