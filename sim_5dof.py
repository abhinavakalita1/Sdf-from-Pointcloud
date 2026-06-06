import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass, field
from typing import Optional
import json
import os


# ══════════════════════════════════════════════════════════════
# 1.  AABB-BVH
# ══════════════════════════════════════════════════════════════

bvh_list = []

@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left:   Optional["BVHNode"] = field(default=None, repr=False)
    right:  Optional["BVHNode"] = field(default=None, repr=False)
    points: Optional[np.ndarray] = field(default=None, repr=False)


def build_bvh(pts: np.ndarray, leaf_size: int = 8) -> BVHNode:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    node   = BVHNode(aabb_min=lo, aabb_max=hi)
    if len(pts) <= leaf_size:
        node.points = pts
        return node
    axis   = int(np.argmax(hi - lo))
    median = np.median(pts[:, axis])
    lmask  = pts[:, axis] <= median
    lpts, rpts = pts[lmask], pts[~lmask]
    if len(lpts) == 0 or len(rpts) == 0:
        node.points = pts
        return node
    node.left  = build_bvh(lpts, leaf_size)
    node.right = build_bvh(rpts, leaf_size)
    return node


def draw_bvh_boxes(node):
    if node is None:
        return

    size = node.aabb_max - node.aabb_min
    center = (node.aabb_max + node.aabb_min) / 2

    create_box(size.tolist(), center.tolist())

    draw_bvh_boxes(node.left)
    draw_bvh_boxes(node.right)

bvh_obsid = []

def draw_leaf_boxes(node):
    if node is None:
        return

    if node.left is None and node.right is None:
        size = node.aabb_max - node.aabb_min
        center = (node.aabb_max + node.aabb_min) / 2

        bvh_obsid.append(create_box(size.tolist(), center.tolist()))

        bvh_list.append(node)
        return

    draw_leaf_boxes(node.left)
    draw_leaf_boxes(node.right)


def _pt_aabb_sqdist(pt, lo, hi) -> float:
    return float(np.sum((pt - np.clip(pt, lo, hi)) ** 2))


def _bvh_nearest_sq(pt: np.ndarray, node: BVHNode, best: list) -> None:
    if _pt_aabb_sqdist(pt, node.aabb_min, node.aabb_max) >= best[0][0]:
        return
    if node.points is not None:
        sq  = ((node.points - pt) ** 2).sum(axis=1)
        idx = int(np.argmin(sq))
        if sq[idx] < best[0][0]:
            best[0] = (float(sq[idx]), node.points[idx].copy())
        return
    l_sq = _pt_aabb_sqdist(pt, node.left.aabb_min,  node.left.aabb_max)
    r_sq = _pt_aabb_sqdist(pt, node.right.aabb_min, node.right.aabb_max)
    first, second = (node.left, node.right) if l_sq <= r_sq else (node.right, node.left)
    _bvh_nearest_sq(pt, first,  best)
    _bvh_nearest_sq(pt, second, best)


def sdf_single(pt: np.ndarray, bvh: BVHNode) -> float:
    """
    Signed distance to one obstacle BVH.
    Sign: root AABB membership (inside AABB → negative).
    """
    best   = [(np.inf, None)]
    _bvh_nearest_sq(pt, bvh, best)
    dist   = float(np.sqrt(best[0][0]))
    inside = np.all(pt >= bvh.aabb_min) and np.all(pt <= bvh.aabb_max)
    return -dist if inside else dist


def sdf_scene(pt: np.ndarray) -> float:
    """Min signed distance over all obstacle BVHs."""
    return min(sdf_single(pt, bvh) for bvh in bvh_list)


# ══════════════════════════════════════════════════════════════
# 2.  CLUSTERING
# ══════════════════════════════════════════════════════════════

CONFIG_FILE = "dbscan_config.json"

def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded config from {path}")
        return cfg
    print(f"[INFO] No config found — using defaults")
    return None

params      = load_config()
eps         = params["eps"]         if params else 0.082
min_samples = params["min_samples"] if params else 3


def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl, cnt in cluster_sizes.items()
                      if lbl != -1 and cnt < min_points_threshold]
    for small_lbl in small_clusters:
        small_mask   = (labels == small_lbl)
        small_points = points[small_mask]
        distances    = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1:
                continue
            centroid = points[labels == lbl].mean(axis=0)
            dist     = np.mean(np.linalg.norm(small_points - centroid, axis=1))
            distances.append((lbl, dist))
        if distances:
            closest_lbl = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest_lbl
            print(f"  Merged cluster {small_lbl} ({len(small_points)} pts) "
                  f"→ cluster {closest_lbl}")
    return labels


def find_clusters(points, e=eps, ms=min_samples):
    db     = DBSCAN(eps=e, min_samples=ms).fit(points)
    labels = db.labels_.copy()
    labels = group(points, labels)
    return labels


# ══════════════════════════════════════════════════════════════
# 3.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

physicsClient    = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId          = p.loadURDF("plane.urdf")

armId = p.loadURDF(
    "arm_5.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=(p.URDF_USE_INERTIA_FROM_FILE |
           p.URDF_USE_SELF_COLLISION     |
           p.URDF_USE_IMPLICIT_CYLINDER)
)
NUM_JOINTS = p.getNumJoints(armId)
print(f"[INFO] Arm loaded — {NUM_JOINTS} joints")


def create_box(half_extents=[1,1,1], position=[0,0,0],
               orientation=[0,0,0], mass=0, color=[1,0,0,1]):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_sphere(radius=0.5, position=[0,0,0],
                  orientation=[0,0,0], mass=0, color=[0,1,0,1]):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_cylinder(radius=0.5, height=1.0, position=[0,0,0],
                    orientation=[0,0,0], mass=0, color=[0,0,1,1]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))


# boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],      orientation=[0.2,1.1,0.4])
# sphereId   = create_sphere(radius=1,               position=[0,2,1])
# cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1],   orientation=[1.3,0,0])
# obstacle_ids = [boxId, cylinderId, sphereId]
# obstacle_names = ["box", "cylinder", "sphere"]

def load_mesh_obstacle(obj_path, position=[0,0,0],
                       orientation=[0,0,0], scale=1.0,
                       color=[0.8, 0.5, 0.2, 1]):
    col  = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    vis  = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        rgbaColor=color
    )
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(0, col, vis, position, quat)

# glassId   = load_mesh_obstacle("glass.obj",   position=[1, 0.3, 1], scale=0.1)
# bottleId = load_mesh_obstacle("Plastic-Bottle.obj", position=[-1,   0, 0], scale=0.1)
concaveId = load_mesh_obstacle("concave.obj", position=[0,   0.5, 0], scale=0.4)

obstacle_ids   = [concaveId]
obstacle_names = ["Concave"]

# ══════════════════════════════════════════════════════════════
# 4.  LOAD POINTS → CLUSTER → BUILD BVH
# ══════════════════════════════════════════════════════════════

points = np.load("points.npy")
print(f"[INFO] Loaded {len(points)} points")

labels        = find_clusters(points)
unique_labels = np.unique(labels[labels >= 0])
print(f"[INFO] {len(unique_labels)} clusters after merging")


cluster_pts = []

for lbl in unique_labels:
    cpts = points[labels == lbl]
    cluster_pts.append(cpts)
    bvh  = build_bvh(cpts, leaf_size=8)
    print(f"  Cluster {lbl}: {len(cpts)} pts | "
          f"AABB {bvh.aabb_min.round(3)} → {bvh.aabb_max.round(3)}")

    draw_leaf_boxes(bvh)


# Load convex hull bodies for GUI visualisation
# for cpts in cluster_pts:
#     if len(cpts) < 4:
#         continue
#     try:
#         hull   = ConvexHull(cpts)
#         verts  = cpts[hull.vertices]
#         col_id = p.createCollisionShape(p.GEOM_MESH,
#                                         vertices=verts.tolist(),
#                                         meshScale=[1, 1, 1])
#         p.createMultiBody(baseMass=0,
#                           baseCollisionShapeIndex=col_id,
#                           basePosition=[0, 0, 0])
#     except Exception as e:
#         print(f"  [WARN] Visual body failed: {e}")


# ══════════════════════════════════════════════════════════════
# 5.  HELPERS
# ══════════════════════════════════════════════════════════════

def get_link_positions(arm_id: int) -> dict:
    base_pos, _ = p.getBasePositionAndOrientation(arm_id)
    pos = {-1: np.array(base_pos)}
    for j in range(p.getNumJoints(arm_id)):
        state  = p.getLinkState(arm_id, j, computeForwardKinematics=True)
        pos[j] = np.array(state[4])
    return pos


def set_config(arm_id: int, config) -> None:
    for j, angle in enumerate(config):
        p.resetJointState(arm_id, j, float(angle))


def bvh_min_distance(arm_id: int) -> float:
    """Min SDF over all links using AABB-BVH."""
    threshold = 10
    min_d = threshold
    for obs_id in bvh_obsid:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=obs_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
    # return min(sdf_scene(pos) for pos in get_link_positions(arm_id).values())
    return min_d

def gt_min_distance(arm_id: int, obs_ids: list, threshold=10.0) -> float:
    """Ground truth via p.getClosestPoints."""
    min_d = threshold
    for obs_id in obs_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=obs_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
    return min_d


# ══════════════════════════════════════════════════════════════
# 6.  SAMPLE 10 000 RANDOM C-SPACE CONFIGS + COMPARE
# ══════════════════════════════════════════════════════════════

N_TOTAL  = 10000
INTERVAL = 1000

print(f"\n[INFO] Sampling {N_TOTAL} random configs…\n")

configs   = np.random.uniform(-np.pi, np.pi, size=(N_TOTAL, NUM_JOINTS))
bvh_dists = np.zeros(N_TOTAL)
gt_dists  = np.zeros(N_TOTAL)

t_bvh = 0.0
t_gt  = 0.0

interval_mae       = []
interval_rmse      = []
interval_max       = []
interval_bias      = []
interval_col_agree = []
interval_idx       = []

for i, cfg in enumerate(configs):
    set_config(armId, cfg)
    p.stepSimulation()

    t0 = time.perf_counter()
    bvh_dists[i] = bvh_min_distance(armId)
    t_bvh += time.perf_counter() - t0

    t0 = time.perf_counter()
    gt_dists[i]  = gt_min_distance(armId, obstacle_ids)
    t_gt += time.perf_counter() - t0

    if (i + 1) % INTERVAL == 0:
        sl    = slice(i + 1 - INTERVAL, i + 1)
        err   = np.abs(bvh_dists[sl] - gt_dists[sl])
        bias  = bvh_dists[sl] - gt_dists[sl]
        agree = float(np.mean((bvh_dists[sl] < 0) == (gt_dists[sl] < 0))) * 100

        interval_mae.append(float(np.mean(err)))
        interval_rmse.append(float(np.sqrt(np.mean(err**2))))
        interval_max.append(float(np.max(err)))
        interval_bias.append(float(np.mean(bias)))
        interval_col_agree.append(agree)
        interval_idx.append(i + 1)

        print(f"  [{i+1:5d}]  "
              f"MAE={interval_mae[-1]:.4f}m  "
              f"RMSE={interval_rmse[-1]:.4f}m  "
              f"MaxErr={interval_max[-1]:.4f}m  "
              f"Bias={interval_bias[-1]:+.4f}m  "
              f"Agree={agree:.1f}%")

all_err  = np.abs(bvh_dists - gt_dists)
all_bias = bvh_dists - gt_dists
bvh_col  = bvh_dists < 0
gt_col   = gt_dists  < 0
speedup  = t_gt / max(t_bvh, 1e-9)

print(f"\n{'═'*58}")
print(f"  GLOBAL ACCURACY  (N={N_TOTAL})  —  AABB-BVH method")
print(f"{'═'*58}")
print(f"  MAE                    : {np.mean(all_err):.5f} m")
print(f"  RMSE                   : {np.sqrt(np.mean(all_err**2)):.5f} m")
print(f"  Max absolute error     : {np.max(all_err):.5f} m")
print(f"  Mean bias (BVH - GT)   : {np.mean(all_bias):+.5f} m")
print(f"  Collision agreement    : {np.mean(bvh_col == gt_col)*100:.2f}%")
print(f"  False positives        : {np.sum(bvh_col & ~gt_col)}")
print(f"  False negatives        : {np.sum(~bvh_col & gt_col)}")
print(f"  BVH time / query       : {t_bvh/N_TOTAL*1000:.3f} ms")
print(f"  GT  time / query       : {t_gt/N_TOTAL*1000:.3f} ms")
print(f"  Speedup                : {speedup:.2f}x")
print(f"{'═'*58}")


# ══════════════════════════════════════════════════════════════
# 7.  PLOTS
# ══════════════════════════════════════════════════════════════

BLUE   = "#3A7DC9"
ORANGE = "#E8882A"
RED    = "#C93A3A"
GREEN  = "#2E9E5B"
PURPLE = "#7F3FBF"
LBLUE  = "#B5D4F4"
GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
})

xs = interval_idx


# ── Plot 1: MAE & RMSE ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, interval_mae,  "o-",  color=BLUE,   lw=2, ms=7, label="MAE")
ax.plot(xs, interval_rmse, "s--", color=ORANGE, lw=2, ms=7, label="RMSE")
ax.set_title("MAE and RMSE per 1 000-config interval\n(AABB-BVH method)")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Error  (m)")
ax.set_xticks(xs)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot1_mae_rmse.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot1_mae_rmse.png")


# ── Plot 2: Max error ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(xs, interval_max, width=700, color=RED, edgecolor="white", lw=0.5)
for x, v in zip(xs, interval_max):
    ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=RED)
ax.set_title("Max absolute error per 1 000-config interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("|BVH − GT|  max  (m)")
ax.set_xticks(xs)
plt.tight_layout()
plt.savefig("plot2_max_error.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot2_max_error.png")


# ── Plot 3: Bias ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = [GREEN if b >= 0 else RED for b in interval_bias]
ax.bar(xs, interval_bias, width=700, color=colors, edgecolor="white", lw=0.5)
ax.axhline(0, color="black", lw=0.8, zorder=3)
for x, v in zip(xs, interval_bias):
    va = "bottom" if v >= 0 else "top"
    ax.text(x, v + (0.001 if v >= 0 else -0.001),
            f"{v:+.4f}", ha="center", va=va, fontsize=9)
ax.set_title("Mean signed bias (BVH − GT) per interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Bias  (m)   [+ = BVH overestimates, safe]")
ax.set_xticks(xs)
ax.legend(handles=[Patch(color=GREEN, label="BVH overestimates (safe)"),
                   Patch(color=RED,   label="BVH underestimates (unsafe)")],
          frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig("plot3_bias.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot3_bias.png")


# ── Plot 4: Collision agreement ──────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(xs, interval_col_agree, "D-", color=PURPLE, lw=2, ms=8, zorder=3)
ax.fill_between(xs, interval_col_agree, alpha=0.12, color=PURPLE)
ax.axhline(100, color=GREEN, lw=1.2, ls="--", label="Perfect (100%)")
ax.set_ylim(0, 105)
ax.set_title("Collision detection agreement per interval")
ax.set_xlabel("Configs evaluated")
ax.set_ylabel("Agreement  (%)")
ax.set_xticks(xs)
for x, v in zip(xs, interval_col_agree):
    ax.text(x, v - 3.5, f"{v:.1f}%", ha="center", va="top", fontsize=9, color=PURPLE)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot4_collision_agree.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot4_collision_agree.png")


# ── Plot 5: Scatter ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lim = max(np.abs(bvh_dists).max(), np.abs(gt_dists).max()) * 1.05
ax.scatter(gt_dists, bvh_dists, s=1.5, alpha=0.12, color=BLUE, rasterized=True)
ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1.5, ls="--", label="Perfect  y = x")
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axvline(0, color="black", lw=0.5, alpha=0.4)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title("AABB-BVH SDF vs GT distance — all configs")
ax.set_xlabel("GT  p.getClosestPoints  (m)")
ax.set_ylabel("AABB-BVH SDF  (m)")
ax.legend(frameon=False)
ax.text( lim*0.55,  lim*0.82, "Both free",      fontsize=9, color=GRAY, ha="center")
ax.text(-lim*0.55, -lim*0.82, "Both collision",  fontsize=9, color=GRAY, ha="center")
ax.text(-lim*0.55,  lim*0.82, "False positive",  fontsize=9, color=RED,  ha="center")
ax.text( lim*0.55, -lim*0.82, "False negative",  fontsize=9, color=ORANGE, ha="center")
plt.tight_layout()
plt.savefig("plot5_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot5_scatter.png")


# ── Plot 6: Error histogram ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(all_err, bins=80, color=LBLUE, edgecolor=BLUE, linewidth=0.3)
ax.axvline(np.mean(all_err),   color=RED,    lw=2, ls="--",
           label=f"MAE    = {np.mean(all_err):.4f} m")
ax.axvline(np.median(all_err), color=ORANGE, lw=2, ls=":",
           label=f"Median = {np.median(all_err):.4f} m")
ax.set_title("Absolute error distribution — all configs")
ax.set_xlabel("|AABB-BVH SDF − GT|  (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot6_error_hist.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot6_error_hist.png")


# ── Plot 7: Timing ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
methods = ["AABB-BVH\n(this work)", "p.getClosestPoints\n(ground truth)"]
times   = [t_bvh/N_TOTAL*1000, t_gt/N_TOTAL*1000]
bars    = ax.bar(methods, times, color=[BLUE, RED],
                 edgecolor="white", lw=0.5, width=0.45)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2,
            t + 0.005, f"{t:.3f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title(f"Mean query time per config  (N = {N_TOTAL:,})")
ax.set_ylabel("Time  (ms / config)")
ax.text(0.98, 0.96,
        f"{'BVH' if t_bvh < t_gt else 'GT'} is "
        f"{max(speedup, 1/speedup if speedup > 0 else 1):.1f}× faster",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=BLUE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=0.8))
plt.tight_layout()
plt.savefig("plot7_timing.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot7_timing.png")


# ── Plot 8: Confusion matrix ─────────────────────────────────
fp = int(np.sum(bvh_col & ~gt_col))
fn = int(np.sum(~bvh_col & gt_col))
tp = int(np.sum(bvh_col & gt_col))
tn = int(np.sum(~bvh_col & ~gt_col))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

cm        = np.array([[tn, fp], [fn, tp]])
labels_cm = [["True Neg\n(both free)", "False Pos\n(BVH collision,\nGT free)"],
             ["False Neg\n(BVH free,\nGT collision)", "True Pos\n(both collision)"]]
colors_cm = [[GREEN, RED], [ORANGE, GREEN]]

ax = axes[0]
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["GT: Free", "GT: Collision"])
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["BVH: Collision", "BVH: Free"])
ax.set_title("Collision detection confusion matrix\n(AABB-BVH)")
ax.spines[:].set_visible(False)
ax.grid(False)
for r in range(2):
    for c in range(2):
        val = cm[r, c]
        ax.add_patch(plt.Rectangle((c, 1-r), 1, 1,
                                   fc=colors_cm[r][c], alpha=0.25, ec="white", lw=2))
        ax.text(c + 0.5, 1 - r + 0.5,
                f"{labels_cm[r][c]}\n{val:,}\n({val/N_TOTAL*100:.1f}%)",
                ha="center", va="center", fontsize=9)

ax2 = axes[1]
cats   = ["True\nNeg", "True\nPos", "False\nPos", "False\nNeg"]
counts = [tn, tp, fp, fn]
cols   = [GREEN, GREEN, RED, ORANGE]
bars2  = ax2.bar(cats, counts, color=cols, edgecolor="white", lw=0.5, width=0.5)
for bar, v in zip(bars2, counts):
    ax2.text(bar.get_x() + bar.get_width()/2,
             v + N_TOTAL*0.005, f"{v:,}",
             ha="center", va="bottom", fontsize=10, fontweight="bold")
ax2.set_title("Classification counts")
ax2.set_ylabel("Count")

plt.tight_layout()
plt.savefig("plot8_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot8_confusion.png")


p.disconnect()
print("\n[INFO] All done. 8 plots saved.")