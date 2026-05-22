import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# 1.  AABB-BVH
# ══════════════════════════════════════════════════════════════

@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left:   Optional["BVHNode"] = field(default=None, repr=False)
    right:  Optional["BVHNode"] = field(default=None, repr=False)
    points: Optional[np.ndarray] = field(default=None, repr=False)   # leaf only


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


def _pt_aabb_sqdist(pt, lo, hi) -> float:
    return float(np.sum((pt - np.clip(pt, lo, hi)) ** 2))


def _bvh_nearest_sq(pt: np.ndarray, node: BVHNode, best: list) -> None:
    if _pt_aabb_sqdist(pt, node.aabb_min, node.aabb_max) >= best[0][0]:
        return                                      # prune

    if node.points is not None:                     # leaf
        sq = ((node.points - pt) ** 2).sum(axis=1)
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
    Signed distance from pt to one obstacle BVH.
      d > 0  →  outside (safe)
      d = 0  →  on surface
      d < 0  →  inside  (collision)
    Sign approximated from root AABB membership.
    """
    best = [(np.inf, None)]
    _bvh_nearest_sq(pt, bvh, best)
    dist = float(np.sqrt(best[0][0]))
    inside = np.all(pt >= bvh.aabb_min) and np.all(pt <= bvh.aabb_max)
    return -dist if inside else dist


def sdf_scene(pt: np.ndarray, bvh_list: list) -> float:
    """Min signed distance over all obstacles (most negative wins)."""
    return min(sdf_single(pt, bvh) for bvh in bvh_list)


# ══════════════════════════════════════════════════════════════
# 2.  YOUR ORIGINAL CLUSTERING CODE  (unchanged)
# ══════════════════════════════════════════════════════════════

def plot_point_cloud(points, squish=0):
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1)

    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]
    z_limits = [points[:, 2].min(), points[:, 2].max()]

    max_range = np.ptp(np.array([x_limits, y_limits, z_limits])).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range + squish, mid_z + max_range - squish)
    plt.show()


def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl, cnt in cluster_sizes.items()
                      if lbl != -1 and cnt < min_points_threshold]

    for small_lbl in small_clusters:
        small_mask   = (labels == small_lbl)
        small_points = points[small_mask]

        distances = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1:
                continue
            centroid = points[labels == lbl].mean(axis=0)
            dist     = np.mean(np.linalg.norm(small_points - centroid, axis=1))
            distances.append((lbl, dist))

        if distances:
            closest_lbl = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest_lbl
            print(f"Merged cluster {small_lbl} ({len(small_points)} pts) "
                  f"→ into cluster {closest_lbl}")

    return labels


def find_clusters(points, eps=0.082, min_samples=3):
    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_.copy()
    labels = group(points, labels)
    return labels


# ══════════════════════════════════════════════════════════════
# 3.  PYBULLET SETUP  (your original code)
# ══════════════════════════════════════════════════════════════

physicsClient   = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId         = p.loadURDF("plane.urdf")
startPos        = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

armId = p.loadURDF(
    "arm_5.urdf",
    startPos,
    startOrientation,
    useFixedBase=True,
    flags=(
        p.URDF_USE_INERTIA_FROM_FILE |
        p.URDF_USE_SELF_COLLISION     |
        p.URDF_USE_IMPLICIT_CYLINDER
    )
)

NUM_JOINTS = p.getNumJoints(armId)
print(f"[INFO] Arm loaded — {NUM_JOINTS} joints")


# ══════════════════════════════════════════════════════════════
# 4.  LOAD POINTS  →  CLUSTER  →  BUILD BVH PER CLUSTER
# ══════════════════════════════════════════════════════════════

points = np.load("points.npy")
print(f"[INFO] Loaded {len(points)} points")

labels       = find_clusters(points)
unique_labels = np.unique(labels[labels >= 0])
print(f"[INFO] {len(unique_labels)} clusters after merging")

bvh_list    = []
cluster_pts = []

for lbl in unique_labels:
    cpts = points[labels == lbl]
    cluster_pts.append(cpts)
    bvh  = build_bvh(cpts, leaf_size=8)
    bvh_list.append(bvh)
    print(f"  Cluster {lbl}: {len(cpts)} pts | "
          f"AABB {bvh.aabb_min.round(3)} → {bvh.aabb_max.round(3)}")

# Optional: load convex-hull bodies so you can see the obstacles in the GUI
for cpts in cluster_pts:
    if len(cpts) < 4:
        continue
    try:
        hull   = ConvexHull(cpts)
        verts  = cpts[hull.vertices]
        col_id = p.createCollisionShape(p.GEOM_MESH,
                                        vertices=verts.tolist(),
                                        meshScale=[1, 1, 1])
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=col_id,
                          basePosition=[0, 0, 0])
    except Exception as e:
        print(f"  [WARN] Could not create visual body: {e}")


# ══════════════════════════════════════════════════════════════
# 5.  HELPER — link positions from forward kinematics
# ══════════════════════════════════════════════════════════════

def get_link_positions(arm_id: int) -> dict:
    """Returns {link_idx: np.array([x,y,z])}.  -1 = base."""
    base_pos, _ = p.getBasePositionAndOrientation(arm_id)
    pos = {-1: np.array(base_pos)}
    for j in range(p.getNumJoints(arm_id)):
        state  = p.getLinkState(arm_id, j, computeForwardKinematics=True)
        pos[j] = np.array(state[4])
    return pos


# ══════════════════════════════════════════════════════════════
# 6.  SIMULATION LOOP — compute SDF every step
# ══════════════════════════════════════════════════════════════

print("\n[INFO] Starting simulation loop…  (Ctrl-C to stop)\n")

sdf_log = {lk: [] for lk in range(-1, NUM_JOINTS)}   # history per link

try:
    for i in range(10000):
        p.resetJointState(armId, 2, i/100)
        p.stepSimulation()

        # ── Query SDF for every arm link ──────────────────────
        link_positions = get_link_positions(armId)
        sdf_values     = {}

        for lk, pos in link_positions.items():
            sd = sdf_scene(pos, bvh_list)
            sdf_values[lk] = sd
            sdf_log[lk].append(sd)

        # ── Console print every 100 steps ─────────────────────
        if i % 100 == 0:
            min_sd   = min(sdf_values.values())
            min_link = min(sdf_values, key=sdf_values.get)
            status   = ("⚠ COLLISION" if min_sd < 0
                        else ("⚡ NEAR"   if min_sd < 0.05
                        else  "✓ SAFE"))
            print(f"Step {i:5d}  |  min SDF = {min_sd:+.4f} m  "
                  f"(link {min_link:2d})  {status}")
            for lk, sd in sdf_values.items():
                print(f"           link {lk:2d}  SDF = {sd:+.4f} m")

        time.sleep(1. / 240.)

except KeyboardInterrupt:
    print("\n[INFO] Interrupted")


# ══════════════════════════════════════════════════════════════
# 7.  POST-SIM: PLOT SDF HISTORY + EXPOSE sdf() FUNCTION
# ══════════════════════════════════════════════════════════════

# ── Plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("BVH-SDF history — all arm links", fontsize=13)
cmap = plt.cm.get_cmap("tab10", NUM_JOINTS + 1)

ax = axes[0]
for idx, lk in enumerate(range(-1, NUM_JOINTS)):
    lbl = "base" if lk == -1 else f"link {lk}"
    ax.plot(sdf_log[lk], label=lbl, color=cmap(idx), lw=1)
ax.axhline(0, color="red", lw=1, ls="--", label="surface")
ax.set_ylabel("SDF (m)")
ax.set_xlabel("sim step")
ax.legend(fontsize=7, ncol=4)
ax.grid(alpha=0.3)

ax2 = axes[1]
final_sdfs = [sdf_log[lk][-1] if sdf_log[lk] else 0.0
              for lk in range(-1, NUM_JOINTS)]
link_names = ["base"] + [f"link {j}" for j in range(NUM_JOINTS)]
bar_colors = ["red" if v < 0 else ("orange" if v < 0.05 else "steelblue")
              for v in final_sdfs]
ax2.bar(link_names, final_sdfs, color=bar_colors, edgecolor="black", lw=0.5)
ax2.axhline(0, color="red", lw=1, ls="--")
ax2.set_ylabel("Final SDF (m)")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("sdf_history.png", dpi=150)
plt.show()
print("[INFO] Plot saved → sdf_history.png")

p.disconnect()


# ══════════════════════════════════════════════════════════════
# 8.  CALLABLE SDF  — use this anywhere after the sim
#
#   sdf(query_point)          → scalar (min over all obstacles)
#   sdf(query_point, idx=0)   → scalar (obstacle 0 only)
# ══════════════════════════════════════════════════════════════

def sdf(query_point, idx: int = None) -> float:
    """
    Public SDF function.

    Parameters
    ----------
    query_point : array-like, shape (3,)
        World-frame point to query.
    idx : int or None
        If given, query only obstacle cluster `idx`.
        If None (default), return min SDF over all obstacles.

    Returns
    -------
    float
        Signed distance in metres.
        Positive → outside (safe), negative → inside (collision).

    Examples
    --------
    >>> sdf([0.1, 0.2, 0.3])          # scene min
    0.0423
    >>> sdf([0.1, 0.2, 0.3], idx=1)   # obstacle 1 only
    0.1102
    """
    pt = np.asarray(query_point, dtype=float)
    if idx is not None:
        return sdf_single(pt, bvh_list[idx])
    return sdf_scene(pt, bvh_list)