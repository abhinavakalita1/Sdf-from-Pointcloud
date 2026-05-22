import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass, field
from typing import Optional
import time


# ══════════════════════════════════════════════════════════════
# 1.  AABB-BVH
# ══════════════════════════════════════════════════════════════

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
    best   = [(np.inf, None)]
    _bvh_nearest_sq(pt, bvh, best)
    dist   = float(np.sqrt(best[0][0]))
    inside = np.all(pt >= bvh.aabb_min) and np.all(pt <= bvh.aabb_max)
    return -dist if inside else dist


def sdf_scene(pt: np.ndarray, bvh_list: list) -> float:
    return min(sdf_single(pt, bvh) for bvh in bvh_list)


# ══════════════════════════════════════════════════════════════
# 2.  CLUSTERING
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
# 3.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

physicsClient    = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId          = p.loadURDF("plane.urdf")
startPos         = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])

armId = p.loadURDF(
    "arm_3.urdf",               # ← swap to your 3-DOF URDF name
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

# Joint limits — all -π to +π (change per joint if needed)
JOINT_LIMITS = [(-np.pi, np.pi)] * NUM_JOINTS


# ══════════════════════════════════════════════════════════════
# 4.  POINT CLOUD  →  CLUSTER  →  BVH
# ══════════════════════════════════════════════════════════════

points = np.load("points.npy")
print(f"[INFO] Loaded {len(points)} points")

labels        = find_clusters(points)
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

# Load convex-hull bodies for GUI visualisation
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
        print(f"  [WARN] Visual body failed: {e}")


# ══════════════════════════════════════════════════════════════
# 5.  HELPERS
# ══════════════════════════════════════════════════════════════

def get_link_positions(arm_id: int) -> dict:
    """World-frame XYZ for base (-1) and every joint link."""
    base_pos, _ = p.getBasePositionAndOrientation(arm_id)
    pos = {-1: np.array(base_pos)}
    for j in range(p.getNumJoints(arm_id)):
        state  = p.getLinkState(arm_id, j, computeForwardKinematics=True)
        pos[j] = np.array(state[4])
    return pos


def set_joint_config(arm_id: int, config: list) -> None:
    """Instantly set joint positions (no dynamics)."""
    for j, angle in enumerate(config):
        p.resetJointState(arm_id, j, angle)


def arm_min_sdf(config: list) -> float:
    """
    Forward-kinematics → get all link positions → query BVH SDF
    → return the MINIMUM sdf across all links (worst case proximity).
    Negative = collision.
    """
    set_joint_config(armId, config)
    p.stepSimulation()                          # update FK
    link_positions = get_link_positions(armId)
    return min(sdf_scene(pos, bvh_list) for pos in link_positions.values())


# ══════════════════════════════════════════════════════════════
# 6.  SIMULATION LOOP  (your original 10 000 steps)
# ══════════════════════════════════════════════════════════════

print("\n[INFO] Starting simulation loop…\n")

sdf_log = {lk: [] for lk in range(-1, NUM_JOINTS)}

try:
    for i in range(10000):
        p.resetJointState(armId, 2, i / 100)
        p.stepSimulation()

        link_positions = get_link_positions(armId)
        sdf_values     = {}

        for lk, pos in link_positions.items():
            sd = sdf_scene(pos, bvh_list)
            sdf_values[lk] = sd
            sdf_log[lk].append(sd)

        if i % 100 == 0:
            min_sd   = min(sdf_values.values())
            min_link = min(sdf_values, key=sdf_values.get)
            status   = ("⚠ COLLISION" if min_sd < 0
                        else ("⚡ NEAR"   if min_sd < 0.05
                        else  "✓ SAFE"))
            print(f"Step {i:5d}  |  min SDF = {min_sd:+.4f} m  "
                  f"(link {min_link})  {status}")

        time.sleep(1. / 240.)

except KeyboardInterrupt:
    print("\n[INFO] Sim interrupted")


# ══════════════════════════════════════════════════════════════
# 7.  C-SPACE SAMPLING  —  random configs, SDF per config
# ══════════════════════════════════════════════════════════════

N_SAMPLES = 5000   # ↑ for denser plot, ↓ for speed
print(f"\n[INFO] Sampling {N_SAMPLES} random configs in C-space…")

# Only use first 3 joints for C-space axes (θ1, θ2, θ3)
CSPACE_JOINTS = min(3, NUM_JOINTS)

configs  = np.random.uniform(-np.pi, np.pi, size=(N_SAMPLES, CSPACE_JOINTS))
sdf_vals = np.zeros(N_SAMPLES)

for i, cfg in enumerate(configs):
    # Pad to NUM_JOINTS if arm has more than 3
    full_cfg = list(cfg) + [0.0] * (NUM_JOINTS - CSPACE_JOINTS)
    sdf_vals[i] = arm_min_sdf(full_cfg)

    if i % 500 == 0:
        print(f"  Sampled {i}/{N_SAMPLES}  "
              f"min_sdf so far = {sdf_vals[:i+1].min():+.4f}")

print(f"[INFO] C-space sampling done.")
print(f"  Collision configs (SDF < 0) : {(sdf_vals < 0).sum()}")
print(f"  Near configs  (0–0.05 m)    : {((sdf_vals >= 0) & (sdf_vals < 0.05)).sum()}")
print(f"  Free configs  (SDF >= 0.05) : {(sdf_vals >= 0.05).sum()}")


# ══════════════════════════════════════════════════════════════
# 8.  C-SPACE PLOT
#
#   Three regions, clearly separated:
#     ■ RED (opaque)         collision  (SDF < 0)
#     ■ GREY (semi-trans)    boundary   (0 ≤ SDF < 0.05 m)
#     ■ GRADIENT blue→green  free space (SDF ≥ 0.05 m)
# ══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(12, 9))
ax  = fig.add_subplot(111, projection='3d')
ax.set_title("C-Space SDF  (θ₁, θ₂, θ₃)", fontsize=14, pad=15)

theta1, theta2, theta3 = configs[:, 0], configs[:, 1], configs[:, 2]

# ── Masks ─────────────────────────────────────────────────────
mask_collision = sdf_vals < 0
mask_boundary  = (sdf_vals >= 0)     & (sdf_vals < 0.05)
mask_free      = sdf_vals >= 0.05


cspace = np.vstack((theta1[mask_collision],theta2[mask_collision],theta3[mask_collision]))
np.save("collision_points.npy", cspace)


# ── 1. Free space — colour-mapped by SDF value ────────────────
free_sdf = sdf_vals[mask_free]
norm     = mcolors.Normalize(vmin=free_sdf.min(), vmax=free_sdf.max())
cmap_free = plt.cm.get_cmap("cool")           # blue (low) → magenta (high)
free_colors = cmap_free(norm(free_sdf))

sc_free = ax.scatter(
    theta1[mask_free], theta2[mask_free], theta3[mask_free],
    c=free_sdf, cmap="cool",
    s=8, alpha=0.55, label=f"Free ({mask_free.sum()})",
    vmin=free_sdf.min(), vmax=free_sdf.max()
)
cbar = fig.colorbar(sc_free, ax=ax, shrink=0.55, pad=0.1)
cbar.set_label("SDF — min link distance to obstacle (m)", fontsize=9)

# ── 2. Boundary — grey, semi-transparent ──────────────────────
if mask_boundary.sum() > 0:
    ax.scatter(
        theta1[mask_boundary], theta2[mask_boundary], theta3[mask_boundary],
        c="silver", s=18, alpha=0.7,
        edgecolors="grey", linewidths=0.3,
        label=f"Boundary 0–5cm ({mask_boundary.sum()})"
    )

# ── 3. Collision — solid red, largest markers ─────────────────
if mask_collision.sum() > 0:
    ax.scatter(
        theta1[mask_collision], theta2[mask_collision], theta3[mask_collision],
        c="red", s=30, alpha=0.95,
        edgecolors="darkred", linewidths=0.4,
        label=f"Collision SDF<0 ({mask_collision.sum()})"
    )

# ── Axes ──────────────────────────────────────────────────────
ax.set_xlabel("θ₁  (rad)", fontsize=11, labelpad=8)
ax.set_ylabel("θ₂  (rad)", fontsize=11, labelpad=8)
ax.set_zlabel("θ₃  (rad)", fontsize=11, labelpad=8)
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-np.pi, np.pi)
ax.set_zlim(-np.pi, np.pi)

ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = ["-π", "-π/2", "0", "π/2", "π"]
ax.set_xticks(ticks); ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks); ax.set_yticklabels(tick_labels)
ax.set_zticks(ticks); ax.set_zticklabels(tick_labels)

ax.legend(loc="upper left", fontsize=9, framealpha=0.8)
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("cspace_sdf.png", dpi=150)
plt.show()
print("[INFO] C-space plot saved → cspace_sdf.png")


# ══════════════════════════════════════════════════════════════
# 9.  SDF HISTORY PLOT  (post-sim)
# ══════════════════════════════════════════════════════════════

fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
fig2.suptitle("BVH-SDF history — all arm links", fontsize=13)
cmap2 = plt.cm.get_cmap("tab10", NUM_JOINTS + 1)

ax_line = axes[0]
for idx, lk in enumerate(range(-1, NUM_JOINTS)):
    lbl = "base" if lk == -1 else f"link {lk}"
    ax_line.plot(sdf_log[lk], label=lbl, color=cmap2(idx), lw=1)
ax_line.axhline(0, color="red", lw=1, ls="--", label="surface (SDF=0)")
ax_line.set_ylabel("SDF (m)")
ax_line.set_xlabel("sim step")
ax_line.legend(fontsize=7, ncol=4)
ax_line.grid(alpha=0.3)

ax_bar = axes[1]
final_sdfs = [sdf_log[lk][-1] if sdf_log[lk] else 0.0
              for lk in range(-1, NUM_JOINTS)]
link_names = ["base"] + [f"link {j}" for j in range(NUM_JOINTS)]
bar_colors = ["red"      if v < 0
              else "orange"   if v < 0.05
              else "steelblue"
              for v in final_sdfs]
ax_bar.bar(link_names, final_sdfs, color=bar_colors, edgecolor="black", lw=0.5)
ax_bar.axhline(0, color="red", lw=1, ls="--")
ax_bar.set_ylabel("Final SDF (m)")
ax_bar.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("sdf_history.png", dpi=150)
plt.show()
print("[INFO] SDF history plot saved → sdf_history.png")

print(time.time()-t)
p.disconnect()


# ══════════════════════════════════════════════════════════════
# 10.  PUBLIC SDF FUNCTION  (callable after script finishes)
# ══════════════════════════════════════════════════════════════

def sdf(query_point, idx: int = None) -> float:
    """
    Signed distance from a world-frame point to the obstacle scene.

    Parameters
    ----------
    query_point : array-like (3,)   world-frame XYZ
    idx         : int or None       specific cluster, or None = all

    Returns
    -------
    float   positive → free, zero → surface, negative → collision
    """
    pt = np.asarray(query_point, dtype=float)
    if idx is not None:
        return sdf_single(pt, bvh_list[idx])
    return sdf_scene(pt, bvh_list)