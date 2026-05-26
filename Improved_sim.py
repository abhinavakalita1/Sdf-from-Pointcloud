import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree, Delaunay, ConvexHull
import pybullet as p
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ══════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

armId = p.loadURDF(
    "arm_5.urdf",
    [0, 0, 0],
    p.getQuaternionFromEuler([0, 0, 0]),
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
# 2.  PRIMITIVE OBSTACLES
# ══════════════════════════════════════════════════════════════

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

boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],      orientation=[0.2,1.1,0.4])
sphereId   = create_sphere(radius=1,               position=[0,2,1])
cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1],   orientation=[1.3,0,0])
obstacle_ids   = [boxId, sphereId, cylinderId]
obstacle_names = ["Box", "Sphere", "Cylinder"]
print(f"[INFO] Obstacles: box={boxId}, sphere={sphereId}, cylinder={cylinderId}")


# ══════════════════════════════════════════════════════════════
# 3.  DENSE POINTCLOUD GENERATION
#     Three ray passes for full surface coverage:
#       Pass A — vertical (top-down grid)
#       Pass B — horizontal ring (36 angles × 20 heights)
#       Pass C — diagonal (4 elevation angles × 36 azimuths)
# ══════════════════════════════════════════════════════════════

def sample_pointcloud(body_ids, n_vertical=30000, regen=False):
    if not regen:
        try:
            pts = np.load("points.npy")
            print(f"[INFO] Loaded existing points.npy  ({len(pts)} pts)")
            return pts
        except FileNotFoundError:
            pass

    print("[INFO] Generating dense pointcloud (3 ray passes)…")
    pts = []

    # ── Pass A: vertical top-down ─────────────────────────────
    spread = 6.0
    for _ in range(n_vertical):
        dx, dy = np.random.uniform(-spread, spread, 2)
        r = p.rayTest([dx, dy, 6.0], [dx, dy, -1.0])
        if r[0] in body_ids:
            pts.append(r[3])

    # ── Pass B: horizontal ring (catches vertical faces) ──────
    for angle in np.linspace(0, 2*np.pi, 72, endpoint=False):   # 72 azimuths
        for height in np.linspace(0.0, 3.0, 30):                 # 30 heights
            ox, oy = 8*np.cos(angle), 8*np.sin(angle)
            r = p.rayTest([ox, oy, height], [-ox, -oy, height])
            if r[0] in body_ids:
                pts.append(r[3])

    # ── Pass C: diagonal (catches angled faces of rotated box) ─
    for elev in [20, 40, 60, 80]:                                 # degrees
        elev_r = np.radians(elev)
        for az in np.linspace(0, 2*np.pi, 36, endpoint=False):
            dist = 7.0
            fx   = dist * np.cos(elev_r) * np.cos(az)
            fy   = dist * np.cos(elev_r) * np.sin(az)
            fz   = dist * np.sin(elev_r)
            r    = p.rayTest([fx, fy, fz], [-fx, -fy, -fz])
            if r[0] in body_ids:
                pts.append(r[3])

    pts = np.array(pts)
    np.save("points.npy", pts)
    print(f"[INFO] Generated {len(pts)} surface points → saved points.npy")
    return pts


# Set regen=True to force new pointcloud, False to reuse existing
points = sample_pointcloud(obstacle_ids, n_vertical=30000, regen=False)


# ══════════════════════════════════════════════════════════════
# 4.  CLUSTERING
# ══════════════════════════════════════════════════════════════

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
            mask = labels == lbl
            if not mask.any():
                continue
            centroid = points[mask].mean(axis=0)
            distances.append((lbl, np.mean(np.linalg.norm(small_points - centroid, axis=1))))
        if distances:
            closest = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest
            print(f"  Merged cluster {small_lbl} ({small_mask.sum()} pts) → cluster {closest}")
    return labels


def find_clusters(points, eps=0.082, min_samples=3):
    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_.copy()
    labels = group(points, labels)
    return labels


labels        = find_clusters(points)
unique_labels = np.unique(labels[labels >= 0])
print(f"[INFO] {len(unique_labels)} clusters found")

cluster_pts = []
for lbl in unique_labels:
    cpts = points[labels == lbl]
    cluster_pts.append(cpts)
    print(f"  Cluster {lbl}: {len(cpts)} pts  "
          f"| centre {cpts.mean(axis=0).round(3)}")


# ══════════════════════════════════════════════════════════════
# 5.  BUILD cKDTREE + QHULL PER CLUSTER
# ══════════════════════════════════════════════════════════════

t = time.time()
from scipy.spatial import cKDTree, ConvexHull

kdtrees   = []
qhulls    = []   # stores (equations,) from ConvexHull

for i, cpts in enumerate(cluster_pts):
    kdtrees.append(cKDTree(cpts))

    try:
        hull = ConvexHull(cpts)
        # hull.equations: each row is [normal_x, normal_y, normal_z, offset]
        # A point p is INSIDE the hull if:
        #   hull.equations[:, :3] @ p + hull.equations[:, 3] <= 0  for ALL rows
        qhulls.append(hull.equations)
        print(f"  Cluster {i}: KDTree + QHull built  "
              f"({len(hull.equations)} facets)")
    except Exception as e:
        qhulls.append(None)
        print(f"  Cluster {i}: KDTree built, QHull failed ({e}) — outside assumed")

print(time.time()-t)

# ══════════════════════════════════════════════════════════════
# 6.  SDF FUNCTION  (QHull sign)
# ══════════════════════════════════════════════════════════════

def _inside_qhull(pt: np.ndarray, equations: np.ndarray) -> bool:
    """
    Returns True if pt is inside the convex hull defined by equations.

    Each row of equations is [nx, ny, nz, d] where the half-space is:
        nx*x + ny*y + nz*z + d <= 0  (pointing inward)

    A point is inside if it satisfies ALL half-space inequalities.
    We add a small tolerance (1e-10) to handle numerical boundary cases.
    """
    return bool(np.all(equations[:, :3] @ pt + equations[:, 3] <= 1e-10))


def sdf_scene(pt) -> float:
    """
    Signed distance using:
      cKDTree   → distance magnitude  (fast compiled C)
      QHull     → sign via half-space equations  (direct Qhull output)
    """
    pt       = np.asarray(pt, dtype=float)
    best_d   = np.inf
    best_sgn = +1.0

    for kdt, equations in zip(kdtrees, qhulls):
        d = float(kdt.query(pt, k=1)[0])
        if d < best_d:
            best_d = d
            if equations is not None:
                best_sgn = -1.0 if _inside_qhull(pt, equations) else +1.0
            else:
                best_sgn = +1.0

    return best_sgn * best_d

# ══════════════════════════════════════════════════════════════
# 7.  HELPERS
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
    """Min SDF over all arm links — worst-case proximity."""
    return min(sdf_scene(pos) for pos in get_link_positions(arm_id).values())


def gt_min_distance(arm_id: int, obs_ids: list, threshold=10.0) -> float:
    """Ground truth via p.getClosestPoints — signed (negative = penetrating)."""
    min_d = threshold
    for obs_id in obs_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=obs_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d:
                min_d = d
    return min_d


# ══════════════════════════════════════════════════════════════
# 8.  SAMPLE 10 000 RANDOM C-SPACE CONFIGS + COMPARE
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

print(f"\n{'═'*58}")
print(f"  GLOBAL ACCURACY  (N={N_TOTAL})")
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
speedup = t_gt / max(t_bvh, 1e-9)
print(f"  Speedup                : {speedup:.2f}x")
print(f"{'═'*58}")


# ══════════════════════════════════════════════════════════════
# 9.  CLEAN SEPARATE PLOTS
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
ax.set_title("MAE and RMSE per 1 000-config interval")
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


# ── Plot 5: BVH vs GT scatter ────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lim = max(np.abs(bvh_dists).max(), np.abs(gt_dists).max()) * 1.05
ax.scatter(gt_dists, bvh_dists, s=1.5, alpha=0.12, color=BLUE, rasterized=True)
ax.plot([-lim, lim], [-lim, lim], color=RED, lw=1.5, ls="--", label="Perfect  y = x")
ax.axhline(0, color="black", lw=0.5, alpha=0.4)
ax.axvline(0, color="black", lw=0.5, alpha=0.4)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_aspect("equal")
ax.set_title("BVH SDF vs GT distance — all configs")
ax.set_xlabel("GT  p.getClosestPoints  (m)")
ax.set_ylabel("BVH SDF  (m)")
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
ax.set_xlabel("|BVH SDF − GT|  (m)")
ax.set_ylabel("Count")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig("plot6_error_hist.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot6_error_hist.png")


# ── Plot 7: Timing ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
methods = ["cKDTree SDF\n(this work)", "p.getClosestPoints\n(ground truth)"]
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
        f"{'KDTree' if t_bvh < t_gt else 'GT'} is "
        f"{max(speedup, 1/speedup if speedup>0 else 1):.1f}× faster",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=BLUE,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=BLUE, lw=0.8))
plt.tight_layout()
plt.savefig("plot7_timing.png", dpi=150, bbox_inches="tight")
plt.show()
print("[INFO] Saved plot7_timing.png")


# ── Plot 8: False positive / negative summary ────────────────
fp = int(np.sum(bvh_col & ~gt_col))
fn = int(np.sum(~bvh_col & gt_col))
tp = int(np.sum(bvh_col & gt_col))
tn = int(np.sum(~bvh_col & ~gt_col))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

# Left: confusion matrix style
cm = np.array([[tn, fp], [fn, tp]])
labels_cm = [["True Neg\n(both free)", "False Pos\n(BVH collision,\nGT free)"],
             ["False Neg\n(BVH free,\nGT collision)", "True Pos\n(both collision)"]]
colors_cm = [[GREEN, RED], [ORANGE, GREEN]]
ax = axes[0]
ax.set_xlim(0, 2); ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["GT: Free", "GT: Collision"])
ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["BVH: Collision", "BVH: Free"])
ax.set_title("Collision detection confusion matrix")
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

# Right: bar of FP/FN/TP/TN counts
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
print(f"  → plot1_mae_rmse.png")
print(f"  → plot2_max_error.png")
print(f"  → plot3_bias.png")
print(f"  → plot4_collision_agree.png")
print(f"  → plot5_scatter.png")
print(f"  → plot6_error_hist.png")
print(f"  → plot7_timing.png")
print(f"  → plot8_confusion.png")