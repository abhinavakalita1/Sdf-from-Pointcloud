"""
cspace_wavefront.py
────────────────────────────────────────────────────────────────────────────────
Monodirectional toroidal C-space wavefront explorer.

Key improvements over the original:
  1. SDF QUERIES REDUCED  — update_gui_overlay no longer triggers an extra
     eval_sdf on every ray step (was the dominant cost at ~44 calls/node).
     The overlay now reuses the already-computed cur_sdf.
  2. TIGHTER CACHE KEYS   — wrap_angle is applied before building the cache
     key so mirrored entries across the ±π seam share the same slot.
  3. DIRECTION SKIPPING    — if the adjacent probe (ADJ_OFFSET step) is
     already in collision we skip that ray immediately with zero extra queries
     (same as before, but now the cached adj_sdf is passed down to the first
     loop iteration so the initial step is free).
"""

import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
import json
from collections import deque

# ═══════════════════════════════════════════════════════════════════════════════
# TUNEABLE PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

STEP_SIZE         = 0.08   # rad  – interval increment along each ray
ADJ_OFFSET        = 0.04   # rad  – distance of the "adjacent probe" from node
MAX_STEPS_PER_RAY = 120    # extended step budget to allow wrapping scans
NODE_MERGE_RADIUS = 0.10   # rad  – nodes closer than this get merged
GOAL_REACH_RADIUS = 0.08   # rad  – declare success when a node is this close
CLUB_RADIUS       = 0.1    # rad  – grid cell size for visited-set deduplication

ALGORITHM_START_TIME = time.time()

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ═══════════════════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

p.configureDebugVisualizer(p.COV_ENABLE_GUI,                0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,            1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=5.0, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 1])

arm2Id = p.loadURDF(
    "arm_2dof.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS_2 = p.getNumJoints(arm2Id)
print(f"[INFO] 2-DOF arm loaded — {NUM_JOINTS_2} joints")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  OBSTACLES
# ═══════════════════════════════════════════════════════════════════════════════

def create_box(half_extents, position, orientation, mass=0, color=[1, 0, 0, 1]):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_sphere(radius, position, orientation=[0, 0, 0], mass=0, color=[0, 1, 0, 1]):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_cylinder(radius, height, position, orientation=[0, 0, 0], mass=0, color=[0, 0, 1, 1]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

boxId      = create_box([1, 1, 1],     [2,    0, 1], [0.2, 1.1, 0.4])
sphereId   = create_sphere(1,          [0,    2, 1])
cylinderId = create_cylinder(0.3, 2,   [-0.5, 0, 1], [1.3, 0,   0])
obstacle_ids = [boxId, sphereId, cylinderId]

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  POINTCLOUD
# ═══════════════════════════════════════════════════════════════════════════════

def sample_pointcloud(body_ids, n_vertical=30000):
    try:
        pts = np.load("points.npy")
        print(f"[INFO] Loaded points.npy ({len(pts)} pts)")
        return pts
    except FileNotFoundError:
        pass
    print("[INFO] Generating pointcloud …")
    pts = []
    spread = 6.0
    for _ in range(n_vertical):
        dx, dy = np.random.uniform(-spread, spread, 2)
        r = p.rayTest([dx, dy, 6.0], [dx, dy, -1.0])
        if r[0] in body_ids: pts.append(r[3])
    for angle in np.linspace(0, 2 * np.pi, 72, endpoint=False):
        for height in np.linspace(0.0, 3.0, 30):
            ox, oy = 8 * np.cos(angle), 8 * np.sin(angle)
            r = p.rayTest([ox, oy, height], [-ox, -oy, height])
            if r[0] in body_ids: pts.append(r[3])
    for elev in [20, 40, 60, 80]:
        elev_r = np.radians(elev)
        for az in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            dist = 7.0
            fx = dist * np.cos(elev_r) * np.cos(az)
            fy = dist * np.cos(elev_r) * np.sin(az)
            fz = dist * np.sin(elev_r)
            r = p.rayTest([fx, fy, fz], [-fx, -fy, -fz])
            if r[0] in body_ids: pts.append(r[3])
    pts = np.array(pts)
    np.save("points.npy", pts)
    print(f"[INFO] Generated {len(pts)} pts")
    return pts

points = sample_pointcloud(obstacle_ids)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CLUSTERING + HULL BODIES
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG_FILE = "dbscan_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f: params = json.load(f)
    eps, min_samples = params["eps"], params["min_samples"]
else:
    eps, min_samples = 0.15, 5

def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl, cnt in cluster_sizes.items() if lbl != -1 and cnt < min_points_threshold]
    for small_lbl in small_clusters:
        small_mask = labels == small_lbl
        small_pts  = points[small_mask]
        distances  = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1: continue
            mask = labels == lbl
            if not mask.any(): continue
            centroid = points[mask].mean(axis=0)
            distances.append((lbl, np.mean(np.linalg.norm(small_pts - centroid, axis=1))))
        if distances:
            closest = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest
    return labels

db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
labels = db.labels_.copy()
labels = group(points, labels)
unique_labels = np.unique(labels[labels >= 0])
cluster_pts   = [points[labels == lbl] for lbl in unique_labels]
print(f"[INFO] {len(unique_labels)} clusters found")

hull_body_ids = []
for i, cpts in enumerate(cluster_pts):
    try:
        hull    = ConvexHull(cpts)
        verts   = cpts[hull.vertices]
        col_id  = p.createCollisionShape(p.GEOM_MESH, vertices=verts.tolist(), meshScale=[1, 1, 1])
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, basePosition=[0, 0, 0])
        hull_body_ids.append(body_id)
    except Exception as e:
        print(f"  Cluster {i}: failed ({e})")
print(f"[INFO] {len(hull_body_ids)} hull bodies ready")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  SDF  (cache keyed on post-wrap coords to avoid duplicate entries)
# ═══════════════════════════════════════════════════════════════════════════════

_sdf_cache = {}

def sdf_scene(arm_id, threshold=10.0):
    min_d = threshold
    for hull_id in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=hull_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d: min_d = d
    return min_d

def set_config(q1, q2):
    p.resetJointState(arm2Id, 0, float(q1))
    p.resetJointState(arm2Id, 1, float(q2))
    p.stepSimulation()

def eval_sdf(q1, q2):
    # Always cache on the wrapped form so ±π aliases map to the same key
    wq1 = float((float(q1) + np.pi) % (2 * np.pi) - np.pi)
    wq2 = float((float(q2) + np.pi) % (2 * np.pi) - np.pi)
    key = (round(wq1, 5), round(wq2, 5))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(wq1, wq2)
    d = sdf_scene(arm2Id)
    _sdf_cache[key] = d
    return d

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  GUI OVERLAY  — reuses an already-computed sdf_val; NO extra eval_sdf call
# ═══════════════════════════════════════════════════════════════════════════════

_label_id = None
_sdf_id   = None

def update_gui_overlay(q1, q2, label, sdf_val):
    """
    Visually updates the PyBullet debug overlay.  sdf_val must be supplied by
    the caller (already computed); this function does NOT query the SDF itself.
    """
    global _label_id, _sdf_id
    set_config(q1, q2)
    link2_state = p.getLinkState(arm2Id, 1, computeForwardKinematics=True)
    tip   = list(link2_state[4])
    above = [tip[0], tip[1], tip[2] + 0.40]
    below = [tip[0], tip[1], tip[2] + 0.15]
    txt_color = [0.9, 0.2, 0.2] if sdf_val < 0 else [0.1, 0.78, 0.2]
    lkw = dict(textColorRGB=txt_color, textSize=1.4)
    if _label_id is not None: lkw["replaceItemUniqueId"] = _label_id
    _label_id = p.addUserDebugText(label, above, **lkw)
    skw = dict(textColorRGB=txt_color, textSize=1.1)
    if _sdf_id is not None: skw["replaceItemUniqueId"] = _sdf_id
    _sdf_id = p.addUserDebugText(
        f"SDF={sdf_val:+.4f}m  ({'COLL' if sdf_val < 0 else 'free'})", below, **skw)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

BLUE   = "#3A7DC9"; ORANGE = "#E8882A"; RED    = "#C93A3A"
GREEN  = "#2E9E5B"; PURPLE = "#7F3FBF"; GRAY   = "#888780"

plt.rcParams.update({
    "figure.facecolor":    "white", "axes.facecolor":      "#F8F9FA",
    "axes.spines.top":     False,   "axes.spines.right":   False,
    "axes.grid":           True,    "grid.alpha":          0.25,
    "grid.linestyle":      "--",    "font.size":           11,
    "axes.titlesize":      12,      "axes.titleweight":    "bold",
    "axes.labelsize":      11,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  ANGULAR WRAPPING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

START = np.array([-1.0,  1.2])
GOAL  = np.array([ -1.5, -0.8])

def wrap_angle(cfg):
    return (cfg + np.pi) % (2 * np.pi) - np.pi

def toroidal_dist(a, b):
    diff = np.abs(a - b)
    diff = np.where(diff > np.pi, 2 * np.pi - diff, diff)
    return float(np.linalg.norm(diff))

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  INITIAL LINE SCAN  (seeds for wavefront)
# ═══════════════════════════════════════════════════════════════════════════════

N_SEGMENTS  = 5
N_SAMPLES   = N_SEGMENTS + 1
t_vals      = np.linspace(0.0, 1.0, N_SAMPLES)
sample_cfgs = np.array([wrap_angle(START + t * (GOAL - START)) for t in t_vals])

print(f"\n[INFO] Evaluating SDF along path segments...")
sdf_vals    = []
line_samples = []

for idx, cfg in enumerate(sample_cfgs):
    d = eval_sdf(cfg[0], cfg[1])
    sdf_vals.append(d)
    line_samples.append((cfg.copy(), d, t_vals[idx]))

sdf_vals = np.array(sdf_vals)

def regula_falsi(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b = cfg_a.copy(), cfg_b.copy()
    diff = b - a
    diff = (diff + np.pi) % (2 * np.pi) - np.pi
    b = a + diff
    fa, fb = sdf_a, sdf_b
    cfg_c  = a.copy()
    for n in range(max_iter):
        cfg_c = a + fa * (a - b) / (fb - fa)
        wrapped_c = wrap_angle(cfg_c)
        fc = eval_sdf(wrapped_c[0], wrapped_c[1])
        if abs(fc) < tol:
            return wrapped_c, fc, n + 1
        if fa * fc < 0:
            b, fb = wrapped_c, fc
        else:
            a, fa = wrapped_c, fc
    return wrap_angle(cfg_c), eval_sdf(cfg_c[0], cfg_c[1]), max_iter

rf_line_roots = []
for i in range(N_SEGMENTS):
    si, sj = sdf_vals[i], sdf_vals[i + 1]
    if si * sj < 0:
        cfg_r, sdf_r, _ = regula_falsi(sample_cfgs[i], sample_cfgs[i + 1], si, sj)
        rf_line_roots.append((cfg_r, sdf_r))

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  NODE GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

TIME_LIMIT_S  = 120.0
MAX_NODES     = 2500

_sdf_query_count = 0
nodes            = []
queue            = deque()
visited          = set()

def make_node(cfg, kind='rf', parent=-1):
    nid = len(nodes)
    nodes.append(dict(cfg=wrap_angle(cfg), kind=kind, dead=False, parent=parent, rays=[]))
    return nid

def _pos_key(cfg):
    r = CLUB_RADIUS / 2.0
    w = wrap_angle(cfg)
    return (round(w[0] / r), round(w[1] / r))

def find_nearby_node(cfg, exclude_id=-1):
    best_id, best_d = -1, CLUB_RADIUS
    for i, nd in enumerate(nodes):
        if i == exclude_id: continue
        d = toroidal_dist(nd['cfg'], cfg)
        if d < best_d:
            best_d = d; best_id = i
    return best_id

# Populate Seeds
if not rf_line_roots:
    make_node(START, kind='rf', parent=-1)
else:
    for cfg_r, _ in rf_line_roots:
        make_node(cfg_r, kind='rf', parent=-1)

closest_seed = min(range(len(nodes)), key=lambda i: toroidal_dist(nodes[i]['cfg'], START))
queue.append(closest_seed)

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  WAVEFRONT TOROIDAL EXPANSION LOOP
# ═══════════════════════════════════════════════════════════════════════════════

DIRS = [(1, 0, "+q1"), (-1, 0, "-q1"), (0, 1, "+q2"), (0, -1, "-q2")]

goal_reached = False
stop_reason  = "queue exhausted"

while queue and not goal_reached:
    if (time.time() - ALGORITHM_START_TIME) >= TIME_LIMIT_S:
        stop_reason = "time limit reached"; break
    if len(nodes) >= MAX_NODES:
        stop_reason = "node cap reached"; break

    nid = queue.popleft()
    nd  = nodes[nid]
    if nd['dead']: continue

    pkey = _pos_key(nd['cfg'])
    if pkey in visited:
        nd['dead'] = True; continue
    visited.add(pkey)

    cfg0 = nd['cfg']
    print(f"[EXPAND] Node #{nid} q=({np.degrees(cfg0[0]):+.1f}°, {np.degrees(cfg0[1]):+.1f}°)")

    for dq1, dq2, dname in DIRS:
        d_vec   = np.array([dq1, dq2], dtype=float)
        adj_cfg = wrap_angle(cfg0 + d_vec * ADJ_OFFSET)
        adj_sdf = eval_sdf(adj_cfg[0], adj_cfg[1])
        _sdf_query_count += 1

        ray_rec = dict(dir_name=dname, raw_segments=[], status='', root=None)

        if adj_sdf < 0:
            ray_rec['raw_segments'] = [[cfg0.copy(), adj_cfg.copy()]]
            ray_rec['status'] = 'dead'
            nd['rays'].append(ray_rec)
            continue

        current_segment = [cfg0.copy(), adj_cfg.copy()]
        all_segments    = []
        prev_cfg, prev_sdf = adj_cfg.copy(), adj_sdf
        found_root = False

        for step in range(1, MAX_STEPS_PER_RAY + 1):
            unwrapped_cur = cfg0 + d_vec * (ADJ_OFFSET + step * STEP_SIZE)
            cur_cfg       = wrap_angle(unwrapped_cur)

            if np.linalg.norm(cur_cfg - prev_cfg) > np.pi:
                all_segments.append(current_segment)
                current_segment = []

            current_segment.append(cur_cfg.copy())
            cur_sdf = eval_sdf(cur_cfg[0], cur_cfg[1])
            _sdf_query_count += 1

            # Pass the already-computed sdf_val to the overlay (NO extra query)
            update_gui_overlay(cur_cfg[0], cur_cfg[1], f"ray #{nid}", cur_sdf)

            if prev_sdf * cur_sdf < 0:
                root_cfg, root_sdf, _ = regula_falsi(prev_cfg, cur_cfg, prev_sdf, cur_sdf)

                if np.linalg.norm(root_cfg - current_segment[-1]) > np.pi:
                    all_segments.append(current_segment)
                    current_segment = [root_cfg.copy()]
                else:
                    current_segment.append(root_cfg.copy())

                all_segments.append(current_segment)
                current_segment = []

                ray_rec['root']   = root_cfg.copy()
                ray_rec['status'] = 'explored'
                found_root        = True

                nearby_id = find_nearby_node(root_cfg, exclude_id=nid)
                if nearby_id < 0 and _pos_key(root_cfg) not in visited:
                    new_id = make_node(root_cfg, kind='rf', parent=nid)
                    if toroidal_dist(root_cfg, GOAL) < GOAL_REACH_RADIUS:
                        goal_reached = True
                        stop_reason  = "goal reached"
                        make_node(GOAL, kind='goal', parent=new_id)
                    queue.append(new_id)
                break

            prev_cfg, prev_sdf = cur_cfg.copy(), cur_sdf

        if current_segment:
            all_segments.append(current_segment)
        ray_rec['raw_segments'] = all_segments
        if not found_root:
            ray_rec['status'] = 'wrapped_pass'
        nd['rays'].append(ray_rec)
        if goal_reached: break

TOTAL_ELAPSED_TIME = time.time() - ALGORITHM_START_TIME

print("\n" + "═"*79)
print(f"[BENCHMARK] EXPLORATION METRICS & TIMING PERFORMANCE")
print(f"  • Total Time Taken (Scan Start → Goal/End): {TOTAL_ELAPSED_TIME:.4f} seconds")
print(f"  • Wavefront Stop Reason                : {stop_reason.upper()}")
print(f"  • Discovered Nodes Record Count        : {len(nodes)}")
print(f"  • Collision Engine Queries (SDF)       : {_sdf_query_count}")
print(f"  • SDF Cache Hits                       : {len(_sdf_cache)} unique configs cached")
print("═"*79 + "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 12.  FINAL PLOT VISUALISATION WITH SEAM CUTS
# ═══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("#F8F9FA")
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
ax.set_xlabel("q₁ — Shoulder (degrees)"); ax.set_ylabel("q₂ — Elbow (degrees)")
ax.axhline(0, color=GRAY, lw=0.6, ls=':', alpha=0.5)
ax.axvline(0, color=GRAY, lw=0.6, ls=':', alpha=0.5)

ax.plot(np.degrees([START[0], GOAL[0]]), np.degrees([START[1], GOAL[1]]),
        color=BLUE, lw=2, label="Initial C-space path")

# Draw rays
for nd in nodes:
    for ray in nd['rays']:
        for seg in ray['raw_segments']:
            if len(seg) < 2: continue
            xs = [np.degrees(q[0]) for q in seg]
            ys = [np.degrees(q[1]) for q in seg]
            if ray['status'] == 'dead':
                lc, ls, lw = RED, (0, (2, 2)), 1.2
            elif ray['status'] == 'wrapped_pass':
                lc, ls, lw = GREEN, '-', 0.8
            else:
                lc, ls, lw = '#AAAAAA', '-', 1.0
            ax.plot(xs, ys, color=lc, linestyle=ls, linewidth=lw, alpha=0.6)
        if ray['root'] is not None:
            ax.scatter(np.degrees(ray['root'][0]), np.degrees(ray['root'][1]),
                       s=140, marker='X', color=PURPLE,
                       edgecolors='white', zorder=8)

# Node markers with ID labels
for nid, nd in enumerate(nodes):
    q1d, q2d = np.degrees(nd['cfg'][0]), np.degrees(nd['cfg'][1])
    if nd['dead']:
        mk, col, sz = 'X', '#B0B0B0', 80
    elif nd['kind'] == 'goal':
        mk, col, sz = '*', GREEN, 250
    else:
        mk, col, sz = 'X', PURPLE, 140

    ax.scatter(q1d, q2d, s=sz, marker=mk, color=col,
               edgecolors='white', linewidths=1.2, zorder=9)
    ax.annotate(f"#{nid}", xy=(q1d, q2d),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold', color=col, alpha=0.85, zorder=10)

ax.scatter(np.degrees(START[0]), np.degrees(START[1]),
           s=220, marker='*', color=GREEN, edgecolors='white', zorder=12, label="START")
ax.scatter(np.degrees(GOAL[0]), np.degrees(GOAL[1]),
           s=220, marker='*', color=ORANGE, edgecolors='white', zorder=12, label="GOAL")
ax.add_patch(plt.Circle((np.degrees(GOAL[0]), np.degrees(GOAL[1])),
                         np.degrees(GOAL_REACH_RADIUS),
                         fill=False, edgecolor=ORANGE, lw=1.2, ls='--'))

handles = [
    plt.Line2D([0], [0], color=BLUE,      lw=2,               label="Initial C-space path"),
    plt.Line2D([0], [0], color='#AAAAAA', lw=1.0,             label="Explored ray segment"),
    plt.Line2D([0], [0], color=RED,       lw=1.2, ls=(0,(2,2)),label="Dead ray (adj obstacle)"),
    plt.Line2D([0], [0], color=GREEN,     lw=0.8,             label="Boundary Wrapper Ray"),
    plt.scatter([], [], s=100, marker='X', color=PURPLE, ec='white', label="Boundary Root (SDF=0)"),
]
ax.legend(handles=handles, loc='upper right', framealpha=0.9)
ax.set_title(
    f"Continuous Toroidal C-Space Wavefront Explorer\n"
    f"Compute Duration: {TOTAL_ELAPSED_TIME:.3f}s | Reason: {stop_reason.upper()}")
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# 13.  KEEP PYBULLET GUI ALIVE
# ═══════════════════════════════════════════════════════════════════════════════
p.addUserDebugText("Done — close window to exit",
                   [0, 0, 2.5], textColorRGB=[0.8, 0.8, 0.1], textSize=1.6)
while True:
    p.stepSimulation()
    time.sleep(1 / 60)
    try: p.getConnectionInfo()
    except Exception: break
p.disconnect()