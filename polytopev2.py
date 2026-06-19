"""
cspace_ray_frontier.py
──────────────────────
Keeps everything up to and including triangulation + line-scan + regula-falsi
unchanged. Replaces Dijkstra with a two-frontier ray-expansion approach.

Bootstrap:
  • Original line-scan midpoint → 360° fan of n_rays
  • Boundary points split into left/right hemispheres (relative to START→GOAL)
  • Adjacent pairs of boundary points → midpoints stored
  • Leftmost-left midpoint  → LEFT  frontier seed
  • Rightmost-right midpoint → RIGHT frontier seed

Recursive expansion (both frontiers run simultaneously, one step at a time):
  • Each frontier: shoot n_rays within a 180° sector centered on the
    *forward* direction along the centerline (previous_midpoint → current_midpoint).
    The centerline itself is the symmetric axis: top = one side of the line,
    bottom = the other side. The backward half (toward the parent) is excluded.
  • Rays have max length l_ray, sampled at l_ray/5 intervals
  • Stop at first >=0 sdf point, then regula-falsi for exact boundary
  • Pair adjacent found-boundary-points (top with bottom) → midpoints stored
  • Pick NEW extreme midpoint (furthest left or right) as next frontier seed

Termination per frontier:
  • All new midpoints within d_threshold of each other → stop that frontier
  • Fan angle < 20° → stop (only applies once a centerline exists)

Plotting:
  • All midpoints stored (not just frontier ones)
  • Boundary points, rays, midpoints all visualised on final plot
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
from collections import deque, defaultdict
import math

# ═══════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ═══════════════════════════════════════════════════════

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

p.configureDebugVisualizer(p.COV_ENABLE_GUI,            0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS,        1)
p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
p.resetDebugVisualizerCamera(
    cameraDistance=5.0, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0, 0, 1])

arm2Id = p.loadURDF(
    "arm_2dof.urdf", basePosition=[0,0,0],
    baseOrientation=p.getQuaternionFromEuler([0,0,0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS_2 = p.getNumJoints(arm2Id)
print(f"[INFO] 2-DOF arm loaded — {NUM_JOINTS_2} joints")

# ═══════════════════════════════════════════════════════
# 2.  OBSTACLES
# ═══════════════════════════════════════════════════════

def create_box(half_extents, position, orientation, mass=0, color=[1,0,0,1]):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_sphere(radius, position, orientation=[0,0,0], mass=0, color=[0,1,0,1]):
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

def create_cylinder(radius, height, position, orientation=[0,0,0], mass=0, color=[0,0,1,1]):
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    return p.createMultiBody(mass, col, vis, position, p.getQuaternionFromEuler(orientation))

boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],      orientation=[0.2,1.1,0.4])
sphereId   = create_sphere(radius=1,               position=[0,2,1])
cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1],   orientation=[1.3,0,0])
obstacle_ids   = [boxId, sphereId, cylinderId]
obstacle_names = ["box", "sphere", "cylinder"]

# ═══════════════════════════════════════════════════════
# 3.  POINTCLOUD
# ═══════════════════════════════════════════════════════

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
        r = p.rayTest([dx,dy,6.0],[dx,dy,-1.0])
        if r[0] in body_ids: pts.append(r[3])
    for angle in np.linspace(0,2*np.pi,72,endpoint=False):
        for height in np.linspace(0.0,3.0,30):
            ox,oy = 8*np.cos(angle),8*np.sin(angle)
            r = p.rayTest([ox,oy,height],[-ox,-oy,height])
            if r[0] in body_ids: pts.append(r[3])
    for elev in [20,40,60,80]:
        elev_r = np.radians(elev)
        for az in np.linspace(0,2*np.pi,36,endpoint=False):
            dist=7.0; fx=dist*np.cos(elev_r)*np.cos(az)
            fy=dist*np.cos(elev_r)*np.sin(az); fz=dist*np.sin(elev_r)
            r=p.rayTest([fx,fy,fz],[-fx,-fy,-fz])
            if r[0] in body_ids: pts.append(r[3])
    pts = np.array(pts)
    np.save("points.npy", pts)
    print(f"[INFO] Generated {len(pts)} pts")
    return pts

points = sample_pointcloud(obstacle_ids)

# ═══════════════════════════════════════════════════════
# 4.  CLUSTERING + HULL BODIES
# ═══════════════════════════════════════════════════════

CONFIG_FILE = "dbscan_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f: params = json.load(f)
    eps, min_samples = params["eps"], params["min_samples"]
else:
    eps, min_samples = 0.15, 5

def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl,cnt in cluster_sizes.items()
                      if lbl != -1 and cnt < min_points_threshold]
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
        col_id  = p.createCollisionShape(p.GEOM_MESH, vertices=verts.tolist(), meshScale=[1,1,1])
        body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, basePosition=[0,0,0])
        hull_body_ids.append(body_id)
    except Exception as e:
        print(f"  Cluster {i}: failed ({e})")
print(f"[INFO] {len(hull_body_ids)} hull bodies ready")

# ═══════════════════════════════════════════════════════
# 5.  SDF
# ═══════════════════════════════════════════════════════

_sdf_cache = {}
_sdf_query_count = 0

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
    global _sdf_query_count
    key = (round(float(q1), 6), round(float(q2), 6))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2)
    d = sdf_scene(arm2Id)
    _sdf_cache[key] = d
    _sdf_query_count += 1
    return d

# ═══════════════════════════════════════════════════════
# 6.  GUI OVERLAY
# ═══════════════════════════════════════════════════════

_label_id = None
_sdf_id   = None

def update_gui_overlay(q1, q2, label, sdf_val):
    global _label_id, _sdf_id
    set_config(q1, q2)
    link2_state = p.getLinkState(arm2Id, 1, computeForwardKinematics=True)
    tip = list(link2_state[4])
    above = [tip[0], tip[1], tip[2]+0.40]
    below = [tip[0], tip[1], tip[2]+0.15]
    txt_color = [0.9,0.2,0.2] if sdf_val < 0 else [0.1,0.78,0.2]
    lkw = dict(textColorRGB=txt_color, textSize=1.4)
    if _label_id is not None: lkw["replaceItemUniqueId"] = _label_id
    _label_id = p.addUserDebugText(label, above, **lkw)
    skw = dict(textColorRGB=txt_color, textSize=1.1)
    if _sdf_id is not None: skw["replaceItemUniqueId"] = _sdf_id
    _sdf_id = p.addUserDebugText(
        f"SDF={sdf_val:+.4f}m  ({'COLL' if sdf_val<0 else 'free'})", below, **skw)

# ═══════════════════════════════════════════════════════
# 7.  COLOUR PALETTE
# ═══════════════════════════════════════════════════════

BLUE   = "#3A7DC9"; ORANGE = "#E8882A"; RED    = "#C93A3A"
GREEN  = "#2E9E5B"; PURPLE = "#7F3FBF"; GRAY   = "#888780"
LBLUE  = "#B5D4F4"; YELLOW = "#E8C22A"; TEAL   = "#1A9E8F"
PINK   = "#E83A8A"; CYAN   = "#1AC9C9"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#F8F9FA",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
    "font.size":11,"axes.titlesize":12,"axes.titleweight":"bold","axes.labelsize":11,
})

# ═══════════════════════════════════════════════════════
# 8.  START / GOAL + LINE SCAN
# ═══════════════════════════════════════════════════════

START = np.array([.508, 1.0])
GOAL  = np.array([-.482, -1.443])

l_ray = float(np.linalg.norm(GOAL - START))   # max ray length = dist(start,goal)
print(f"[INFO] l_ray = {l_ray:.4f} rad")

# START→GOAL direction and perpendicular
sg_dir = (GOAL - START) / l_ray               # unit vector along path
# left perpendicular (CCW 90°)
sg_perp_left  = np.array([-sg_dir[1],  sg_dir[0]])
sg_perp_right = np.array([ sg_dir[1], -sg_dir[0]])

N_SEGMENTS  = 5
N_SAMPLES   = N_SEGMENTS + 1
t_vals      = np.linspace(0.0, 1.0, N_SAMPLES)
sample_cfgs = np.array([START + t*(GOAL-START) for t in t_vals])

print(f"\n[INFO] Start: q1={np.degrees(START[0]):.1f}°  q2={np.degrees(START[1]):.1f}°")
print(f"[INFO] Goal : q1={np.degrees(GOAL[0]):.1f}°  q2={np.degrees(GOAL[1]):.1f}°")

# ═══════════════════════════════════════════════════════
# 9.  TRIANGULATION (unchanged — kept for plot mesh)
# ═══════════════════════════════════════════════════════

GRID_N = int(np.ceil((2 * np.pi) / 0.1)) + 1
q_vals_g = np.linspace(-np.pi, np.pi, GRID_N)
QQ1, QQ2 = np.meshgrid(q_vals_g, q_vals_g)
vertices_tri = np.column_stack([QQ1.ravel(), QQ2.ravel()])
N_G = GRID_N

triangles_tri = []
for i in range(N_G-1):
    for j in range(N_G-1):
        tl = i*N_G + j; tr = i*N_G + j+1
        bl = (i+1)*N_G + j; br = (i+1)*N_G + j+1
        triangles_tri.append([tl, tr, bl])
        triangles_tri.append([tr, br, bl])
triangles_tri = np.array(triangles_tri)
vertices_deg = np.degrees(vertices_tri)
print(f"[INFO] Triangulation: {N_G}×{N_G} grid → {len(triangles_tri)} triangles (for display)")

# ═══════════════════════════════════════════════════════
# 10. LINE SCAN + REGULA FALSI
# ═══════════════════════════════════════════════════════

start_processing_time = time.perf_counter()

print(f"\n[INFO] Evaluating SDF at {N_SAMPLES} samples …\n")
sdf_vals_line = []
plot_samples  = []

for idx, cfg in enumerate(sample_cfgs):
    d = eval_sdf(cfg[0], cfg[1])
    if idx == 0 or idx == N_SAMPLES - 1:
        d = abs(d) if d != 0 else 1e-6
    sdf_vals_line.append(d)
    print(f"  Sample {idx}  t={t_vals[idx]:.2f}"
          f"  q=({np.degrees(cfg[0]):+.2f}°,{np.degrees(cfg[1]):+.2f}°)"
          f"  SDF={d:+.5f} ({'COLL' if d<0 else 'free'})")
    update_gui_overlay(cfg[0], cfg[1], f"Sample {idx}  t={t_vals[idx]:.1f}", d)
    plot_samples.append((cfg, d, t_vals[idx]))

sdf_vals_line = np.array(sdf_vals_line)

def regula_falsi(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b = cfg_a.copy(), cfg_b.copy()
    fa, fb = sdf_a, sdf_b
    cfg_c, fc = a, fa
    for n in range(max_iter):
        denom = fb - fa
        if abs(denom) < 1e-15: break
        cfg_c = a + fa*(a-b)/denom
        fc    = eval_sdf(cfg_c[0], cfg_c[1])
        if abs(fc) < tol:
            return cfg_c, fc, n+1
        if fa*fc < 0: b, fb = cfg_c, fc
        else:          a, fa = cfg_c, fc
    return cfg_c, fc, max_iter

rf_line_roots = []
print(f"\n[INFO] Line-scan Regula Falsi …\n")
for i in range(N_SEGMENTS):
    si, sj = sdf_vals_line[i], sdf_vals_line[i+1]
    if si*sj < 0:
        cfg_r, sdf_r, nit = regula_falsi(sample_cfgs[i], sample_cfgs[i+1], si, sj)
        rf_line_roots.append((cfg_r, sdf_r))
        print(f"  RF root seg {i}→{i+1}:  q=({np.degrees(cfg_r[0]):+.3f}°,"
              f"{np.degrees(cfg_r[1]):+.3f}°)  SDF={sdf_r:+.6f}  ({nit} iters)")
        update_gui_overlay(cfg_r[0], cfg_r[1], f"RF zero  seg {i}→{i+1}", sdf_r)

if not rf_line_roots:
    print("  [NOTE] No sign changes on the line — cannot bootstrap frontier expansion.")

# ═══════════════════════════════════════════════════════
# 11. RAY-FRONTIER PARAMETERS
# ═══════════════════════════════════════════════════════

N_RAYS       = 12        # number of rays per fan
RAY_STEPS    = 5         # samples per ray (intervals = l_ray / RAY_STEPS)
THETA_TERM   = np.radians(20.0)   # terminate a frontier when fan angle < 20°
D_THRESHOLD  = 0.1       # terminate when all new midpoints within this distance (rad)
MAX_DEPTH    = 30        # safety cap on recursion depth
FRONTIER_FAN_ANGLE = np.radians(160.0)  # fan width for recursive frontier steps
                                          # (centered on forward dir along centerline,
                                          # strictly excludes points behind the parent)

# ── Storage for plot ────────────────────────────────────
all_midpoints     = []   # list of np.array cfg (all, not just frontier)
all_boundary_pts  = []   # list of np.array cfg
all_rays          = []   # list of (origin, endpoint) tuples
all_centerlines   = []   # list of (parent_midpt, child_midpt) tuples, for plotting

# colour tagging: 'left' / 'right' / 'seed'
midpoint_tags  = []       # parallel to all_midpoints
free_midpoints = []       # midpoints that landed in free space (sdf >= 0)

def register_midpoint(cfg, tag):
    """Evaluate SDF at cfg, store in all_midpoints (collision) or free_midpoints."""
    sdf_v = eval_sdf(cfg[0], cfg[1])
    if sdf_v < 0:
        all_midpoints.append(cfg.copy())
        midpoint_tags.append(tag)
    else:
        free_midpoints.append(cfg.copy())
    return sdf_v

# ═══════════════════════════════════════════════════════
# 12. RAY UTILITIES
# ═══════════════════════════════════════════════════════

def shoot_ray(origin, direction, l_ray, n_steps=RAY_STEPS):
    """
    Shoot a ray from origin in unit-direction `direction` up to l_ray.
    Sample at l_ray/n_steps intervals.
    Return (boundary_cfg, boundary_sdf) if a sign change is found,
    else None.
    """
    step = l_ray / n_steps
    prev_cfg = origin.copy()
    prev_sdf = eval_sdf(origin[0], origin[1])
    endpoint = origin  # track last sample for plot

    for k in range(1, n_steps + 1):
        t = k * step
        cfg = origin + t * direction
        sdf_v = eval_sdf(cfg[0], cfg[1])
        endpoint = cfg
        if prev_sdf < 0 and sdf_v >= 0:
            # sign change: refine with regula falsi
            boundary, b_sdf, _ = regula_falsi(prev_cfg, cfg, prev_sdf, sdf_v)
            all_rays.append((origin.copy(), boundary.copy()))
            all_boundary_pts.append(boundary.copy())
            return boundary, b_sdf
        prev_cfg = cfg
        prev_sdf = sdf_v

    all_rays.append((origin.copy(), endpoint.copy()))
    return None   # no boundary found

def fan_boundary_points(origin, angle_start, angle_end, n_rays, l_ray):
    """
    Shoot n_rays uniformly spread between angle_start and angle_end (radians,
    measured in C-space from +q1 axis). Return list of boundary cfg arrays
    that were found (may be fewer than n_rays if some rays hit nothing).
    Angles in [0, 2π).
    """
    # evenly space n_rays across the angular sector
    angles = np.linspace(angle_start, angle_end, n_rays, endpoint=False)
    # add half-step offset so rays don't sit exactly on the boundary angles
    offset = (angle_end - angle_start) / (2 * n_rays)
    angles = angles + offset

    found = []
    for ang in angles:
        direction = np.array([np.cos(ang), np.sin(ang)])
        result = shoot_ray(origin, direction, l_ray)
        if result is not None:
            found.append(result[0])   # just the cfg
    return found

def classify_left_right(cfg, origin):
    """
    Returns 'left' if cfg is to the left of the START→GOAL line
    (i.e. positive dot with sg_perp_left), else 'right'.
    """
    v = cfg - origin
    return 'left' if np.dot(v, sg_perp_left) >= 0 else 'right'

def angle_of(direction):
    """Angle of a 2D direction vector in [0, 2π)."""
    a = math.atan2(direction[1], direction[0])
    return a % (2 * math.pi)

def signed_arc(a_start, a_end):
    """CCW arc from a_start to a_end in (-π, π]."""
    diff = (a_end - a_start) % (2 * math.pi)
    if diff > math.pi: diff -= 2 * math.pi
    return diff

def pair_top_bottom(boundary_cfgs, origin, arc_start, arc_end):
    """
    Given boundary cfgs found within an angular arc [arc_start → arc_end] (CCW),
    split the arc at its midpoint angle:
      upper = first half of arc (closer to arc_start),  sorted by pos in arc
      lower = second half of arc (closer to arc_end),   sorted by pos in arc
    Pair: (upper[0], lower[0]) → midpoint 1,  (upper[1], lower[1]) → midpoint 2 …
    Points whose angular position falls outside [0, arc_span) are clamped into
    the nearer half (handles floating-point boundary cases).
    Returns list of (cfg_upper, cfg_lower, midpoint).
    """
    arc_span = (arc_end - arc_start) % (2 * math.pi)
    if arc_span < 1e-9:
        arc_span = 2 * math.pi   # degenerate — treat as full circle

    upper, lower = [], []
    for c in boundary_cfgs:
        ang = angle_of(c - origin)
        pos = (ang - arc_start) % (2 * math.pi)
        # clamp into [0, arc_span) — anything just outside goes to the nearer half
        if pos > arc_span:
            pos = arc_span - 1e-9 if pos > (arc_span + 2*math.pi)/2 else 0.0
        if pos < arc_span / 2.0:
            upper.append((pos, c))
        else:
            lower.append((pos, c))

    upper.sort(key=lambda x: x[0])
    lower.sort(key=lambda x: x[0])

    upper_cfgs = [c for _, c in upper]
    lower_cfgs = [c for _, c in lower]

    pairs = []
    for cu, cl in zip(upper_cfgs, lower_cfgs):
        mid = (cu + cl) / 2.0
        pairs.append((cu, cl, mid))
    return pairs

def pair_top_bottom_centerline(boundary_cfgs, origin, fwd_dir):
    """
    Split boundary_cfgs into 'top'/'bottom' relative to the centerline
    (the line through `origin` along `fwd_dir`), using the SIGN of the
    2D cross product fwd_dir × (c - origin):
      cross > 0  → top    (left of centerline, CCW side)
      cross <= 0 → bottom (right of centerline, CW side)
    Each side is then sorted by raw angle from origin (same convention
    as the original pair_top_bottom), and paired top[i] with bottom[i].
    Returns list of (cfg_top, cfg_bottom, midpoint).
    """
    top, bottom = [], []
    for c in boundary_cfgs:
        v = c - origin
        cross = fwd_dir[0]*v[1] - fwd_dir[1]*v[0]
        ang = angle_of(v)
        if cross > 0:
            top.append((ang, c))
        else:
            bottom.append((ang, c))

    top.sort(key=lambda x: x[0])
    bottom.sort(key=lambda x: x[0])

    top_cfgs    = [c for _, c in top]
    bottom_cfgs = [c for _, c in bottom]

    pairs = []
    for ct, cb in zip(top_cfgs, bottom_cfgs):
        mid = (ct + cb) / 2.0
        pairs.append((ct, cb, mid))
    return pairs

# ═══════════════════════════════════════════════════════
# 13. BOOTSTRAP — omnidirectional fan from first line-scan midpoint
# ═══════════════════════════════════════════════════════

# Build midpoints from consecutive rf_line_roots pairs
line_midpoints = []
for k in range(0, len(rf_line_roots) - 1, 2):
    cfg_a, _ = rf_line_roots[k]
    cfg_b, _ = rf_line_roots[k+1]
    line_midpoints.append((cfg_a, cfg_b, (cfg_a + cfg_b) / 2.0))

if not line_midpoints:
    print("[WARN] No line-scan midpoint pairs — cannot run frontier expansion.")
    left_frontier  = None
    right_frontier = None
else:
    # Use only the first midpoint for bootstrap (as specified)
    cfg_a0, cfg_b0, seed_mid = line_midpoints[0]
    seed_sdf = register_midpoint(seed_mid, 'seed')
    print(f"\n[INFO] Bootstrap seed midpoint: q=({np.degrees(seed_mid[0]):+.3f}°,"
          f"{np.degrees(seed_mid[1]):+.3f}°)  SDF={seed_sdf:+.6f}"
          f"  ({'COLL' if seed_sdf < 0 else 'FREE'})")
    update_gui_overlay(seed_mid[0], seed_mid[1], "Seed midpoint", seed_sdf)

    # ── Full 360° fan (no centerline exists yet — omnidirectional) ─────
    print(f"[INFO] Bootstrap: shooting {N_RAYS} rays omnidirectionally …")
    bootstrap_bpts = fan_boundary_points(
        seed_mid, 0.0, 2 * math.pi, N_RAYS, l_ray)
    print(f"  Found {len(bootstrap_bpts)} boundary points")
    for bp in bootstrap_bpts:
        all_boundary_pts.append(bp.copy())

    # Sort bootstrap boundary points by angle from seed_mid
    def sort_by_angle(bpts, origin):
        return sorted(bpts, key=lambda c: angle_of(c - origin))

    bootstrap_bpts_sorted = sort_by_angle(bootstrap_bpts, seed_mid)

    # Separate into left / right hemispheres
    left_bpts  = [c for c in bootstrap_bpts_sorted
                  if np.dot(c - seed_mid, sg_perp_left) >= 0]
    right_bpts = [c for c in bootstrap_bpts_sorted
                  if np.dot(c - seed_mid, sg_perp_left) < 0]

    print(f"  Left hemisphere: {len(left_bpts)} pts, Right: {len(right_bpts)} pts")

    # Re-sort each side by angle within their hemisphere
    left_bpts  = sort_by_angle(left_bpts,  seed_mid)
    right_bpts = sort_by_angle(right_bpts, seed_mid)

    # Define the arc bounds for each hemisphere.
    # sg_perp_left is at angle (sg_dir_ang + 90°).
    # The LEFT hemisphere arc (180°) is centred on sg_perp_left:
    #   arc_start_left = sg_dir_ang          (forward  = "bottom" of left arc)
    #   arc_end_left   = sg_dir_ang + 180°   (backward = "top"    of left arc)
    # travelling CCW from forward → left → backward passes through sg_perp_left.
    sg_dir_ang = angle_of(sg_dir)

    arc_start_left = sg_dir_ang                              # 0° into left arc
    arc_end_left   = (sg_dir_ang + math.pi) % (2*math.pi)   # 180° CCW = backward

    # RIGHT hemisphere arc (180°) centred on sg_perp_right:
    #   arc_start_right = sg_dir_ang + 180°  (backward = "bottom" of right arc)
    #   arc_end_right   = sg_dir_ang         (forward  = "top"    of right arc)
    arc_start_right = (sg_dir_ang + math.pi) % (2*math.pi)
    arc_end_right   = sg_dir_ang

    # Pair using top/bottom split within each hemisphere arc
    left_pairs  = pair_top_bottom(left_bpts,  seed_mid, arc_start_left,  arc_end_left)
    right_pairs = pair_top_bottom(right_bpts, seed_mid, arc_start_right, arc_end_right)

    print(f"  arc_start_left={np.degrees(arc_start_left):.1f}°  arc_end_left={np.degrees(arc_end_left):.1f}°"
          f"  → {len(left_pairs)} left pair(s)")
    print(f"  arc_start_right={np.degrees(arc_start_right):.1f}°  arc_end_right={np.degrees(arc_end_right):.1f}°"
          f"  → {len(right_pairs)} right pair(s)")

    for cu, cl, mid in left_pairs:
        register_midpoint(mid, 'left')
    for cu, cl, mid in right_pairs:
        register_midpoint(mid, 'right')

    # ── Identify left/right frontier seeds ─────────────
    # "Leftmost" = largest dot product with sg_perp_left
    def frontier_seed_from_pairs(pairs, side):
        """
        Among pairs, pick the one whose midpoint is most extreme
        on the given side ('left' → max dot with sg_perp_left,
                            'right' → min dot).
        Returns (cfg_a, cfg_b, midpoint) or None.
        """
        if not pairs: return None
        if side == 'left':
            return max(pairs, key=lambda t: np.dot(t[2] - seed_mid, sg_perp_left))
        else:
            return min(pairs, key=lambda t: np.dot(t[2] - seed_mid, sg_perp_left))

    left_seed_pair  = frontier_seed_from_pairs(left_pairs,  'left')
    right_seed_pair = frontier_seed_from_pairs(right_pairs, 'right')

    # Frontier state: (current_midpoint, previous_midpoint, side, depth)
    # previous_midpoint = None on the very first step → triggers omnidirectional
    # bootstrap-style behaviour is already done above, so first advance_frontier
    # call always HAS a previous_midpoint (= seed_mid).
    left_frontier  = (left_seed_pair[2],  seed_mid, 'left',  0) \
                     if left_seed_pair  else None
    right_frontier = (right_seed_pair[2], seed_mid, 'right', 0) \
                     if right_seed_pair else None

    if left_seed_pair is not None:
        all_centerlines.append((seed_mid.copy(), left_seed_pair[2].copy()))
    if right_seed_pair is not None:
        all_centerlines.append((seed_mid.copy(), right_seed_pair[2].copy()))

    print(f"[INFO] Left  frontier seed: {np.degrees(left_seed_pair[2])  if left_seed_pair  else 'None'}")
    print(f"[INFO] Right frontier seed: {np.degrees(right_seed_pair[2]) if right_seed_pair else 'None'}")

# ═══════════════════════════════════════════════════════
# 14. FRONTIER EXPANSION
# ═══════════════════════════════════════════════════════

def advance_frontier(frontier):
    """
    Given a frontier tuple (midpt, prev_midpt, side, depth):

      • centerline = midpt - prev_midpt   (the symmetric axis — NOT a
        perpendicular). "Top" = one side of this line, "bottom" = the other.
      • forward direction = unit(centerline), i.e. pointing AWAY from the
        parent and toward where the frontier is growing.
      • fan = FRONTIER_FAN_ANGLE (160°), centered on the forward direction:
            fan_start = angle(forward) - 80°
            fan_end   = angle(forward) + 80°
        This excludes a backward wedge (which would retrace toward the
        parent) and is left/right-symmetric about the centerline itself.
      • shoot n_rays across that fan from `midpt`
      • split found boundary points into top/bottom by which side of the
        centerline they fall on, pair top[i] with bottom[i] → new midpoints
      • pick the new seed as the pair whose midpoint has the GREATEST
        forward progress along fwd_dir (i.e. max dot((mid - midpt), fwd_dir)),
        so the frontier always advances away from its own parent rather
        than drifting sideways/backward relative to the original seed.

    Returns new frontier tuple, or None if termination condition met.
    """
    midpt, prev_midpt, side, depth = frontier

    if depth >= MAX_DEPTH:
        print(f"    [{side}] Max depth {MAX_DEPTH} reached — terminating.")
        return None

    centerline = midpt - prev_midpt
    cl_norm = np.linalg.norm(centerline)
    if cl_norm < 1e-9:
        print(f"    [{side}] Degenerate centerline — terminating.")
        return None
    fwd_dir = centerline / cl_norm
    fwd_ang = angle_of(fwd_dir)

    fan_angle = FRONTIER_FAN_ANGLE
    fan_start = fwd_ang - fan_angle / 2.0
    fan_end   = fwd_ang + fan_angle / 2.0

    print(f"    [{side}] depth={depth}  fan={np.degrees(fan_angle):.1f}°  fwd={np.degrees(fwd_ang):.1f}°  "
          f"midpt=({np.degrees(midpt[0]):+.2f}°,{np.degrees(midpt[1]):+.2f}°)")

    # Termination: fan angle < THETA_TERM (kept for parity with spec; with a
    # fixed FRONTIER_FAN_ANGLE this only fires if that constant is later set
    # below THETA_TERM, but the check stays in place as a safety guard).
    if fan_angle < THETA_TERM:
        print(f"    [{side}] Fan angle {np.degrees(fan_angle):.1f}° < "
              f"{np.degrees(THETA_TERM):.1f}° — terminating this frontier.")
        return None

    # Shoot n_rays within [fan_start, fan_end] (160°, centered on forward dir)
    new_bpts = fan_boundary_points(midpt, fan_start, fan_end, N_RAYS, l_ray)
    print(f"    [{side}] Found {len(new_bpts)} new boundary points")

    if len(new_bpts) < 2:
        print(f"    [{side}] Fewer than 2 boundary points — terminating.")
        return None

    # Split top/bottom by side of the centerline, pair, get new midpoints
    new_pairs = pair_top_bottom_centerline(new_bpts, midpt, fwd_dir)
    if not new_pairs:
        print(f"    [{side}] No pairs formed — terminating.")
        return None

    # Store all new midpoints (collision → all_midpoints, free → free_midpoints)
    new_mids = []
    for ct, cb, mid in new_pairs:
        sdf_v = register_midpoint(mid, side)
        update_gui_overlay(mid[0], mid[1], f"[{side}] depth={depth+1}", sdf_v)
        if sdf_v < 0:
            new_mids.append(mid)

    # Termination: all new midpoints within D_THRESHOLD of each other
    if len(new_mids) > 1:
        mids_arr = np.array(new_mids)
        pairwise_dists = [np.linalg.norm(mids_arr[i] - mids_arr[j])
                          for i in range(len(mids_arr))
                          for j in range(i+1, len(mids_arr))]
        if max(pairwise_dists) < D_THRESHOLD:
            print(f"    [{side}] All new midpoints within {D_THRESHOLD:.3f} rad of each other — terminating.")
            return None

    # Pick new seed as the pair with the GREATEST forward progress along
    # this frontier's own centerline direction (fwd_dir), measured from the
    # CURRENT midpt — not from the original seed. This is what makes the
    # frontier march strictly forward instead of clustering near its parent.
    best_pair = max(new_pairs, key=lambda t: np.dot(t[2] - midpt, fwd_dir))

    new_midpt = best_pair[2]
    all_centerlines.append((midpt.copy(), new_midpt.copy()))

    new_frontier = (new_midpt, midpt, side, depth + 1)
    return new_frontier


# ── Simultaneous expansion ──────────────────────────────
print(f"\n[INFO] Starting simultaneous left/right frontier expansion …")
print(f"  N_RAYS={N_RAYS}  RAY_STEPS={RAY_STEPS}  l_ray={l_ray:.3f}")
print(f"  THETA_TERM={np.degrees(THETA_TERM):.1f}°  D_THRESHOLD={D_THRESHOLD}")

step_num = 0
while left_frontier is not None or right_frontier is not None:
    step_num += 1
    print(f"\n  ── Step {step_num} ──")
    if left_frontier is not None:
        left_frontier  = advance_frontier(left_frontier)
    if right_frontier is not None:
        right_frontier = advance_frontier(right_frontier)

print(f"\n[INFO] Frontier expansion complete.")
print(f"  Total midpoints collected : {len(all_midpoints)}")
print(f"  Total boundary points     : {len(all_boundary_pts)}")
print(f"  Total rays shot           : {len(all_rays)}")

end_processing_time = time.perf_counter()
total_computation_time = end_processing_time - start_processing_time

print(f"\n⏱️  Computation time : {total_computation_time:.4f} s")
print(f"📊 SDF queries      : {_sdf_query_count}")

# ═══════════════════════════════════════════════════════
# 15. FINAL PLOT
# ═══════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("#F8F9FA")

# ── Triangulation mesh (light) ──────────────────────────
for tri in triangles_tri:
    pts = vertices_deg[tri]
    ax.add_patch(plt.Polygon(pts, fill=False, edgecolor="#DDDDDD", lw=0.2, alpha=0.3, zorder=1))

# ── Rays ───────────────────────────────────────────────
for origin, endpoint in all_rays:
    ax.plot([np.degrees(origin[0]), np.degrees(endpoint[0])],
            [np.degrees(origin[1]), np.degrees(endpoint[1])],
            color=TEAL, lw=0.7, alpha=0.5, zorder=2)

# ── Centerlines (parent midpoint → child midpoint, the symmetric axis) ─
for i, (p_mid, c_mid) in enumerate(all_centerlines):
    ax.plot([np.degrees(p_mid[0]), np.degrees(c_mid[0])],
            [np.degrees(p_mid[1]), np.degrees(c_mid[1])],
            color=PURPLE, lw=1.6, alpha=0.85, zorder=4, ls="-",
            label="Centerline (parent→child)" if i == 0 else "")

# ── Path line ──────────────────────────────────────────
ax.plot(np.degrees([START[0], GOAL[0]]),
        np.degrees([START[1], GOAL[1]]),
        color=BLUE, lw=2.0, ls="-", zorder=3, label="C-Space path")

# ── Line scan samples ──────────────────────────────────
for cfg, sdf_v, tv in plot_samples:
    col = RED if sdf_v < 0 else GREEN
    ax.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]),
               s=80, color=col, edgecolors="white", lw=1.0, zorder=5)

# ── RF line roots (boundary points on original line) ───
for cfg, sdf_v in rf_line_roots:
    ax.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]),
               s=220, marker="X", color=PURPLE, edgecolors="white", lw=1.5, zorder=6,
               label="RF zero (line)" if cfg is rf_line_roots[0][0] else "")

# ── All boundary points found by rays ──────────────────
bp_arr = np.array(all_boundary_pts) if all_boundary_pts else np.empty((0, 2))
if len(bp_arr):
    ax.scatter(np.degrees(bp_arr[:, 0]), np.degrees(bp_arr[:, 1]),
               s=60, color=ORANGE, edgecolors="white", lw=0.8, zorder=6,
               label=f"Ray boundary pts ({len(bp_arr)})", alpha=0.85)
    for i, bp in enumerate(all_boundary_pts):
        ax.annotate(f"B{i+1}",
                    xy=(np.degrees(bp[0]), np.degrees(bp[1])),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color=ORANGE, fontweight="bold", zorder=7)

# ── All midpoints (collision) ──────────────────────────
tag_colors = {'seed': YELLOW, 'left': PINK, 'right': CYAN}
tag_labels = {'seed': 'Seed midpoint', 'left': 'Left-frontier midpoints',
              'right': 'Right-frontier midpoints'}
plotted_tags = set()
for i, (mid, tag) in enumerate(zip(all_midpoints, midpoint_tags)):
    col = tag_colors.get(tag, GRAY)
    lbl = tag_labels.get(tag, tag) if tag not in plotted_tags else None
    plotted_tags.add(tag)
    ax.scatter(np.degrees(mid[0]), np.degrees(mid[1]),
               s=180, marker="D", color=col, edgecolors="black", lw=1.2, zorder=8,
               label=lbl)
    ax.annotate(f"M{i+1}",
                xy=(np.degrees(mid[0]), np.degrees(mid[1])),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, color="black", fontweight="bold", zorder=9)

# ── Free midpoints (landed in free space) ──────────────
if free_midpoints:
    for i, mid in enumerate(free_midpoints):
        lbl = "Free midpoint (not used)" if i == 0 else None
        ax.scatter(np.degrees(mid[0]), np.degrees(mid[1]),
                   s=180, marker="D", color="#AAAAAA", edgecolors="red", lw=1.8,
                   zorder=8, label=lbl)
        ax.annotate(f"FM{i+1}",
                    xy=(np.degrees(mid[0]), np.degrees(mid[1])),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color="red", fontweight="bold", zorder=9)
    print(f"[INFO] {len(free_midpoints)} free midpoint(s) stored (not used for expansion)")

# ── Start / Goal ───────────────────────────────────────
ax.scatter(*np.degrees(START), s=280, marker="*", color=GREEN,
           edgecolors="white", lw=1.5, zorder=9, label="Start")
ax.scatter(*np.degrees(GOAL),  s=280, marker="*", color=ORANGE,
           edgecolors="white", lw=1.5, zorder=9, label="Goal")
ax.annotate("START", xy=np.degrees(START), xytext=(12,-18),
            textcoords="offset points", fontsize=9, fontweight="bold", color=GREEN)
ax.annotate("GOAL",  xy=np.degrees(GOAL),  xytext=(12,-18),
            textcoords="offset points", fontsize=9, fontweight="bold", color=ORANGE)

ax.set_xlim(-185, 185); ax.set_ylim(-185, 185)
ax.set_xlabel("q₁ — Shoulder (degrees)")
ax.set_ylabel("q₂ — Elbow (degrees)")
ax.axhline(0, color=GRAY, lw=0.6, ls=":", alpha=0.5)
ax.axvline(0, color=GRAY, lw=0.6, ls=":", alpha=0.5)
ax.set_title(
    f"C-Space Ray Frontier Expansion\n"
    f"{len(all_midpoints)} midpoints | {len(all_boundary_pts)} boundary pts | "
    f"{len(all_rays)} rays\n"
    f"Time: {total_computation_time:.2f}s | SDF queries: {_sdf_query_count}"
)
ax.legend(frameon=True, fontsize=8, loc="upper right", framealpha=0.92)
ax.grid(True, alpha=0.25, linestyle="--")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)

plt.tight_layout()
fig.savefig("cspace_ray_frontier.png", dpi=150, bbox_inches="tight")
print("[INFO] Saved cspace_ray_frontier.png")
plt.show(block=True)
plt.close(fig)

# ═══════════════════════════════════════════════════════
# 16. KEEP PYBULLET ALIVE
# ═══════════════════════════════════════════════════════

if _label_id is not None: p.removeUserDebugItem(_label_id)
if _sdf_id   is not None: p.removeUserDebugItem(_sdf_id)
p.addUserDebugText("Done — close window to exit", [0,0,2.5],
                   textColorRGB=[0.8,0.8,0.1], textSize=1.6)
print("\n[INFO] PyBullet open — close it to exit.")
while True:
    p.stepSimulation()
    time.sleep(1/60)
    try:    p.getConnectionInfo()
    except: break

p.disconnect()
print("[INFO] All done.")