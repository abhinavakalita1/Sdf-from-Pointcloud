"""
cspace_sdf_check.py
───────────────────
1. Evaluates SDF along a straight C-space line (6 samples, regula falsi on sign changes)
2. Triangulates C-space with a regular grid of resolution ~0.1 rad
3. Takes the first two RF zero-crossings on the line, finds their midpoint,
   and (if that midpoint is in collision) locates the triangle containing it.
4. From that seed triangle, grows TWO independent "red" frontiers outward —
   at each step, each frontier picks the unshaded neighbouring triangle with
   the most negative centroid SDF and shades it red.
5. Tracks the distance between the two frontier centroids over time. Once
   that distance stops growing meaningfully (oscillates / plateaus), or the
   2-minute time limit is hit, the expansion stops.
6. Displays only the final plot upon completion and prints total SDF query
   count and total processing time.
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
from matplotlib.collections import LineCollection
import os
import json
from collections import deque

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
# box2Id      = create_box(half_extents=[1,1,.2],      position=[0,0,3],      orientation=[0, 0, 0])
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

_sdf_cache = {}   # (q1_rounded, q2_rounded) → sdf
_sdf_query_count = 0  # total number of distinct SDF evaluations (cache misses)

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
# 7.  COLOUR PALETTE + PLOT SETUP
# ═══════════════════════════════════════════════════════

BLUE   = "#3A7DC9"; ORANGE = "#E8882A"; RED    = "#C93A3A"
GREEN  = "#2E9E5B"; PURPLE = "#7F3FBF"; GRAY   = "#888780"
LBLUE  = "#B5D4F4"; YELLOW = "#E8C22A"; TEAL   = "#1A9E8F"

plt.rcParams.update({
    "figure.facecolor":"white","axes.facecolor":"#F8F9FA",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.alpha":0.25,"grid.linestyle":"--",
    "font.size":11,"axes.titlesize":12,"axes.titleweight":"bold","axes.labelsize":11,
})

import math
START = np.array([.508, 1])
print("SDF at 50,0 is", eval_sdf(50/180 * math.pi, 0))
GOAL  = np.array([-.482, -1.443])
print("SDF at 100, 0 is", eval_sdf(100/180 * math.pi, 0))

N_SEGMENTS  = 5
N_SAMPLES   = N_SEGMENTS + 1
t_vals      = np.linspace(0.0, 1.0, N_SAMPLES)
sample_cfgs = np.array([START + t*(GOAL-START) for t in t_vals])

print(f"\n[INFO] Start: q1={np.degrees(START[0]):.1f}°  q2={np.degrees(START[1]):.1f}°")
print(f"[INFO] Goal : q1={np.degrees(GOAL[0]):.1f}°  q2={np.degrees(GOAL[1]):.1f}°")

_plot_samples          = []
_plot_rf_line          = []
_plot_tri_mesh         = None
_plot_seed_tris        = []
_plot_red_tris         = []   # list of triangle indices shaded red
_plot_midpoint         = None # (cfg, sdf) of the first-two-crossings midpoint

def _draw_base(ax):
    """Draw the static elements on ax (path line, start/goal)."""
    ax.plot(np.degrees([sample_cfgs[0][0], sample_cfgs[-1][0]]),
            np.degrees([sample_cfgs[0][1], sample_cfgs[-1][1]]),
            color=BLUE, lw=2.0, ls="-", zorder=2, label="C-space path")
    ax.scatter(*np.degrees(START), s=260, marker="*", color=GREEN,
               edgecolors="white", lw=1.5, zorder=9)
    ax.scatter(*np.degrees(GOAL),  s=260, marker="*", color=ORANGE,
               edgecolors="white", lw=1.5, zorder=9)
    ax.annotate("START", xy=np.degrees(START), xytext=(12,-18),
                textcoords="offset points", fontsize=9, fontweight="bold", color=GREEN)
    ax.annotate("GOAL",  xy=np.degrees(GOAL),  xytext=(12,-18),
                textcoords="offset points", fontsize=9, fontweight="bold", color=ORANGE)
    ax.set_xlim(-185,185); ax.set_ylim(-185,185)
    ax.set_xlabel("q₁ — Shoulder (degrees)")
    ax.set_ylabel("q₂ — Elbow (degrees)")
    ax.axhline(0, color=GRAY, lw=0.6, ls=":", alpha=0.5)
    ax.axvline(0, color=GRAY, lw=0.6, ls=":", alpha=0.5)

def build_and_show_final_plot(title):
    """Rebuilds and renders the comprehensive final plot summary."""
    fig, ax = plt.subplots(figsize=(9,9))
    _draw_base(ax)

    # ── Grid Triangulation Mesh ─────────────────────────────
    if _plot_tri_mesh is not None:
        vd, tris = _plot_tri_mesh
        for tri in tris:
            pts = vd[tri]
            triangle = plt.Polygon(pts, fill=False, edgecolor="#CCCCCC", lw=0.3, alpha=0.4, zorder=3)
            ax.add_patch(triangle)

    # ── Seed Triangle Highlight ─────────────────────────────
    if _plot_tri_mesh is not None and _plot_seed_tris:
        vd, tris = _plot_tri_mesh
        for tri_idx in _plot_seed_tris:
            tri = tris[tri_idx]
            pts = vd[tri]
            triangle = plt.Polygon(pts, closed=True, facecolor="#7F3FBF22", edgecolor=PURPLE, lw=1.8, zorder=5)
            ax.add_patch(triangle)

    # ── Red-Shaded Frontier Triangles ───────────────────────
    if _plot_tri_mesh is not None and _plot_red_tris:
        vd, tris = _plot_tri_mesh
        for tri_idx in _plot_red_tris:
            tri = tris[tri_idx]
            pts = vd[tri]
            triangle = plt.Polygon(pts, closed=True, facecolor="#C93A3A55", edgecolor=RED, lw=1.0, zorder=6)
            ax.add_patch(triangle)

    # ── Line Sample Dots ────────────────────────────────────
    for cfg, sdf_v, tv in _plot_samples:
        col = RED if sdf_v < 0 else GREEN
        ax.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=120, color=col, edgecolors="white", lw=1.2, zorder=7)
        ax.annotate(f"t={tv:.1f}\n{sdf_v:+.3f}m", xy=(np.degrees(cfg[0]), np.degrees(cfg[1])),
                    xytext=(10,8), textcoords="offset points", fontsize=7, color=col,
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7))

    # ── Line Scan Root Crossings ────────────────────────────
    for cfg, sdf_v in _plot_rf_line:
        ax.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=260, marker="X", color=PURPLE, edgecolors="white", lw=1.5, zorder=8)

    # ── Midpoint of First Two Crossings ─────────────────────
    if _plot_midpoint is not None:
        cfg_m, sdf_m = _plot_midpoint
        ax.scatter(np.degrees(cfg_m[0]), np.degrees(cfg_m[1]), s=220, marker="D",
                   color=YELLOW, edgecolors="black", lw=1.4, zorder=9)
        ax.annotate("midpoint", xy=(np.degrees(cfg_m[0]), np.degrees(cfg_m[1])),
                    xytext=(10,-12), textcoords="offset points", fontsize=8,
                    fontweight="bold", color="#806000")

    # ── Legend Construction ─────────────────────────────────
    handles = [
        plt.Line2D([0],[0], color=BLUE, lw=2, label="C-Space Path"),
        plt.scatter([],[], s=90,  color=GREEN,  ec="white", label="Free sample"),
        plt.scatter([],[], s=90,  color=RED,    ec="white", label="Coll sample"),
        plt.scatter([],[], s=140, marker="*",  color=GREEN,  ec="white", label="Start"),
        plt.scatter([],[], s=140, marker="*",  color=ORANGE, ec="white", label="Goal"),
        plt.scatter([],[], s=150, marker="X",  color=PURPLE, ec="white", label="RF zero (line)"),
        plt.scatter([],[], s=160, marker="D",  color=YELLOW, ec="black", label="Crossing midpoint"),
        plt.Polygon([[0,0]], closed=True, fc="#7F3FBF22", ec=PURPLE, lw=1.8, label="Seed triangle (contains midpoint)"),
        plt.Polygon([[0,0]], closed=True, fc="#C93A3A55", ec=RED, lw=1.0, label="Red-shaded frontier triangle"),
    ]
    ax.legend(handles=handles, frameon=True, fontsize=8, loc="upper right", framealpha=0.92)
    ax.set_title(title)
    plt.tight_layout()
    plt.show(block=True)
    plt.close(fig)

# ⏰ Start counting computational time before the first line draw action
start_processing_time = time.perf_counter()

# ═══════════════════════════════════════════════════════
# 8.  LINE SCAN (Background Computation Only)
# ═══════════════════════════════════════════════════════

print(f"\n[INFO] Evaluating SDF at {N_SAMPLES} samples …\n")
sdf_vals = []

for idx, cfg in enumerate(sample_cfgs):
    d = eval_sdf(cfg[0], cfg[1])
    # START (t=0) and GOAL (t=1) are always treated as free (sdf > 0) so that
    # regula falsi never places a root at the endpoints themselves.
    if idx == 0 or idx == N_SAMPLES - 1:
        d = abs(d) if d != 0 else 1e-6
    sdf_vals.append(d)
    print(f"  Sample {idx}  t={t_vals[idx]:.2f}"
          f"  q=({np.degrees(cfg[0]):+.2f}°,{np.degrees(cfg[1]):+.2f}°)"
          f"  SDF={d:+.5f} ({'COLL' if d<0 else 'free'})")
    update_gui_overlay(cfg[0], cfg[1], f"Sample {idx}  t={t_vals[idx]:.1f}", d)
    _plot_samples.append((cfg, d, t_vals[idx]))

sdf_vals = np.array(sdf_vals)

def regula_falsi(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b = cfg_a.copy(), cfg_b.copy()
    fa, fb = sdf_a, sdf_b
    for n in range(max_iter):
        cfg_c = a + fa*(a-b)/(fb-fa)
        fc    = eval_sdf(cfg_c[0], cfg_c[1])
        if abs(fc) < tol:
            return cfg_c, fc, n+1
        if fa*fc < 0: b, fb = cfg_c, fc
        else:          a, fa = cfg_c, fc
    return cfg_c, fc, max_iter

rf_line_roots = []
print(f"\n[INFO] Line-scan Regula Falsi …\n")
for i in range(N_SEGMENTS):
    si, sj = sdf_vals[i], sdf_vals[i+1]
    if si*sj < 0:
        cfg_r, sdf_r, nit = regula_falsi(sample_cfgs[i], sample_cfgs[i+1], si, sj)
        rf_line_roots.append((cfg_r, sdf_r, i, i+1))
        print(f"  RF root seg {i}→{i+1}:  q=({np.degrees(cfg_r[0]):+.3f}°,"
              f"{np.degrees(cfg_r[1]):+.3f}°)  SDF={sdf_r:+.6f}  ({nit} iters)")
        update_gui_overlay(cfg_r[0], cfg_r[1], f"RF zero  seg {i}→{i+1}", sdf_r)
        _plot_rf_line.append((cfg_r, sdf_r))

if not rf_line_roots:
    print("  [NOTE] No sign changes on the line — BFS will still run from nearest triangles.")

# ═══════════════════════════════════════════════════════
# 9.  TRIANGULATE C-SPACE (Side Length Resolution = 0.1 rad)
# ═══════════════════════════════════════════════════════

# Side length = 0.1 rad over a 2*pi span requires roughly 63 spans -> 64 nodes
GRID_N = int(np.ceil((2 * np.pi) / 0.1)) + 1
q_vals = np.linspace(-np.pi, np.pi, GRID_N)
QQ1, QQ2 = np.meshgrid(q_vals, q_vals)
vertices = np.column_stack([QQ1.ravel(), QQ2.ravel()])
N = GRID_N

triangles = []
for i in range(N-1):
    for j in range(N-1):
        tl = i*N + j
        tr = i*N + j+1
        bl = (i+1)*N + j
        br = (i+1)*N + j+1
        triangles.append([tl, tr, bl])
        triangles.append([tr, br, bl])

triangles = np.array(triangles)
M = len(triangles)
print(f"\n[INFO] Triangulation: Res=0.10 rad ({N}×{N} grid) → {len(vertices)} vertices, {M} triangles")

vertices_deg = np.degrees(vertices)
_plot_tri_mesh = (vertices_deg, triangles)

# Precompute triangle centroids (in radians) for fast lookup
tri_centroids = vertices[triangles].mean(axis=1)   # shape (M, 2)

# ═══════════════════════════════════════════════════════
# 10. BUILD ADJACENCY MAP
# ═══════════════════════════════════════════════════════

from collections import defaultdict
edge_to_tris = defaultdict(list)
for ti, tri in enumerate(triangles):
    for k in range(3):
        e = tuple(sorted([tri[k], tri[(k+1)%3]]))
        edge_to_tris[e].append(ti)

neighbours = [[] for _ in range(M)]
for e, tis in edge_to_tris.items():
    if len(tis) == 2:
        a, b = tis
        neighbours[a].append(b)
        neighbours[b].append(a)

# ── Toroidal seam stitching ─────────────────────────────────────────
# q1 = -π and q1 = +π represent the SAME physical configuration (the arm's
# shoulder joint wraps around), and likewise for q2. The grid above treats
# these as separate vertices/edges, so triangles on the left/right border
# (and top/bottom border) are currently "dead ends" with a missing neighbour.
# Here we stitch those borders together so expansion can wrap across the
# ±π seam exactly as it would physically.
#
# Cell (i, j) for i,j in [0, N-2] produces two triangles:
#   tri_lower = triangles[2*(i*(N-1)+j)]     -> [tl, tr, bl]
#   tri_upper = triangles[2*(i*(N-1)+j) + 1] -> [tr, br, bl]
#
# Left column  (j=0)        wraps to right column  (j=N-2)
# Top row      (i=0)        wraps to bottom row    (i=N-2)
#
# For a left-right wrap at row i: the left edge of cell (i,0) is the edge
# (tl,bl) of tri_lower(i,0); the right edge of cell (i,N-2) is the edge
# (tr,br) of tri_upper(i,N-2). These correspond to the same physical edge
# (q1 = -π == q1 = +π at the same q2 range), so we link the triangles that
# own those edges.
def cell_tris(i, j):
    base = 2 * (i * (N - 1) + j)
    return base, base + 1   # (tri_lower, tri_upper)

added_seam_links = 0

# Left (j=0) <-> Right (j=N-2): for each row i
for i in range(N - 1):
    left_lower, left_upper   = cell_tris(i, 0)
    right_lower, right_upper = cell_tris(i, N - 2)

    # Left edge of the cell belongs to tri_lower (edge tl-bl);
    # right edge of the cell belongs to tri_upper (edge tr-br).
    # Across the seam these two edges are the same physical edge.
    neighbours[left_lower].append(right_upper)
    neighbours[right_upper].append(left_lower)
    added_seam_links += 1

# Top (i=0) <-> Bottom (i=N-2): for each column j
for j in range(N - 1):
    top_lower, top_upper       = cell_tris(0, j)
    bottom_lower, bottom_upper = cell_tris(N - 2, j)

    # Top edge of the cell belongs to tri_lower (edge tl-tr);
    # bottom edge of the cell belongs to tri_upper (edge bl-br).
    # Across the seam these are the same physical edge.
    neighbours[top_lower].append(bottom_upper)
    neighbours[bottom_upper].append(top_lower)
    added_seam_links += 1

print(f"[INFO] Toroidal seam stitching: {added_seam_links} wraparound "
      f"adjacency links added (left↔right and top↔bottom borders).")

# ═══════════════════════════════════════════════════════
# 11. PAIRED CROSSINGS (1-2, 3-4, 5-6, ...) → SEED TRIANGLES
# ═══════════════════════════════════════════════════════

def point_in_triangle(pt, v0, v1, v2):
    d1 = (pt - v2)
    d2 = (v0 - v2)
    d3 = (v1 - v2)
    dot00 = d2 @ d2; dot01 = d2 @ d3
    dot02 = d2 @ d1; dot11 = d3 @ d3; dot12 = d3 @ d1
    inv = 1.0 / max(dot00*dot11 - dot01*dot01, 1e-15)
    u = (dot11*dot02 - dot01*dot12) * inv
    v = (dot00*dot12 - dot01*dot02) * inv
    return (u >= 0) and (v >= 0) and (u + v <= 1)

def find_containing_triangle(cfg_rad):
    for ti, tri in enumerate(triangles):
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        if point_in_triangle(cfg_rad, v0, v1, v2):
            return ti
    return int(np.argmin(np.linalg.norm(tri_centroids - cfg_rad, axis=1)))

TIME_LIMIT_S = 120.0

# Build list of (cfg_a, cfg_b) pairs: (root0,root1), (root2,root3), (root4,root5), ...
crossing_pairs = []
for k in range(0, len(rf_line_roots) - 1, 2):
    cfg_a, sdf_a_, ia, ib = rf_line_roots[k]
    cfg_b, sdf_b_, ja, jb = rf_line_roots[k+1]
    crossing_pairs.append((cfg_a, cfg_b, k, k+1))

seed_triangle_idxs = []
shaded = set()

if crossing_pairs:
    print(f"\n[INFO] {len(crossing_pairs)} crossing pair(s) found "
          f"(grouped as 1-2, 3-4, 5-6, ...)")

    for pair_num, (cfg_a, cfg_b, k, kp1) in enumerate(crossing_pairs):
        midpoint = (cfg_a + cfg_b) / 2.0
        midpoint_sdf = eval_sdf(midpoint[0], midpoint[1])
        _plot_midpoint = (midpoint, midpoint_sdf)

        print(f"\n[INFO] Pair {pair_num} (crossings {k+1} & {kp1+1}):")
        print(f"  Crossing {k+1}: q=({np.degrees(cfg_a[0]):+.3f}°,{np.degrees(cfg_a[1]):+.3f}°)")
        print(f"  Crossing {kp1+1}: q=({np.degrees(cfg_b[0]):+.3f}°,{np.degrees(cfg_b[1]):+.3f}°)")
        print(f"  Midpoint  : q=({np.degrees(midpoint[0]):+.3f}°,{np.degrees(midpoint[1]):+.3f}°)"
              f"  SDF={midpoint_sdf:+.6f}  ({'COLL' if midpoint_sdf < 0 else 'free'})")

        update_gui_overlay(midpoint[0], midpoint[1], f"Pair {pair_num} midpoint", midpoint_sdf)

        if midpoint_sdf < 0:
            seed_ti = find_containing_triangle(midpoint)
            print(f"  [INFO] Midpoint is in collision — seed triangle = {seed_ti}")
            seed_triangle_idxs.append(seed_ti)
        else:
            print("  [INFO] Midpoint is NOT in collision — skipping frontier expansion for this pair.")
else:
    print("\n[INFO] Fewer than two line-scan crossings found — skipping frontier expansion.")

_plot_seed_tris = list(seed_triangle_idxs)

# ═══════════════════════════════════════════════════════
# 12. DIJKSTRA EXPANSION FROM EACH PAIR'S SEED
# ═══════════════════════════════════════════════════════

import heapq

_stop_reason = "not started"

def centroid_sdf(ti):
    cx, cy = tri_centroids[ti]
    return eval_sdf(cx, cy)

def is_separated():
    """BFS from START tri through free (unshaded) triangles.
    Returns True if GOAL tri is unreachable → separator is complete."""
    start_ti = int(np.argmin(np.linalg.norm(tri_centroids - START, axis=1)))
    goal_ti  = int(np.argmin(np.linalg.norm(tri_centroids - GOAL,  axis=1)))
    visited = set()
    queue   = deque([start_ti])
    while queue:
        ti = queue.popleft()
        if ti in visited: continue
        if ti == goal_ti:  return False   # reached goal → not separated yet
        visited.add(ti)
        for nb in neighbours[ti]:
            if nb not in visited and nb not in shaded:
                queue.append(nb)
    return True   # goal unreachable → separated!

COST_THRESHOLD = -.8  # tweak this — more negative = stricter / thinner chain
CHECK_EVERY = 10      # run separation check every N triangles added

stop_reasons = []

if seed_triangle_idxs:
    for pair_idx, seed_triangle_idx in enumerate(seed_triangle_idxs):
        if seed_triangle_idx in shaded:
            stop_reasons.append(f"pair {pair_idx}: seed already shaded by earlier pair")
            continue

        seed_sdf = centroid_sdf(seed_triangle_idx)
        shaded.add(seed_triangle_idx)
        _plot_red_tris.append(seed_triangle_idx)
        print(f"\n[INFO] Pair {pair_idx} seed tri {seed_triangle_idx}  SDF={seed_sdf:+.5f}")

        heap = []
        for nb in neighbours[seed_triangle_idx]:
            if nb not in shaded:
                nb_sdf = centroid_sdf(nb)
                if nb_sdf < 0:
                    heapq.heappush(heap, (nb_sdf, nb))

        dead = set()
        step = 0
        pair_stop_reason = "heap exhausted (entire collision space explored)"

        print(f"  COST_THRESHOLD = {COST_THRESHOLD}")
        print(f"  {'step':>5}  {'tri':>5}  {'SDF':>10}  {'shaded':>7}  {'dead':>6}  separated")
        while heap:
            cost, ti = heapq.heappop(heap)

            if ti in shaded or ti in dead:
                continue

            sdf_val = centroid_sdf(ti)

            if sdf_val >= 0:
                dead.add(ti)
                continue

            if sdf_val > COST_THRESHOLD:
                dead.add(ti)
                continue

            shaded.add(ti)
            _plot_red_tris.append(ti)
            step += 1

            separated = False
            if step % CHECK_EVERY == 0:
                separated = is_separated()

            print(f"  {step:>5}  {ti:>5}  {sdf_val:>+10.5f}  {len(shaded):>7}  "
                  f"{len(dead):>6}  {'YES — DONE' if separated else ''}")

            if separated:
                pair_stop_reason = f"start/goal separated after {len(shaded)} total triangles"
                break

            for nb in neighbours[ti]:
                if nb not in shaded and nb not in dead:
                    nb_sdf = centroid_sdf(nb)
                    if nb_sdf < 0 and nb_sdf <= COST_THRESHOLD:
                        heapq.heappush(heap, (nb_sdf, nb))

        stop_reasons.append(f"pair {pair_idx}: {pair_stop_reason}")

        if "separated" in pair_stop_reason:
            break  # global separation achieved — no need to process further pairs

    _stop_reason = "; ".join(stop_reasons)
else:
    _stop_reason = "no seed triangles (no midpoints in collision, or <2 crossings)"

# 🛑 Stop counting computation time as the processing engine completes calculation loops
end_processing_time = time.perf_counter()
total_computation_time = end_processing_time - start_processing_time

print(f"\n[INFO] Expansion finished.")
print(f"  Stop reason            : {_stop_reason}")
print(f"  Triangles shaded red   : {len(shaded)}")
print(f"\n⏱️  [TIME REPORT] Computational Engine Loop Time : {total_computation_time:.4f} seconds.")
print(f"📊 [SDF REPORT]  Total distinct SDF queries      : {_sdf_query_count}")

# ═══════════════════════════════════════════════════════
# 13. SAVE TO DISK + FINAL SUMMARY PLOT
# ═══════════════════════════════════════════════════════

fig_f, ax_f = plt.subplots(figsize=(9,9))
_draw_base(ax_f)

if _plot_tri_mesh is not None:
    vd, tris = _plot_tri_mesh
    for tri in tris:
        pts = vd[tri]
        ax_f.add_patch(plt.Polygon(pts, fill=False, edgecolor="#CCCCCC", lw=0.3, alpha=0.4, zorder=3))

    for tri_idx in _plot_seed_tris:
        tri = tris[tri_idx]
        pts = vd[tri]
        ax_f.add_patch(plt.Polygon(pts, closed=True, facecolor="#7F3FBF22", edgecolor=PURPLE, lw=1.8, zorder=5))

    for tri_idx in _plot_red_tris:
        tri = tris[tri_idx]
        pts = vd[tri]
        ax_f.add_patch(plt.Polygon(pts, closed=True, facecolor="#C93A3A55", edgecolor=RED, lw=1.0, zorder=6))

for cfg, sdf_v, tv in _plot_samples:
    col = RED if sdf_v < 0 else GREEN
    ax_f.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=120, color=col, ec="white", lw=1.2, zorder=7)

for cfg, sdf_v in _plot_rf_line:
    ax_f.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=260, marker="X", color=PURPLE, ec="white", lw=1.5, zorder=8)

if _plot_midpoint is not None:
    cfg_m, sdf_m = _plot_midpoint
    ax_f.scatter(np.degrees(cfg_m[0]), np.degrees(cfg_m[1]), s=220, marker="D",
                  color=YELLOW, edgecolors="black", lw=1.4, zorder=9)

ax_f.set_title(f"C-Space Dijkstra Expansion | {len(shaded)} triangles\n"
                f"{_stop_reason} | Time: {total_computation_time:.2f}s | "
                f"SDF queries: {_sdf_query_count}")
fig_f.tight_layout()
fig_f.savefig("cspace_bfs_final.png", dpi=150, bbox_inches="tight")
plt.close(fig_f)
print("[INFO] Saved cspace_bfs_final.png")

# Trigger final plot window display
build_and_show_final_plot(
    f"C-Space Dijkstra Expansion\n"
    f"{len(shaded)} triangles | {_stop_reason}\n"
    f"Time: {total_computation_time:.3f}s | SDF queries: {_sdf_query_count}"
)

# ═══════════════════════════════════════════════════════
# 14. KEEP PYBULLET GUI ALIVE
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