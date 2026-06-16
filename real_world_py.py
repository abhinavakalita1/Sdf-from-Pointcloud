"""
cspace_sdf_check.py
───────────────────
1. Evaluates SDF along a straight C-space line (6 samples, regula falsi on sign changes)
2. Triangulates C-space with a regular grid of resolution ~0.1 rad
3. Finds seed triangles containing the RF zero-crossings
4. BFS flood-fill: for each triangle, check edges for SDF sign changes,
   run Regula Falsi on those edges, propagate to neighbours ONLY if
   at least one sign-change edge was found in the current triangle.
5. Connects interior zero-crossings to dynamically construct and display
   the continuous C-space obstacle boundary.
6. Displays only the final plot upon completion and prints total processing time.
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
    key = (round(float(q1), 6), round(float(q2), 6))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2)
    d = sdf_scene(arm2Id)
    _sdf_cache[key] = d
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
# x = 0
# while eval_sdf(.508, x)<=0:
#     x+=.1

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
_plot_tri_active       = []
_plot_rf_bfs           = []
_plot_boundary_segments = []  # Stores pairs of points: [((x1, y1), (x2, y2)), ...] in degrees

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
    """Rebuilds and renders the comprehensive final plot summary with boundaries."""
    fig, ax = plt.subplots(figsize=(9,9))
    _draw_base(ax)

    # ── Grid Triangulation Mesh ─────────────────────────────
    if _plot_tri_mesh is not None:
        vd, tris = _plot_tri_mesh
        for tri in tris:
            pts = vd[tri]
            triangle = plt.Polygon(pts, fill=False, edgecolor="#CCCCCC", lw=0.3, alpha=0.4, zorder=3)
            ax.add_patch(triangle)

    # ── Visited Triangles (color coded) ─────────────────────
    if _plot_tri_mesh is not None and _plot_tri_active:
        vd, tris = _plot_tri_mesh
        for tri_idx, has_sc in _plot_tri_active:
            tri = tris[tri_idx]
            pts = vd[tri]
            fc  = "#FFD70012" if has_sc else "#AAAAAA08"
            ec  = "#E8C22A" if has_sc else "#888780"
            lw  = 0.8  if has_sc else 0.4
            triangle = plt.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw, alpha=0.3, zorder=4)
            ax.add_patch(triangle)

    # ── Seed Triangles Highlight ────────────────────────────
    if _plot_tri_mesh is not None and _plot_seed_tris:
        vd, tris = _plot_tri_mesh
        for tri_idx in _plot_seed_tris:
            tri = tris[tri_idx]
            pts = vd[tri]
            triangle = plt.Polygon(pts, closed=True, facecolor="#7F3FBF15", edgecolor=PURPLE, lw=1.2, zorder=5)
            ax.add_patch(triangle)

    # ── Continuous Obstacle Boundary Construction ───────────
    if _plot_boundary_segments:
        lc = LineCollection(_plot_boundary_segments, colors="#E63946", linewidths=2.5, linestyle="-", zorder=6)
        ax.add_collection(lc)

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

    # ── Flood fill Zero Points ──────────────────────────────
    for cfg, sdf_v in _plot_rf_bfs:
        ax.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=60, marker="o", color=TEAL, edgecolors="white", lw=0.6, zorder=8)

    # ── Legend Construction ─────────────────────────────────
    handles = [
        plt.Line2D([0],[0], color=BLUE, lw=2, label="C-Space Path"),
        plt.Line2D([0],[0], color="#E63946", lw=2.5, label="Extracted Obstacle Boundary (SDF=0)"),
        plt.scatter([],[], s=90,  color=GREEN,  ec="white", label="Free sample"),
        plt.scatter([],[], s=90,  color=RED,    ec="white", label="Coll sample"),
        plt.scatter([],[], s=140, marker="*",  color=GREEN,  ec="white", label="Start"),
        plt.scatter([],[], s=140, marker="*",  color=ORANGE, ec="white", label="Goal"),
        plt.scatter([],[], s=150, marker="X",  color=PURPLE, ec="white", label="RF zero (line)"),
        plt.scatter([],[], s=60,  marker="o",  color=TEAL,   ec="white", label="RF intersection vertices"),
        plt.Polygon([[0,0]], closed=True, fc="#FFD70030", ec=YELLOW, lw=1.2, label="Active boundary triangle"),
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

# ═══════════════════════════════════════════════════════
# 11. FIND SEED TRIANGLES
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
    centroids = vertices[triangles].mean(axis=1)
    return int(np.argmin(np.linalg.norm(centroids - cfg_rad, axis=1)))

seed_triangle_indices = []
if rf_line_roots:
    for cfg_r, sdf_r, ia, ib in rf_line_roots:
        ti = find_containing_triangle(cfg_r)
        if ti not in seed_triangle_indices:
            seed_triangle_indices.append(ti)
    print(f"\n[INFO] Seed triangles: {seed_triangle_indices}")
else:
    mid = (START + GOAL) / 2
    ti  = find_containing_triangle(mid)
    seed_triangle_indices = [ti]
    print(f"\n[INFO] No line RF roots — seeding from midpoint triangle {ti}")

_plot_seed_tris = seed_triangle_indices.copy()

# ═══════════════════════════════════════════════════════
# 12. SDF VERTEX CACHING
# ═══════════════════════════════════════════════════════

vertex_sdf = {}

def get_vertex_sdf(vi):
    if vi not in vertex_sdf:
        q1, q2 = vertices[vi]
        vertex_sdf[vi] = eval_sdf(q1, q2)
    return vertex_sdf[vi]

# ═══════════════════════════════════════════════════════
# 13. BFS FLOOD-FILL WITH LOCAL ISOSURFACE TRACING
# ═══════════════════════════════════════════════════════

visited     = set()
queue       = deque(seed_triangle_indices)
visited.update(seed_triangle_indices)
bfs_rf_pts  = []

print(f"\n[INFO] BFS flood-fill starting from {len(seed_triangle_indices)} seed(s) …\n")
bfs_step = 0

while queue:
    ti = queue.popleft()
    tri = triangles[ti]
    corner_sdfs = [get_vertex_sdf(vi) for vi in tri]

    local_triangle_roots = []
    has_sign_change = False

    for k in range(3):
        vi_a = tri[k]
        vi_b = tri[(k+1) % 3]
        sa   = corner_sdfs[k]
        sb   = corner_sdfs[(k+1) % 3]

        if sa * sb < 0:
            has_sign_change = True
            cfg_a = vertices[vi_a]
            cfg_b = vertices[vi_b]
            cfg_r, sdf_r, nit = regula_falsi(cfg_a, cfg_b, sa, sb)

            bfs_rf_pts.append((cfg_r, sdf_r))
            _plot_rf_bfs.append((cfg_r, sdf_r))
            local_triangle_roots.append(np.degrees(cfg_r))

            # Optional print commented out to preserve buffer memory at 0.1 resolution step
            # print(f"  BFS tri {ti} edge ({vi_a},{vi_b}): RF root found.")
            update_gui_overlay(cfg_r[0], cfg_r[1], f"BFS RF  tri {ti}", sdf_r)

    if len(local_triangle_roots) == 2:
        _plot_boundary_segments.append((local_triangle_roots[0], local_triangle_roots[1]))
    elif len(local_triangle_roots) == 3:
        _plot_boundary_segments.append((local_triangle_roots[0], local_triangle_roots[1]))
        _plot_boundary_segments.append((local_triangle_roots[1], local_triangle_roots[2]))
        _plot_boundary_segments.append((local_triangle_roots[2], local_triangle_roots[0]))

    _plot_tri_active.append((ti, has_sign_change))
    bfs_step += 1

    if has_sign_change:
        for nb in neighbours[ti]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)

print(f"\n[INFO] BFS complete.")
print(f"  Triangles visited : {len(visited)}")
print(f"  BFS RF zero pts   : {len(bfs_rf_pts)}")

# 🛑 Stop counting computation time as the processing engine completes calculation loops
end_processing_time = time.perf_counter()
total_computation_time = end_processing_time - start_processing_time

print(f"\n⏱️  [TIME REPORT] Computational Engine Loop Time: {total_computation_time:.4f} seconds.")

# ═══════════════════════════════════════════════════════
# 14. SAVE TO DISK + FINAL SUMMARY PLOT
# ═══════════════════════════════════════════════════════

fig_f, ax_f = plt.subplots(figsize=(9,9))
_draw_base(ax_f)

if _plot_tri_mesh is not None:
    vd, tris = _plot_tri_mesh
    for tri in tris:
        pts = vd[tri]
        ax_f.add_patch(plt.Polygon(pts, fill=False, edgecolor="#CCCCCC", lw=0.4, zorder=3))

    for tri_idx, has_sc in _plot_tri_active:
        tri = tris[tri_idx]
        pts = vd[tri]
        fc  = "#FFD70020" if has_sc else "#AAAAAA18"
        ec  = YELLOW if has_sc else GRAY
        ax_f.add_patch(plt.Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=1.2, zorder=4))

    for tri_idx in _plot_seed_tris:
        tri = tris[tri_idx]
        pts = vd[tri]
        ax_f.add_patch(plt.Polygon(pts, closed=True, facecolor="#7F3FBF22", edgecolor=PURPLE, lw=1.8, zorder=5))

if _plot_boundary_segments:
    lc_f = LineCollection(_plot_boundary_segments, colors="#E63946", linewidths=2.5, zorder=6)
    ax_f.add_collection(lc_f)

for cfg, sdf_v, tv in _plot_samples:
    col = RED if sdf_v < 0 else GREEN
    ax_f.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=120, color=col, ec="white", lw=1.2, zorder=7)

for cfg, sdf_v in _plot_rf_line:
    ax_f.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=260, marker="X", color=PURPLE, ec="white", lw=1.5, zorder=8)

for cfg, sdf_v in _plot_rf_bfs:
    ax_f.scatter(np.degrees(cfg[0]), np.degrees(cfg[1]), s=60, marker="o", color=TEAL, ec="white", lw=0.6, zorder=8)

ax_f.set_title(f"C-Space BFS — Grid Res 0.1 rad | Compute Time: {total_computation_time:.2f}s")
fig_f.tight_layout()
fig_f.savefig("cspace_bfs_final.png", dpi=150, bbox_inches="tight")
plt.close(fig_f)
print("[INFO] Saved cspace_bfs_final.png")

# Trigger final plot window display including the reconstructed continuous boundary path
build_and_show_final_plot(
    f"C-Space Summary — Resolution Side Length: 0.1 rad\n"
    f"Total Algorithm Execution Duration: {total_computation_time:.3f} seconds"
)

# ═══════════════════════════════════════════════════════
# 15. KEEP PYBULLET GUI ALIVE
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