"""
cspace_sdf_check.py
───────────────────
Triangulates the entire C-space using a uniform grid with a specified triangle
side length (~0.1 rad). Evaluates the exact SDF value at every corner vertex.
Shades triangles based on corner validation:
  - All corners positive (SDF >= 0) -> Green (Free Space)
  - All corners negative (SDF < 0)  -> Red (Collision Space)
  - Mixed corner signs               -> Black (Boundary Intersection)
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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import json

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
box2Id      = create_box(half_extents=[1,1,.2],      position=[0,0,3],      orientation=[0, 0, 0])
obstacle_ids   = [boxId, sphereId, cylinderId, box2Id]

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
    small_clusters = [lbl for lbl,cnt in cluster_sizes.items() if lbl != -1 and cnt < min_points_threshold]
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
# 5.  SDF PERFORMANCE ENGINE
# ═══════════════════════════════════════════════════════

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
    key = (round(float(q1), 6), round(float(q2), 6))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2)
    d = sdf_scene(arm2Id)
    _sdf_cache[key] = d
    return d

# ═══════════════════════════════════════════════════════
# 6. UNIFORM C-SPACE TRIANGULATION
# ═══════════════════════════════════════════════════════

TARGET_SIDE_LENGTH = 0.1 # Radian step limit parameters
GRID_N = int(np.ceil((2 * np.pi) / TARGET_SIDE_LENGTH)) + 1

q_vals = np.linspace(-np.pi, np.pi, GRID_N)
QQ1, QQ2 = np.meshgrid(q_vals, q_vals)
vertices = np.column_stack([QQ1.ravel(), QQ2.ravel()])
N = GRID_N

triangles = []
for i in range(N-1):
    for j in range(N-1):
        tl = i * N + j
        tr = i * N + j + 1
        bl = (i + 1) * N + j
        br = (i + 1) * N + j + 1
        triangles.append([tl, tr, bl])
        triangles.append([tr, br, bl])

triangles = np.array(triangles)
M = len(triangles)
print(f"\n[INFO] Complete Triangulation Map: {N}x{N} nodes -> {len(vertices)} vertices, {M} triangles generated.")

# ═══════════════════════════════════════════════════════
# 7. VERTEX SDF EVALUATION & CACHING
# ═══════════════════════════════════════════════════════

print(f"[INFO] Computing precise collision space data across all {len(vertices)} coordinates...")
vertex_sdf = np.zeros(len(vertices))

t_start = time.time()
for vi, (q1, q2) in enumerate(vertices):
    vertex_sdf[vi] = eval_sdf(q1, q2)
    if (vi + 1) % 500 == 0 or (vi + 1) == len(vertices):
        print(f"  Processed {vi + 1}/{len(vertices)} nodes...")
print(f"[INFO] SDF Field generated in {time.time() - t_start:.2f} seconds.")

# ═══════════════════════════════════════════════════════
# 8. TRIANGLE CATEGORIZATION AND RENDERING
# ═══════════════════════════════════════════════════════

print(f"[INFO] Categorizing and sorting triangle geometries...")
free_patches = []
collision_patches = []
boundary_patches = []

# Conversion scale vector for mapping degrees onto matplotlib boundaries
vertices_deg = np.degrees(vertices)

for tri in triangles:
    pts_deg = vertices_deg[tri]
    sdfs = vertex_sdf[tri]

    if np.all(sdfs >= 0):
        free_patches.append(Polygon(pts_deg, closed=True))
    elif np.all(sdfs < 0):
        collision_patches.append(Polygon(pts_deg, closed=True))
    else:
        boundary_patches.append(Polygon(pts_deg, closed=True))

# Create highly optimized vector collections for immediate execution plots
free_collection = PatchCollection(free_patches, facecolor='#2E9E5B', edgecolor='none', alpha=0.85)
collision_collection = PatchCollection(collision_patches, facecolor='#C93A3A', edgecolor='none', alpha=0.85)
boundary_collection = PatchCollection(boundary_patches, facecolor='#111111', edgecolor='none', alpha=0.95)

# Set Up Plot Environment Figures
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor("#FAFAFA")
ax.set_xlim(-180, 180)
ax.set_ylim(-180, 180)
ax.set_xlabel("q₁ — Shoulder (degrees)", fontsize=11, fontweight='bold')
ax.set_ylabel("q₂ — Elbow (degrees)", fontsize=11, fontweight='bold')
ax.axhline(0, color='#888780', lw=0.8, ls=":", alpha=0.6)
ax.axvline(0, color='#888780', lw=0.8, ls=":", alpha=0.6)

# Render complete system profiles
ax.add_collection(free_collection)
ax.add_collection(collision_collection)
ax.add_collection(boundary_collection)

# Legend Layout Mapping
legend_elements = [
    plt.Polygon([[0,0]], closed=True, color='#2E9E5B', label="Free Space (All Corners SDF >= 0)"),
    plt.Polygon([[0,0]], closed=True, color='#C93A3A', label="Collision Space (All Corners SDF < 0)"),
    plt.Polygon([[0,0]], closed=True, color='#111111', label="Boundary Transitions (Mixed Signs)")
]
ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CCCCCC')
ax.set_title(f"Full 2-DOF Configuration Space Map\nGrid Resolution: Δq ~ {TARGET_SIDE_LENGTH} rad ({N}x{N} nodes, {M} triangles)", fontsize=12, fontweight='bold')

plt.tight_layout()
fig.savefig("cspace_full_triangulation.png", dpi=150, bbox_inches="tight")
print("[INFO] Saved execution summary image: cspace_full_triangulation.png")
plt.show(block=True)

# ═══════════════════════════════════════════════════════
# 9. PYBULLET PERSISTENCE ENGINE
# ═══════════════════════════════════════════════════════

p.addUserDebugText("Done — close visualizer window to exit", [0,0,2.5], textColorRGB=[0.8,0.8,0.1], textSize=1.6)
print("\n[INFO] Matplotlib exited. Keeping PyBullet GUI alive. Exit simulation frame to terminate completely.")
while True:
    p.stepSimulation()
    time.sleep(1/60)
    try:    p.getConnectionInfo()
    except: break

p.disconnect()
print("[INFO] Run finished successfully.")