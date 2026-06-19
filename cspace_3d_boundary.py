"""
cspace_3d_boundary.py
─────────────────────
BFS boundary tracing in 3D C-space.
Uses parity-consistent 5-tet cube decomposition so the tet mesh is
fully connected (1 component) and BFS from 2 RF-zero seeds reaches
every boundary tet in the entire space.

Colors:
  BLACK — boundary tets (mixed SDF sign)
  RED   — interior tets (all corners negative)
  GREEN — free tets     (all corners positive)
"""

import pybullet_data
import numpy as np
import pybullet as p
import time
from collections import deque, defaultdict
import pyvista as pv

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET  (headless — no GUI)
# ══════════════════════════════════════════════════════════════════

p.connect(p.DIRECT)                          # ← headless, no GUI window
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0,0,0],
    baseOrientation=p.getQuaternionFromEuler([0,0,0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
print(f"[INFO] Arm loaded — {p.getNumJoints(arm3Id)} joints")

# ══════════════════════════════════════════════════════════════════
# 2.  OBSTACLES
# ══════════════════════════════════════════════════════════════════

def create_box(half_extents=[1,1,1], position=[0,0,0], orientation=[0,0,0], mass=0, color=[1,0,0,1]):
    col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(mass, col, vis, position, quat)

def create_sphere(radius=0.5, position=[0,0,0], orientation=[0,0,0], mass=0, color=[0,1,0,1]):
    col  = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis  = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(mass, col, vis, position, quat)

def create_cylinder(radius=0.5, height=1.0, position=[0,0,0], orientation=[0,0,0], mass=0, color=[0,0,1,1]):
    col  = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis  = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=color)
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(mass, col, vis, position, quat)

boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],      orientation=[0.2,1.1,0.4])
sphereId   = create_sphere(radius=1,               position=[0,2,1])
cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1],   orientation=[1.3,0,0])
box2Id      = create_box(half_extents=[1,1,.2],      position=[0,0,3],      orientation=[0, 0, 0])
ALL_OBSTACLE_IDS   = [boxId, sphereId, cylinderId, box2Id]
obstacle_names = ["box", "sphere", "cylinder", "box2"]

# ══════════════════════════════════════════════════════════════════
# 3.  SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache       = {}
_sdf_query_count = 0          # ← NEW: tracks cache-miss (actual) queries

def set_config(q1, q2, q3):
    p.resetJointState(arm3Id, 0, float(q1))
    p.resetJointState(arm3Id, 1, float(q2))
    p.resetJointState(arm3Id, 2, float(q3))
    p.stepSimulation()

def sdf_scene(threshold=10.0):
    min_d = threshold
    nj = p.getNumJoints(arm3Id)
    for obs_id in ALL_OBSTACLE_IDS:
        for link_idx in range(-1, nj):
            contacts = p.getClosestPoints(
                bodyA=arm3Id, bodyB=obs_id,
                distance=threshold, linkIndexA=link_idx)
            if contacts:
                d = min(c[8] for c in contacts)
                if d < min_d:
                    min_d = d
    return min_d

def eval_sdf_vec(cfg):
    global _sdf_query_count
    key = (round(float(cfg[0]),4), round(float(cfg[1]),4), round(float(cfg[2]),4))
    if key not in _sdf_cache:
        set_config(*cfg)
        _sdf_cache[key] = sdf_scene()
        _sdf_query_count += 1          # ← count only cache-miss calls
    return _sdf_cache[key]

# ══════════════════════════════════════════════════════════════════
# 4.  GRID + PARITY-CONSISTENT 5-TET DECOMPOSITION
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.35
q_vals = np.arange(-np.pi, np.pi + GRID_STEP*0.5, GRID_STEP)
N = len(q_vals)
print(f"[INFO] Grid {N}^3  step={GRID_STEP:.2f} rad")

def vidx(i,j,k): return i*N*N + j*N + k

vertices = np.array([[q_vals[i], q_vals[j], q_vals[k]]
                     for i in range(N) for j in range(N) for k in range(N)])

def cube_tets(i, j, k):
    v = [vidx(i+di, j+dj, k+dk) for di,dj,dk in [
        (0,0,0),(1,0,0),(0,1,0),(1,1,0),
        (0,0,1),(1,0,1),(0,1,1),(1,1,1)
    ]]
    if (i+j+k) % 2 == 0:
        return [
            [v[0],v[1],v[2],v[4]],
            [v[1],v[2],v[3],v[7]],
            [v[1],v[4],v[5],v[7]],
            [v[2],v[4],v[6],v[7]],
            [v[1],v[2],v[4],v[7]],
        ]
    else:
        return [
            [v[0],v[1],v[3],v[5]],
            [v[0],v[2],v[3],v[6]],
            [v[0],v[4],v[5],v[6]],
            [v[3],v[5],v[6],v[7]],
            [v[0],v[3],v[5],v[6]],
        ]

tetrahedra = np.array([
    tet
    for i in range(N-1) for j in range(N-1) for k in range(N-1)
    for tet in cube_tets(i,j,k)
], dtype=np.int32)

M = len(tetrahedra)
tet_centroids = vertices[tetrahedra].mean(axis=1)
print(f"[INFO] {len(vertices):,} vertices | {M:,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# 5.  ADJACENCY
# ══════════════════════════════════════════════════════════════════

print("[INFO] Building adjacency ...")
face_to_tets = defaultdict(list)
for ti, tet in enumerate(tetrahedra):
    for fi in range(4):
        face = tuple(sorted(tet[j] for j in range(4) if j != fi))
        face_to_tets[face].append(ti)

neighbours = [[] for _ in range(M)]
for face, tis in face_to_tets.items():
    if len(tis) == 2:
        a, b = tis
        neighbours[a].append(b)
        neighbours[b].append(a)
print("[INFO] Adjacency done.")

# ══════════════════════════════════════════════════════════════════
# 6.  VERTEX SDF (lazy)
# ══════════════════════════════════════════════════════════════════

vertex_sdf = {}

def get_vertex_sdf(vi):
    if vi not in vertex_sdf:
        vertex_sdf[vi] = eval_sdf_vec(vertices[vi])
    return vertex_sdf[vi]

# ══════════════════════════════════════════════════════════════════
# 7.  PATH + RF ZEROS
# ══════════════════════════════════════════════════════════════════

START = np.array([1.600, -1.902, 0.892])
GOAL  = np.array([-1.400, 1.902, -0.654])
print(eval_sdf_vec(START))
print(eval_sdf_vec(GOAL))


line_len  = np.linalg.norm(GOAL - START)
N_SEG     = max(5, int(np.ceil(line_len / 0.1)))
t_vals    = np.linspace(0, 1, N_SEG+1)
line_cfgs = np.array([START + t*(GOAL-START) for t in t_vals])
line_sdfs = np.array([eval_sdf_vec(c) for c in line_cfgs])

def regula_falsi(a, b, fa, fb, tol=1e-4, max_iter=50):
    for _ in range(max_iter):
        c  = a + fa*(a-b)/(fb-fa)
        fc = eval_sdf_vec(c)
        if abs(fc) < tol: return c
        if fa*fc < 0: b, fb = c, fc
        else:          a, fa = c, fc
    return c

rf_zeros = []
for i in range(N_SEG):
    if line_sdfs[i]*line_sdfs[i+1] < 0:
        z = regula_falsi(line_cfgs[i], line_cfgs[i+1], line_sdfs[i], line_sdfs[i+1])
        rf_zeros.append(z)
        print(f"[RF] crossing at ({np.degrees(z[0]):+.2f}, {np.degrees(z[1]):+.2f}, {np.degrees(z[2]):+.2f}) deg")

print(f"[INFO] {len(rf_zeros)} RF zero crossing(s).")

def find_closest_tet(cfg):
    return int(np.argmin(np.linalg.norm(tet_centroids - cfg, axis=1)))

seed_tets = []
for z in rf_zeros:
    ti = find_closest_tet(z)
    if ti not in seed_tets:
        seed_tets.append(ti)
if not seed_tets:
    seed_tets = [find_closest_tet((START+GOAL)/2)]
    print("[WARN] No RF zeros — seeding from midpoint")
print(f"[INFO] Seed tets: {seed_tets}")

# ══════════════════════════════════════════════════════════════════
# 8.  BFS — all neighbours expand, color by sign
# ══════════════════════════════════════════════════════════════════

t0 = time.perf_counter()
visited       = set(seed_tets)
queue         = deque(seed_tets)
boundary_tets = []

step = 0
print("[INFO] BFS expanding (boundary only) ...")
while queue:
    ti   = queue.popleft()
    tet  = tetrahedra[ti]
    sdfs = [get_vertex_sdf(vi) for vi in tet]

    any_pos = any(s > 0 for s in sdfs)
    any_neg = any(s < 0 for s in sdfs)

    if any_pos and any_neg:
        # BLACK — boundary tet: record it and expand neighbours
        boundary_tets.append(ti)
        for nb in neighbours[ti]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    # RED (all neg) or GREEN (all pos) — do not record, do not expand

    step += 1
    if step % 5000 == 0:
        print(f"  step {step:6d}  visited {len(visited):6d}  black {len(boundary_tets):5d}")

bfs_time = time.perf_counter() - t0

print(f"[INFO] BFS done in {bfs_time:.4f}s")
print(f"  Boundary BLACK : {len(boundary_tets):,}")
print(f"  SDF queries    : {_sdf_query_count}  (cache misses only)")

# ══════════════════════════════════════════════════════════════════
# 9.  BUILD MESHES
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(tet_indices):
    if not tet_indices:
        return None
    sel      = tetrahedra[np.asarray(tet_indices)]
    uvi, inv = np.unique(sel, return_inverse=True)
    lv       = np.degrees(vertices[uvi])
    lt       = inv.reshape(-1, 4)
    n        = len(lt)
    cells    = np.hstack([np.full((n,1), 4, dtype=np.int64), lt]).ravel()
    ct       = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, ct, lv.astype(np.float64))

print("[INFO] Building meshes ...")
boundary_mesh = build_tet_mesh(boundary_tets)

# ══════════════════════════════════════════════════════════════════
# 10.  PLOT
# ══════════════════════════════════════════════════════════════════

line_pts_deg = np.degrees(line_cfgs)

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("white")

if boundary_mesh:
    pl.add_mesh(boundary_mesh, color="black", opacity=0.85,
                show_edges=True, edge_color="#333333", line_width=0.4,
                label=f"Boundary BLACK ({len(boundary_tets):,})")

path_cloud = pv.PolyData(line_pts_deg)
path_cloud["SDF"] = line_sdfs
pl.add_mesh(path_cloud, scalars="SDF", cmap="viridis",
            point_size=13, render_points_as_spheres=True,
            scalar_bar_args={"title": "SDF along path"})
spline = pv.Spline(line_pts_deg, 300)
pl.add_mesh(spline, color="#1F618D", line_width=2.5, label="Path")

pl.add_points(np.degrees(START).reshape(1,3), color="lime",
              point_size=22, render_points_as_spheres=True, label="START")
pl.add_points(np.degrees(GOAL).reshape(1,3), color="orange",
              point_size=22, render_points_as_spheres=True, label="GOAL")

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_legend(bcolor="white", border=True, size=(0.28, 0.20))
pl.add_title(
    f"3D C-Space BFS | Boundary BLACK={len(boundary_tets):,} | "
    f"queries={_sdf_query_count} | time={bfs_time:.2f}s",
    font_size=10)

print("[INFO] Opening window ...")
pl.show()
p.disconnect()