"""
cspace_3d_dijkstra_pyvista.py
─────────────────────────────
3-DOF arm, 3D C-space (q1, q2, q3).

Algorithm:
  1. Sample SDF along START→GOAL line every ~0.1 rad.
  2. Regula Falsi on every sign-change segment → zero crossings.
  3. Pair consecutive zeros (1-2, 3-4, …), find midpoints (SDF < 0).
  4. Find enclosing tet for each midpoint → seed tets.
  5. Dijkstra: expand to neighbour with most-negative centroid SDF.
     Skip tets with SDF > COST_THRESHOLD. Stop when heap exhausted.

Tet decomposition: parity-consistent 5-tet per cube (fully connected mesh).
"""

import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time, os, json, heapq
from collections import deque, defaultdict
import pyvista as pv

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP  (headless — no GUI)
# ══════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS = p.getNumJoints(arm3Id)
print(f"[INFO] 3-DOF arm loaded — {NUM_JOINTS} joints")

# ══════════════════════════════════════════════════════════════════
# 2.  OBSTACLES
# ══════════════════════════════════════════════════════════════════

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

boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],    orientation=[0.2,1.1,0.4])
sphereId   = create_sphere(radius=1,               position=[0,2,1])
cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1], orientation=[1.3,0,0])
box2Id     = create_box(half_extents=[1,1,.2],     position=[0,0,3],    orientation=[0,0,0])
obstacle_ids   = [boxId, sphereId, cylinderId, box2Id]
obstacle_names = ["box", "sphere", "cylinder", "box2"]

# ══════════════════════════════════════════════════════════════════
# 3.  POINTCLOUD
# ══════════════════════════════════════════════════════════════════

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
            dist=7.0
            fx=dist*np.cos(elev_r)*np.cos(az)
            fy=dist*np.cos(elev_r)*np.sin(az)
            fz=dist*np.sin(elev_r)
            r=p.rayTest([fx,fy,fz],[-fx,-fy,-fz])
            if r[0] in body_ids: pts.append(r[3])
    pts = np.array(pts)
    np.save("points.npy", pts)
    print(f"[INFO] Generated {len(pts)} pts")
    return pts

points = sample_pointcloud(obstacle_ids)

# ══════════════════════════════════════════════════════════════════
# 4.  CLUSTERING + HULL BODIES
# ══════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════
# 5.  SDF
# ══════════════════════════════════════════════════════════════════

_sdf_cache       = {}
_sdf_query_count = 0

def sdf_scene(arm_id, threshold=10.0):
    min_d = threshold
    for hull_id in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=hull_id, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d: min_d = d
    return min_d

def set_config(q1, q2, q3):
    p.resetJointState(arm3Id, 0, float(q1))
    p.resetJointState(arm3Id, 1, float(q2))
    p.resetJointState(arm3Id, 2, float(q3))
    p.stepSimulation()

def eval_sdf(q1, q2, q3):
    global _sdf_query_count
    key = (round(float(q1),5), round(float(q2),5), round(float(q3),5))
    if key in _sdf_cache:
        return _sdf_cache[key]
    set_config(q1, q2, q3)
    d = sdf_scene(arm3Id)
    _sdf_cache[key] = d
    _sdf_query_count += 1
    return d

def eval_sdf_vec(cfg):
    return eval_sdf(cfg[0], cfg[1], cfg[2])

# ══════════════════════════════════════════════════════════════════
# 6.  START / GOAL
# ══════════════════════════════════════════════════════════════════

START = np.array([1.600, -1.902, 0.892])
GOAL  = np.array([-1.400, 1.902, -0.654])
print(eval_sdf_vec(START))
print(eval_sdf_vec(GOAL))

# ══════════════════════════════════════════════════════════════════
# 7.  LINE SCAN
# ══════════════════════════════════════════════════════════════════

line_length = np.linalg.norm(GOAL - START)
N_SEGMENTS  = max(5, int(np.ceil(line_length / 0.1)))
N_SAMPLES   = N_SEGMENTS + 1
t_vals      = np.linspace(0.0, 1.0, N_SAMPLES)
sample_cfgs = np.array([START + t*(GOAL-START) for t in t_vals])

print(f"\n[INFO] Line scan: {N_SAMPLES} samples …")
sdf_vals = []
for idx, cfg in enumerate(sample_cfgs):
    d = eval_sdf_vec(cfg)
    if idx == 0 or idx == N_SAMPLES-1:
        d = abs(d) if d != 0 else 1e-6
    sdf_vals.append(d)
    print(f"  [{idx:3d}] t={t_vals[idx]:.2f}  "
          f"q=({np.degrees(cfg[0]):+.1f}°,{np.degrees(cfg[1]):+.1f}°,{np.degrees(cfg[2]):+.1f}°)  "
          f"SDF={d:+.4f} ({'COLL' if d<0 else 'free'})")
sdf_vals = np.array(sdf_vals)

# ══════════════════════════════════════════════════════════════════
# 8.  REGULA FALSI
# ══════════════════════════════════════════════════════════════════

def regula_falsi_3d(cfg_a, cfg_b, sdf_a, sdf_b, tol=1e-4, max_iter=50):
    a, b = cfg_a.copy(), cfg_b.copy()
    fa, fb = sdf_a, sdf_b
    for _ in range(max_iter):
        cfg_c = a + fa*(a-b)/(fb-fa)
        fc    = eval_sdf_vec(cfg_c)
        if abs(fc) < tol: return cfg_c, fc
        if fa*fc < 0: b, fb = cfg_c, fc
        else:          a, fa = cfg_c, fc
    return cfg_c, fc

rf_zeros = []
print(f"\n[INFO] Regula Falsi …")
for i in range(N_SEGMENTS):
    si, sj = sdf_vals[i], sdf_vals[i+1]
    if si*sj < 0:
        cfg_r, sdf_r = regula_falsi_3d(sample_cfgs[i], sample_cfgs[i+1], si, sj)
        rf_zeros.append(cfg_r)
        print(f"  Seg {i}→{i+1}: zero at "
              f"({np.degrees(cfg_r[0]):+.2f}°,{np.degrees(cfg_r[1]):+.2f}°,{np.degrees(cfg_r[2]):+.2f}°)  "
              f"SDF={sdf_r:+.5f}")
print(f"  Total RF zeros: {len(rf_zeros)}")

# ══════════════════════════════════════════════════════════════════
# 9.  TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.35
q_vals = np.arange(-np.pi, np.pi + GRID_STEP*0.5, GRID_STEP)
N = len(q_vals)
print(f"\n[INFO] Grid: {N}x{N}x{N} nodes (step={GRID_STEP:.2f} rad)")

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

M             = len(tetrahedra)
tri_centroids = vertices[tetrahedra].mean(axis=1)
print(f"[INFO] {len(vertices):,} vertices, {M:,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# 10.  ADJACENCY
# ══════════════════════════════════════════════════════════════════

print("[INFO] Building adjacency …")
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
print("[INFO] Adjacency built.")

# ══════════════════════════════════════════════════════════════════
# 11.  FIND CONTAINING TET
# ══════════════════════════════════════════════════════════════════

def find_containing_tet(cfg):
    return int(np.argmin(np.linalg.norm(tri_centroids - cfg, axis=1)))

# ══════════════════════════════════════════════════════════════════
# 12.  MIDPOINTS → SEED TETS
# ══════════════════════════════════════════════════════════════════

start_processing_time = time.perf_counter()

seed_tets = []
midpoints = []
print(f"\n[INFO] Computing midpoints for consecutive RF zero pairs …")
for k in range(0, len(rf_zeros)-1, 2):
    mid     = (rf_zeros[k] + rf_zeros[k+1]) / 2.0
    mid_sdf = eval_sdf_vec(mid)
    midpoints.append((mid, mid_sdf))
    print(f"  Pair ({k},{k+1}) midpoint SDF={mid_sdf:+.5f} ({'COLL' if mid_sdf<0 else 'free'})")
    if mid_sdf < 0:
        ti = find_containing_tet(mid)
        seed_tets.append(ti)
        print(f"    → seed tet {ti}")

if not seed_tets and rf_zeros:
    ti = find_containing_tet(rf_zeros[0])
    if eval_sdf_vec(tri_centroids[ti]) < 0:
        seed_tets.append(ti)
        print(f"[INFO] Fallback seed from first RF zero → tet {ti}")

if not seed_tets:
    print("[WARN] No seed tet from RF zeros — scanning …")
    for ti in range(0, M, max(1, M//500)):
        if eval_sdf_vec(tri_centroids[ti]) < 0:
            seed_tets.append(ti)
            print(f"  Found fallback seed tet {ti}")
            break

print(f"\n[INFO] Seed tets: {seed_tets}")

# ══════════════════════════════════════════════════════════════════
# 13.  DIJKSTRA — flood all collision tets (SDF < COST_THRESHOLD)
#      No separation check — runs until heap is exhausted.
# ══════════════════════════════════════════════════════════════════

COST_THRESHOLD = -0.3

shaded   = set()
red_tets = []

if seed_tets:
    for pair_idx, seed_ti in enumerate(seed_tets):
        if seed_ti in shaded:
            print(f"[INFO] pair {pair_idx}: seed already shaded, skipping")
            continue

        seed_sdf = eval_sdf_vec(tri_centroids[seed_ti])
        shaded.add(seed_ti)
        red_tets.append(seed_ti)

        heap = []
        for nb in neighbours[seed_ti]:
            if nb not in shaded:
                nb_sdf = eval_sdf_vec(tri_centroids[nb])
                if nb_sdf < 0:
                    heapq.heappush(heap, (nb_sdf, nb))

        dead = set()
        step = 0

        print(f"\n[INFO] Dijkstra from seed tet {seed_ti}  (SDF={seed_sdf:+.5f})")

        while heap:
            cost, ti = heapq.heappop(heap)
            if ti in shaded or ti in dead:
                continue

            sdf_val = eval_sdf_vec(tri_centroids[ti])
            if sdf_val >= 0 or sdf_val > COST_THRESHOLD:
                dead.add(ti)
                continue

            shaded.add(ti)
            red_tets.append(ti)
            step += 1

            if step % 50 == 0:
                print(f"  step {step:6d}  tet {ti:6d}  SDF={sdf_val:+.5f}  shaded={len(shaded)}")

            for nb in neighbours[ti]:
                if nb not in shaded and nb not in dead:
                    nb_sdf = eval_sdf_vec(tri_centroids[nb])
                    if nb_sdf < 0 and nb_sdf <= COST_THRESHOLD:
                        heapq.heappush(heap, (nb_sdf, nb))

        print(f"  Heap exhausted after {step} steps.")

    _stop_reason = f"heap exhausted — {len(red_tets)} tets flooded"
else:
    _stop_reason = "no valid seed tetrahedra found"

end_processing_time = time.perf_counter()
total_time = end_processing_time - start_processing_time

print(f"\n[INFO] Dijkstra done.")
print(f"  Stop reason : {_stop_reason}")
print(f"  Shaded tets : {len(red_tets)}")
print(f"  Time        : {total_time:.4f}s")
print(f"  SDF queries : {_sdf_query_count}  (cache misses only)")

# ══════════════════════════════════════════════════════════════════
# 14.  BUILD PYVISTA MESH
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(tet_indices, verts_rad):
    if len(tet_indices) == 0:
        return None
    sel_tets       = tetrahedra[np.array(tet_indices, dtype=np.int64)]
    unique_vi, inv = np.unique(sel_tets, return_inverse=True)
    local_verts    = np.degrees(verts_rad[unique_vi])
    local_tets     = inv.reshape(-1, 4)
    n_tets         = len(local_tets)
    cells          = np.hstack([np.full((n_tets,1), 4, dtype=np.int64), local_tets]).ravel()
    celltypes      = np.full(n_tets, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, local_verts.astype(np.float64))

print("\n[INFO] Building PyVista mesh …")
red_mesh = build_tet_mesh(red_tets, vertices)

# ══════════════════════════════════════════════════════════════════
# 15.  PLOT
# ══════════════════════════════════════════════════════════════════

pl = pv.Plotter(window_size=[1400, 900])
pl.set_background("white")

if red_mesh is not None:
    pl.add_mesh(red_mesh, color="#C0392B", opacity=0.85,
                show_edges=True, edge_color="#7B241C", line_width=0.4,
                label=f"Collision tets ({len(red_tets)})")

line_pts = np.degrees(sample_cfgs)
spline   = pv.Spline(line_pts, 200)
pl.add_mesh(spline, color="dodgerblue", line_width=4, label="C-space path")

if rf_zeros:
    pl.add_points(np.degrees(np.array(rf_zeros)), color="purple", point_size=18,
                  render_points_as_spheres=True, label="RF zeros")

if midpoints:
    pl.add_points(np.degrees(np.array([m for m,_ in midpoints])), color="yellow",
                  point_size=16, render_points_as_spheres=True, label="Midpoints")

pl.add_points(np.degrees(START).reshape(1,3), color="lime",   point_size=22,
              render_points_as_spheres=True, label="Start")
pl.add_points(np.degrees(GOAL).reshape(1,3),  color="orange", point_size=22,
              render_points_as_spheres=True, label="Goal")

pl.add_axes(xlabel="q1 (deg)", ylabel="q2 (deg)", zlabel="q3 (deg)")
pl.add_legend(bcolor="white", border=True, size=(0.30, 0.22))
pl.add_title(
    f"3D C-Space Dijkstra | {len(red_tets)} collision tets | "
    f"queries={_sdf_query_count} | time={total_time:.2f}s",
    font_size=10)

print("[INFO] Opening PyVista window …")
pl.show()
p.disconnect()
print("[INFO] Done.")