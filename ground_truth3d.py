"""
ground_truth3d_sliders.py
──────────────────────────
3-DOF arm, 3D C-space. Exhaustive vertex-SDF scan → interior (collision)
tets rendered RED. Six PyVista sliders let you drag Start/Goal through
C-space interactively; SDF + FREE/COLL status updates in real time.
"""

import pybullet_data
import numpy as np
import pybullet as p
import time
import pyvista as pv

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP
# ══════════════════════════════════════════════════════════════════

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
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

_sdf_cache = {}

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
    key = (round(float(cfg[0]), 4), round(float(cfg[1]), 4), round(float(cfg[2]), 4))
    if key not in _sdf_cache:
        set_config(*cfg)
        _sdf_cache[key] = sdf_scene()
    return _sdf_cache[key]

# ══════════════════════════════════════════════════════════════════
# 4.  TETRAHEDRALISE C-SPACE
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.3
q_vals = np.arange(-np.pi, np.pi + GRID_STEP * 0.5, GRID_STEP)
N = len(q_vals)
print(f"[INFO] Grid: {N}³ = {N**3} vertices  step={GRID_STEP:.2f} rad")

def vidx(i, j, k):
    return i * N * N + j * N + k

vertices = np.zeros((N * N * N, 3))
for i in range(N):
    for j in range(N):
        for k in range(N):
            vertices[vidx(i,j,k)] = [q_vals[i], q_vals[j], q_vals[k]]

TET_OFFSETS = [
    (0,0,0),(1,0,0),(0,1,0),(0,0,1),
    (1,0,0),(1,1,0),(0,1,0),(1,0,1),
    (0,1,0),(1,1,0),(1,1,1),(0,1,1),
    (0,0,1),(1,0,1),(0,1,1),(1,1,1),
    (1,0,0),(0,1,0),(0,0,1),(1,0,1),
    (0,1,0),(0,0,1),(1,0,1),(1,1,1),
]
TET_PATTERNS = [TET_OFFSETS[t*4:(t+1)*4] for t in range(6)]

tetrahedra = np.array([
    [vidx(i+di, j+dj, k+dk) for (di,dj,dk) in pat]
    for i in range(N-1) for j in range(N-1) for k in range(N-1)
    for pat in TET_PATTERNS
], dtype=np.int32)

M = len(tetrahedra)
print(f"[INFO] {len(vertices):,} vertices | {M:,} tetrahedra")

# ══════════════════════════════════════════════════════════════════
# 5.  VERTEX SDF SCAN + TET CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

t0 = time.perf_counter()
print(f"[INFO] Evaluating SDF at all {len(vertices):,} vertices …")

vertex_sdf = np.zeros(len(vertices))
REPORT = max(1, len(vertices) // 20)
for vi in range(len(vertices)):
    vertex_sdf[vi] = eval_sdf_vec(vertices[vi])
    if (vi+1) % REPORT == 0 or vi == len(vertices)-1:
        print(f"  {100*(vi+1)/len(vertices):.0f}%  {time.perf_counter()-t0:.1f}s")

corner_sdfs   = vertex_sdf[tetrahedra]
interior_mask = (~(corner_sdfs > 0).any(axis=1)) & ((corner_sdfs < 0).any(axis=1))
interior_tets = np.where(interior_mask)[0]
print(f"[INFO] Interior tets: {len(interior_tets):,}  ({time.perf_counter()-t0:.1f}s)")

# ══════════════════════════════════════════════════════════════════
# 6.  BUILD STATIC RED MESH
# ══════════════════════════════════════════════════════════════════

def build_tet_mesh(tet_indices):
    if len(tet_indices) == 0:
        return None
    sel_tets = tetrahedra[tet_indices]
    unique_vi, inv = np.unique(sel_tets, return_inverse=True)
    local_verts = np.degrees(vertices[unique_vi])
    local_tets  = inv.reshape(-1, 4)
    n = len(local_tets)
    cells     = np.hstack([np.full((n,1), 4, dtype=np.int64), local_tets]).ravel()
    celltypes = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, local_verts.astype(np.float64))

interior_mesh = build_tet_mesh(interior_tets)

# ══════════════════════════════════════════════════════════════════
# 7.  INTERACTIVE PYVISTA PLOT WITH SLIDERS
# ══════════════════════════════════════════════════════════════════


state = {
    "start": np.array([1.6,  -1.9,   0.891]),   # radians
    "goal":  np.array([-1.4, 1.9,  -0.654]),
}

pl = pv.Plotter(window_size=[1500, 950])
pl.set_background("white")

if interior_mesh is not None:
    pl.add_mesh(interior_mesh, color="#C0392B", opacity=0.85,
                show_edges=True, edge_color="#7B241C", line_width=0.4,
                label=f"Collision tets ({len(interior_tets):,})")

# ── All dynamic actors use stable string names so remove_actor is reliable ──

def sdf_label(cfg):
    s = eval_sdf_vec(cfg)
    return "FREE" if s > 0 else "COLL", s

def refresh():
    """Remove named actors and re-add them at current state positions."""
    for name in ("start_pt", "goal_pt", "sdf_text"):
        pl.remove_actor(name)

    s_status, s_val = sdf_label(state["start"])
    g_status, g_val = sdf_label(state["goal"])

    pl.add_points(np.degrees(state["start"]).reshape(1, 3),
                  color="lime", point_size=22, render_points_as_spheres=True,
                  name="start_pt")
    pl.add_points(np.degrees(state["goal"]).reshape(1, 3),
                  color="orange", point_size=22, render_points_as_spheres=True,
                  name="goal_pt")

    text = (
        f"Start  ({np.degrees(state['start'][0]):+.1f}°,"
        f" {np.degrees(state['start'][1]):+.1f}°,"
        f" {np.degrees(state['start'][2]):+.1f}°)"
        f"  SDF={s_val:+.4f}  [{s_status}]\n"
        f"Goal   ({np.degrees(state['goal'][0]):+.1f}°,"
        f" {np.degrees(state['goal'][1]):+.1f}°,"
        f" {np.degrees(state['goal'][2]):+.1f}°)"
        f"  SDF={g_val:+.4f}  [{g_status}]"
    )
    pl.add_text(text, position="lower_left", font_size=10,
                color="black", name="sdf_text")

# Seed initial actors
refresh()

# ── Slider callbacks ──────────────────────────────────────────────

def make_cb(which, axis):
    def cb(val):
        state[which][axis] = np.radians(val)
        refresh()
    return cb

DEG_MIN, DEG_MAX = -180.0, 180.0

SLIDER_STYLE = dict(slider_width=0.02, tube_width=0.004,
                    title_height=0.022)

slider_defs = [
    # title        pointa        pointb        which     axis  init_deg
    ("Start q1", (0.02,0.92), (0.22,0.92), "start", 0),
    ("Start q2", (0.02,0.84), (0.22,0.84), "start", 1),
    ("Start q3", (0.02,0.76), (0.22,0.76), "start", 2),
    ("Goal  q1", (0.78,0.92), (0.98,0.92), "goal",  0),
    ("Goal  q2", (0.78,0.84), (0.98,0.84), "goal",  1),
    ("Goal  q3", (0.78,0.76), (0.98,0.76), "goal",  2),
]

for title, pa, pb, which, axis in slider_defs:
    pl.add_slider_widget(
        make_cb(which, axis),
        rng=[DEG_MIN, DEG_MAX],
        value=np.degrees(state[which][axis]),
        title=title,
        pointa=pa, pointb=pb,
        style="modern",
        color="steelblue",
        **SLIDER_STYLE,
    )

pl.add_axes(xlabel="q₁ (°)", ylabel="q₂ (°)", zlabel="q₃ (°)")
pl.add_legend(bcolor="white", border=True, size=(0.22, 0.10))
pl.add_title(
    f"3D C-Space — Collision Tets ({len(interior_tets):,}) | "
    f"Sliders: Start (lime) / Goal (orange)",
    font_size=10)

print("[INFO] Opening PyVista window …")
pl.show()
p.disconnect()