import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time, os, json, heapq
from collections import deque, defaultdict

# ══════════════════════════════════════════════════════════════════
# 1.  PYBULLET SETUP  (headless)
# ══════════════════════════════════════════════════════════════════

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
p.loadURDF("plane.urdf")

arm3Id = p.loadURDF(
    "arm_3.urdf", basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
NUM_JOINTS = p.getNumJoints(arm3Id)
print(f"[INFO] Arm loaded — {NUM_JOINTS} joints")

# ══════════════════════════════════════════════════════════════════
# 2.  OBSTACLES
# ══════════════════════════════════════════════════════════════════

def create_box(he, pos, ori, mass=0, color=[1,0,0,1]):
    c = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
    v = p.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=color)
    return p.createMultiBody(mass, c, v, pos, p.getQuaternionFromEuler(ori))

def create_sphere(r, pos, ori=[0,0,0], mass=0, color=[0,1,0,1]):
    c = p.createCollisionShape(p.GEOM_SPHERE, radius=r)
    v = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
    return p.createMultiBody(mass, c, v, pos, p.getQuaternionFromEuler(ori))

def create_cylinder(r, h, pos, ori=[0,0,0], mass=0, color=[0,0,1,1]):
    c = p.createCollisionShape(p.GEOM_CYLINDER, radius=r)
    v = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=color)
    return p.createMultiBody(mass, c, v, pos, p.getQuaternionFromEuler(ori))

boxId      = create_box([1,1,1],    [2,0,1],    [0.2,1.1,0.4])
sphereId   = create_sphere(1,       [0,2,1])
cylinderId = create_cylinder(0.3,2, [-0.5,0,1], [1.3,0,0])
ALL_OBSTACLE_IDS = [boxId, sphereId, cylinderId]

# ══════════════════════════════════════════════════════════════════
# 3.  POINTCLOUD + DBSCAN + HULL BODIES  (for Method 2 / hull SDF)
# ══════════════════════════════════════════════════════════════════

try:
    points = np.load("points.npy")
    print(f"[INFO] Loaded points.npy ({len(points)} pts)")
except FileNotFoundError:
    print("[WARN] points.npy not found — hull bodies will be empty, Method 2 SDF unreliable")
    points = np.zeros((0, 3))

CONFIG_FILE = "dbscan_config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f: params = json.load(f)
    eps, min_samples = params["eps"], params["min_samples"]
else:
    eps, min_samples = 0.15, 5

hull_body_ids = []
if len(points) > 0:
    def _group(pts, labels, min_pts=100):
        ul, counts = np.unique(labels, return_counts=True)
        small = [l for l,c in zip(ul,counts) if l != -1 and c < min_pts]
        for sl in small:
            sm   = labels == sl
            spts = pts[sm]
            dists = []
            for lbl in ul:
                if lbl == sl or lbl == -1: continue
                mask = labels == lbl
                if not mask.any(): continue
                dists.append((lbl, np.mean(np.linalg.norm(spts - pts[mask].mean(0), axis=1))))
            if dists:
                labels[sm] = min(dists, key=lambda x: x[1])[0]
        return labels

    db     = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = _group(points, db.labels_.copy())
    ul     = np.unique(labels[labels >= 0])
    for i, cpts in enumerate([points[labels==l] for l in ul]):
        try:
            hull   = ConvexHull(cpts)
            col_id = p.createCollisionShape(p.GEOM_MESH,
                               vertices=cpts[hull.vertices].tolist(), meshScale=[1,1,1])
            hull_body_ids.append(
                p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, basePosition=[0,0,0]))
        except Exception as e:
            print(f"  Cluster {i}: failed ({e})")
    print(f"[INFO] {len(hull_body_ids)} hull bodies ready")

# ══════════════════════════════════════════════════════════════════
# 4.  TWO SDF BACKENDS
#     primitives_sdf  — per-link queries against raw obstacle shapes
#                       (used by Ground Truth and Method 1)
#     hull_sdf        — queries convex-hull bodies
#                       (used by Method 2)
#
#     Both share the same PyBullet arm; both use independent caches.
# ══════════════════════════════════════════════════════════════════

_cache_prim = {}
_cache_hull = {}

def _set_config(q):
    p.resetJointState(arm3Id, 0, float(q[0]))
    p.resetJointState(arm3Id, 1, float(q[1]))
    p.resetJointState(arm3Id, 2, float(q[2]))
    p.stepSimulation()

def _raw_prim(threshold=10.0):
    min_d = threshold
    for obs_id in ALL_OBSTACLE_IDS:
        for link_idx in range(-1, NUM_JOINTS):
            contacts = p.getClosestPoints(bodyA=arm3Id, bodyB=obs_id,
                                          distance=threshold, linkIndexA=link_idx)
            if contacts:
                d = min(c[8] for c in contacts)
                if d < min_d: min_d = d
    return min_d

def _raw_hull(threshold=10.0):
    min_d = threshold
    for hid in hull_body_ids:
        contacts = p.getClosestPoints(bodyA=arm3Id, bodyB=hid, distance=threshold)
        if contacts:
            d = min(c[8] for c in contacts)
            if d < min_d: min_d = d
    return min_d

def sdf_prim(cfg):
    key = (round(float(cfg[0]),4), round(float(cfg[1]),4), round(float(cfg[2]),4))
    if key not in _cache_prim:
        _set_config(cfg)
        _cache_prim[key] = _raw_prim()
    return _cache_prim[key]

def sdf_hull(cfg):
    key = (round(float(cfg[0]),5), round(float(cfg[1]),5), round(float(cfg[2]),5))
    if key not in _cache_hull:
        _set_config(cfg)
        _cache_hull[key] = _raw_hull()
    return _cache_hull[key]

# ══════════════════════════════════════════════════════════════════
# 5.  SHARED 0.35 RAD TETRAHEDRAL GRID  (identical in all 3 scripts)
# ══════════════════════════════════════════════════════════════════

GRID_STEP = 0.35
q_vals    = np.arange(-np.pi, np.pi + GRID_STEP*0.5, GRID_STEP)
N         = len(q_vals)

def vidx(i,j,k): return i*N*N + j*N + k

VERTS = np.array([[q_vals[i], q_vals[j], q_vals[k]]
                  for i in range(N) for j in range(N) for k in range(N)])

TET_OFFSETS = [
    (0,0,0),(1,0,0),(0,1,0),(0,0,1),
    (1,0,0),(1,1,0),(0,1,0),(1,0,1),
    (0,1,0),(1,1,0),(1,1,1),(0,1,1),
    (0,0,1),(1,0,1),(0,1,1),(1,1,1),
    (1,0,0),(0,1,0),(0,0,1),(1,0,1),
    (0,1,0),(0,0,1),(1,0,1),(1,1,1),
]
TET_PATS = [TET_OFFSETS[t*4:(t+1)*4] for t in range(6)]

TETS = np.array([
    [vidx(i+di,j+dj,k+dk) for (di,dj,dk) in pat]
    for i in range(N-1) for j in range(N-1) for k in range(N-1)
    for pat in TET_PATS
], dtype=np.int32)

M        = len(TETS)
CENTROIDS = VERTS[TETS].mean(axis=1)

print(f"\n[INFO] Grid {N}³  step={GRID_STEP:.2f} rad  →  {len(VERTS):,} verts | {M:,} tets")

# ── Build adjacency once ──────────────────────────────────────────
print("[INFO] Building adjacency …")
_face_map = defaultdict(list)
for ti, tet in enumerate(TETS):
    for fi in range(4):
        face = tuple(sorted(tet[j] for j in range(4) if j != fi))
        _face_map[face].append(ti)

NBRS = [[] for _ in range(M)]
for face, tis in _face_map.items():
    if len(tis) == 2:
        a, b = tis
        NBRS[a].append(b)
        NBRS[b].append(a)
print("[INFO] Adjacency done.\n")

# ══════════════════════════════════════════════════════════════════
# 6.  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════

def find_closest_tet(cfg):
    return int(np.argmin(np.linalg.norm(CENTROIDS - cfg, axis=1)))

def line_scan_rf(start, goal, eval_fn, interval=0.1):
    """Return (rf_zeros, sample_cfgs, sdf_vals) along start→goal."""
    length  = np.linalg.norm(goal - start)
    n_seg   = max(5, int(np.ceil(length / interval)))
    t_vals  = np.linspace(0, 1, n_seg+1)
    cfgs    = np.array([start + t*(goal-start) for t in t_vals])
    sdfs    = np.array([eval_fn(c) for c in cfgs])
    # force endpoints to look free
    sdfs[0]  = abs(sdfs[0])  if sdfs[0]  != 0 else 1e-6
    sdfs[-1] = abs(sdfs[-1]) if sdfs[-1] != 0 else 1e-6

    zeros = []
    for i in range(n_seg):
        if sdfs[i]*sdfs[i+1] < 0:
            a, b, fa, fb = cfgs[i].copy(), cfgs[i+1].copy(), sdfs[i], sdfs[i+1]
            for _ in range(50):
                c  = a + fa*(a-b)/(fb-fa)
                fc = eval_fn(c)
                if abs(fc) < 1e-4: break
                if fa*fc < 0: b, fb = c, fc
                else:          a, fa = c, fc
            zeros.append(c)
    return zeros, cfgs, sdfs

def metrics(pred_set, gt_set):
    if not pred_set and not gt_set:
        return dict(precision=1.0, recall=1.0, f1=1.0, jaccard=1.0, tp=0, fp=0, fn=0)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set  - pred_set)
    prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
    jac  = tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0.0
    return dict(precision=prec, recall=rec, f1=f1, jaccard=jac, tp=tp, fp=fp, fn=fn)

# ══════════════════════════════════════════════════════════════════
# 7.  GROUND TRUTH  — ground_truth3d.py logic
#     Exhaustive vertex-SDF scan (primitives SDF), 0.35 grid.
#     Collision set = boundary tets ∪ interior tets.
#     SDF query count = number of NEW prim-cache misses during this call.
# ══════════════════════════════════════════════════════════════════

def run_ground_truth(start, goal):
    q_before = len(_cache_prim)
    t0 = time.perf_counter()

    # evaluate SDF at every vertex
    v_sdf = np.array([sdf_prim(VERTS[vi]) for vi in range(len(VERTS))])

    corner_sdfs   = v_sdf[TETS]                     # (M, 4)
    any_pos        = (corner_sdfs > 0).any(axis=1)
    any_neg        = (corner_sdfs < 0).any(axis=1)
    boundary_mask  = any_pos & any_neg
    interior_mask  = ~any_pos & any_neg
    collision_set  = set(np.where(boundary_mask | interior_mask)[0].tolist())

    elapsed   = time.perf_counter() - t0
    n_queries = len(_cache_prim) - q_before
    return collision_set, elapsed, n_queries

# ══════════════════════════════════════════════════════════════════
# 8.  METHOD 1  — cspace_3d_boundary.py logic
#     BFS from RF-zero seed tets, primitives SDF, 0.35 grid.
#     Propagates through boundary+interior, stops at free tets.
# ══════════════════════════════════════════════════════════════════

def run_method1_bfs(start, goal):
    q_before = len(_cache_prim)
    t0 = time.perf_counter()

    # line scan + RF with primitives SDF
    zeros, _, _ = line_scan_rf(start, goal, sdf_prim)

    # seed tets from RF zeros (seed at zero-crossing itself, like boundary script)
    seed_tets = []
    for z in zeros:
        ti = find_closest_tet(z)
        if ti not in seed_tets:
            seed_tets.append(ti)
    if not seed_tets:
        seed_tets = [find_closest_tet((start+goal)/2)]

    # lazy vertex SDF cache (only vertices touched by BFS)
    v_sdf_cache = {}
    def get_v_sdf(vi):
        if vi not in v_sdf_cache:
            v_sdf_cache[vi] = sdf_prim(VERTS[vi])
        return v_sdf_cache[vi]

    visited       = set(seed_tets)
    queue         = deque(seed_tets)
    boundary_tets = []
    interior_tets = []

    while queue:
        ti   = queue.popleft()
        sdfs = [get_v_sdf(vi) for vi in TETS[ti]]
        any_pos = any(s > 0 for s in sdfs)
        any_neg = any(s < 0 for s in sdfs)

        if any_pos and any_neg:          # boundary — propagate to all neighbours
            boundary_tets.append(ti)
            for nb in NBRS[ti]:
                if nb not in visited:
                    visited.add(nb); queue.append(nb)
        elif any_neg and not any_pos:    # interior — propagate through
            interior_tets.append(ti)
            for nb in NBRS[ti]:
                if nb not in visited:
                    visited.add(nb); queue.append(nb)
        # all-positive = free → stop propagation here

    collision_set = set(boundary_tets) | set(interior_tets)
    elapsed       = time.perf_counter() - t0
    n_queries     = len(_cache_prim) - q_before
    return collision_set, elapsed, n_queries

# ══════════════════════════════════════════════════════════════════
# 9.  METHOD 2  — cspace_3d_dijkstra.py logic
#     Dijkstra from midpoint-of-RF-zero-pair seeds, hull SDF, 0.35 grid.
#     Expands only through tets with centroid SDF < COST_THRESHOLD.
#     Stops when START and GOAL are topologically separated.
# ══════════════════════════════════════════════════════════════════

COST_THRESHOLD = 0.0   # tets with centroid SDF ≥ this are skipped
CHECK_EVERY    = 20

def run_method2_dijkstra(start, goal):
    q_before = len(_cache_hull)
    t0 = time.perf_counter()

    zeros, sample_cfgs, sdf_vals = line_scan_rf(start, goal, sdf_hull)

    # midpoints of consecutive RF-zero pairs → seeds
    seed_tets = []
    for k in range(0, len(zeros)-1, 2):
        mid = (zeros[k] + zeros[k+1]) / 2.0
        if sdf_hull(mid) < 0:
            seed_tets.append(find_closest_tet(mid))

    if not seed_tets and zeros:
        ti = find_closest_tet(zeros[0])
        if sdf_hull(CENTROIDS[ti]) < 0:
            seed_tets.append(ti)

    if not seed_tets:
        for ti in range(0, M, max(1, M//500)):
            if sdf_hull(CENTROIDS[ti]) < 0:
                seed_tets.append(ti); break

    def is_separated(shaded_set):
        s_ti = find_closest_tet(start)
        g_ti = find_closest_tet(goal)
        vis  = set()
        q    = deque([s_ti])
        while q:
            ti = q.popleft()
            if ti in vis: continue
            if ti == g_ti: return False
            vis.add(ti)
            for nb in NBRS[ti]:
                if nb not in vis and nb not in shaded_set:
                    q.append(nb)
        return True

    shaded   = set()
    red_tets = []

    for seed_ti in seed_tets:
        if seed_ti in shaded: continue
        shaded.add(seed_ti); red_tets.append(seed_ti)

        heap = []
        for nb in NBRS[seed_ti]:
            if nb not in shaded:
                nb_sdf = sdf_hull(CENTROIDS[nb])
                if nb_sdf < 0:
                    heapq.heappush(heap, (nb_sdf, nb))

        dead = set()
        step = 0
        separated = False

        while heap and not separated:
            cost, ti = heapq.heappop(heap)
            if ti in shaded or ti in dead: continue
            sdf_val = sdf_hull(CENTROIDS[ti])
            if sdf_val >= 0 or sdf_val > COST_THRESHOLD:
                dead.add(ti); continue
            shaded.add(ti); red_tets.append(ti)
            step += 1
            if step % CHECK_EVERY == 0:
                separated = is_separated(shaded)
            if not separated:
                for nb in NBRS[ti]:
                    if nb not in shaded and nb not in dead:
                        nb_sdf = sdf_hull(CENTROIDS[nb])
                        if nb_sdf < 0 and nb_sdf <= COST_THRESHOLD:
                            heapq.heappush(heap, (nb_sdf, nb))

    elapsed   = time.perf_counter() - t0
    n_queries = len(_cache_hull) - q_before
    return set(red_tets), elapsed, n_queries

# ══════════════════════════════════════════════════════════════════
# 10. FIND VALID START / GOAL PAIRS
#      Both endpoints must have SDF > 0 (primitives), be at least
#      1.5 rad apart, and the straight line must cross collision.
# ══════════════════════════════════════════════════════════════════

N_PAIRS   = 5
MAX_TRIES = 10_000

def path_has_collision(ca, cb, n_steps=30):
    for i in range(1, n_steps-1):
        t = i/(n_steps-1)
        if sdf_prim(ca + t*(cb-ca)) <= 0:
            return True
    return False

print("[INFO] Searching for valid START/GOAL pairs …")
np.random.seed(7)
pairs = []
attempt = 0
while len(pairs) < N_PAIRS and attempt < MAX_TRIES:
    attempt += 1
    ca = np.random.uniform(-np.pi, np.pi, 3)
    cb = np.random.uniform(-np.pi, np.pi, 3)
    if sdf_prim(ca) <= 0: continue
    if sdf_prim(cb) <= 0: continue
    if np.linalg.norm(cb - ca) < 1.5: continue
    if not path_has_collision(ca, cb): continue
    pairs.append((ca.copy(), cb.copy()))
    print(f"  Pair {len(pairs)}: "
          f"START=({np.degrees(ca[0]):+.1f}°,{np.degrees(ca[1]):+.1f}°,{np.degrees(ca[2]):+.1f}°)  "
          f"GOAL=({np.degrees(cb[0]):+.1f}°,{np.degrees(cb[1]):+.1f}°,{np.degrees(cb[2]):+.1f}°)")

print(f"\n[INFO] Found {len(pairs)} pairs in {attempt} attempts.\n")

# ══════════════════════════════════════════════════════════════════
# 11. RUN BENCHMARK
# ══════════════════════════════════════════════════════════════════

W = 10   # column width
HDR = (f"{'Pair':>4}  {'Method':<14}  {'Time(s)':>{W}}  {'SDF Qs':>{W}}  "
       f"{'Tets':>{W}}  {'Prec':>{W}}  {'Recall':>{W}}  {'F1':>{W}}  {'Jaccard':>{W}}")
SEP = "─" * len(HDR)

print(SEP)
print(HDR)
print(SEP)

all_results = []

for pi, (start, goal) in enumerate(pairs):
    rec = {"pair": pi+1, "start": start.tolist(), "goal": goal.tolist()}

    # ── Ground Truth ─────────────────────────────────────────────
    gt_set, gt_t, gt_q = run_ground_truth(start, goal)
    rec["ground_truth"] = {"time": gt_t, "sdf_queries": gt_q, "n_tets": len(gt_set)}
    print(f"{pi+1:>4}  {'GT-Exhaustive':<14}  {gt_t:>{W}.3f}  {gt_q:>{W}}  "
          f"{len(gt_set):>{W}}  {'—':>{W}}  {'—':>{W}}  {'—':>{W}}  {'—':>{W}}")

    # ── Method 1: BFS Boundary ───────────────────────────────────
    m1_set, m1_t, m1_q = run_method1_bfs(start, goal)
    m1_met = metrics(m1_set, gt_set)
    rec["method1_bfs"] = {"time": m1_t, "sdf_queries": m1_q,
                          "n_tets": len(m1_set), **m1_met}
    print(f"{pi+1:>4}  {'M1-BFS':<14}  {m1_t:>{W}.3f}  {m1_q:>{W}}  "
          f"{len(m1_set):>{W}}  {m1_met['precision']:>{W}.3f}  {m1_met['recall']:>{W}.3f}  "
          f"{m1_met['f1']:>{W}.3f}  {m1_met['jaccard']:>{W}.3f}")

    # ── Method 2: Dijkstra ───────────────────────────────────────
    m2_set, m2_t, m2_q = run_method2_dijkstra(start, goal)
    m2_met = metrics(m2_set, gt_set)
    rec["method2_dijkstra"] = {"time": m2_t, "sdf_queries": m2_q,
                               "n_tets": len(m2_set), **m2_met}
    print(f"{pi+1:>4}  {'M2-Dijkstra':<14}  {m2_t:>{W}.3f}  {m2_q:>{W}}  "
          f"{len(m2_set):>{W}}  {m2_met['precision']:>{W}.3f}  {m2_met['recall']:>{W}.3f}  "
          f"{m2_met['f1']:>{W}.3f}  {m2_met['jaccard']:>{W}.3f}")

    print(SEP)
    all_results.append(rec)

# ══════════════════════════════════════════════════════════════════
# 12. AGGREGATE SUMMARY
# ══════════════════════════════════════════════════════════════════

def avg(key, method):
    vals = [r[method][key] for r in all_results if method in r]
    return sum(vals)/len(vals) if vals else float('nan')

print(f"\n{'═'*len(HDR)}")
print("AGGREGATE AVERAGES ACROSS ALL PAIRS")
print(f"{'═'*len(HDR)}")

rows = [
    ("GT-Exhaustive", "ground_truth",      False),
    ("M1-BFS",        "method1_bfs",       True),
    ("M2-Dijkstra",   "method2_dijkstra",  True),
]
for label, key, has_metrics in rows:
    t  = avg("time",        key)
    q  = avg("sdf_queries", key)
    nt = avg("n_tets",      key)
    if has_metrics:
        prec = avg("precision", key)
        rec  = avg("recall",    key)
        f1   = avg("f1",        key)
        jac  = avg("jaccard",    key)
        print(f"  {label:<14}  time={t:.3f}s  sdf_q={q:.0f}  tets={nt:.0f}  "
              f"prec={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  jaccard={jac:.3f}")
    else:
        print(f"  {label:<14}  time={t:.3f}s  sdf_q={q:.0f}  tets={nt:.0f}  "
              f"(ground truth — no accuracy metrics)")

# ══════════════════════════════════════════════════════════════════
# 13. SAVE
# ══════════════════════════════════════════════════════════════════

out = "benchmark_results.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n[INFO] Results saved to {out}")

p.disconnect()
print("[INFO] Done.")