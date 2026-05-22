"""
RRT Wall-Hitting  →  SVM Boundary  →  Adaptive Inflation
→  Selective Coxeter Triangulation  →  Ray-Cast Topology Check

Pipeline
────────
1.  Two RRT trees grow from Start and Goal until each accumulates
    at least MIN_WALL_HITS nodes that either landed inside a wall
    (placed at last free position) or are within WALL_PROX pixels
    of a wall segment.

2.  RRT nodes → SVM training samples.
    Wall-hit nodes  → wall class (−1).
    Free RRT nodes + random free samples → free class (+1).
    A soft-margin linear SVM is trained via scikit-learn.

3.  Marching-squares extracts the SVM decision boundary (score = 0)
    as a set of polyline chains.

4.  Adaptive inflation per chain:
      a) Start at offset d = −MAX_INFLATE  (shell fully inside free space).
      b) Inflate by INFLATE_STEP each iteration.
      c) First d where any chain point touches a wall  → "first-touch shell".
         Record which point indices touched.
      d) Keep inflating until ALL those tracked points lift off the wall
         → "lift-off shell".  Stop.
    The band between first-touch and lift-off is the uncertain zone.

5.  Coxeter (A₂*) equilateral triangulation over the canvas.
    Only triangles whose centroid falls inside the inflation band
    get rendered.  All others are declared "intact" (no triangulation).

6.  Ray-casting topology:
      For Start and Goal, cast a horizontal ray to the right and count
      how many times it crosses each SVM boundary chain.
      Odd crossings  → inside that chain.
      Even crossings → outside.
      Build a binary signature (inside/outside) for each chain.
      If Start and Goal share the same signature → SAME REGION (path may exist).
      Different signatures → DIFFERENT REGIONS (path cannot exist).

    Additionally, check whether Start / Goal fall inside the Coxeter
    band itself (the uncertain zone).  Points in the band are flagged
    as "IN UNCERTAIN ZONE — topology inconclusive".

Output
──────
  • OpenCV window "RRT · SVM · Coxeter · Topology"
  • Console summary of all steps
"""

import cv2
import numpy as np
import math
import time
import random
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
W, H = 480, 480

START  = (230, 350)   # (col, row) = (x, y)
GOAL   = (100, 100)

# RRT
RRT_STEP       = 14    # pixels per extension
WALL_PROX      = 6     # px — a free node this close to a wall → wall-hit
MIN_WALL_HITS  = 30    # minimum wall-hit nodes per tree
MAX_RRT_ITER   = 15000

# SVM
EXTRA_FREE_SAMPLES = 400
EXTRA_WALL_SAMPLES = 600

# Marching squares grid step
MS_GRID = 5

# Adaptive inflation
INFLATE_STEP = 2      # px per inflation step
MAX_INFLATE  = 44     # maximum offset magnitude
TOUCH_TOL    = INFLATE_STEP + 2   # px — "touching" threshold

# Coxeter triangulation
TRI_STEP = 20   # side length of equilateral triangles (px)

# Ray-casting
RAY_EPS  = 1e-6   # nudge to avoid degenerate crossings

# ═══════════════════════════════════════════════════════════════════
#  WALL GEOMETRY
# ═══════════════════════════════════════════════════════════════════
# Every wall is described as a filled rectangle; the "wall" pixels
# are those on the thick border (5-px inset).

def is_wall(x: float, y: float) -> bool:
    r, c = int(round(y)), int(round(x))
    if r < 0 or r >= H or c < 0 or c >= W:
        return True
    # outer border 5 px
    if r < 5 or r >= H - 5 or c < 5 or c >= W - 5:
        return True
    # inner rectangle 1  (80,80)→(250,250), border 5 px
    if 80 <= r < 250 and 80 <= c < 250:
        if r < 85 or r >= 245 or c < 85 or c >= 245:
            return True
    # inner rectangle 2  (200,250)→(300,400), border 5 px
    if 250 <= r < 400 and 200 <= c < 300:
        if r < 255 or r >= 395 or c < 205 or c >= 295:
            return True
    return False


# Analytical wall segments (for distance and intersection tests)
WALL_SEGS = [
    # outer border
    ((5,5),   (475,5)),   ((475,5),  (475,475)),
    ((475,475),(5,475)),  ((5,475),  (5,5)),
    # inner rect 1
    ((85,85),  (245,85)), ((245,85), (245,245)),
    ((245,245),(85,245)), ((85,245), (85,85)),
    # inner rect 2
    ((205,255),(295,255)),((295,255),(295,395)),
    ((295,395),(205,395)),((205,395),(205,255)),
]


def dist_to_wall_segs(x: float, y: float) -> float:
    """Minimum distance from (x,y) to any wall segment."""
    best = 1e9
    for (ax, ay), (bx, by) in WALL_SEGS:
        dx, dy = bx - ax, by - ay
        l2 = dx*dx + dy*dy
        if l2 < 1e-9:
            continue
        t = max(0.0, min(1.0, ((x-ax)*dx + (y-ay)*dy) / l2))
        d = math.hypot(x - ax - t*dx, y - ay - t*dy)
        if d < best:
            best = d
    return best


def pt_touches_wall(x: float, y: float, tol: float) -> bool:
    return dist_to_wall_segs(x, y) <= tol


def seg_free(x1, y1, x2, y2) -> bool:
    """True if the straight segment does not pass through any wall."""
    n = max(2, int(math.hypot(x2-x1, y2-y1) / 2))
    for i in range(n + 1):
        t = i / n
        if is_wall(x1 + t*(x2-x1), y1 + t*(y2-y1)):
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
#  STEP 1 — RRT WALL-HITTING
# ═══════════════════════════════════════════════════════════════════

def rrt_wall_hitting(root, min_wall_hits: int, goal_pos=None):
    """
    Grow an RRT from `root` until `min_wall_hits` wall-hit nodes
    have been collected.

    Returns list of dicts:
        {'pos': (x,y), 'parent': int_or_None, 'wall_hit': bool}
    """
    nodes = [{'pos': root, 'parent': None, 'wall_hit': False}]
    wall_hit_count = 0

    for _ in range(MAX_RRT_ITER):
        if wall_hit_count >= min_wall_hits:
            break

        # Sample: 8% goal bias, rest uniform
        if goal_pos and random.random() < 0.08:
            rx, ry = goal_pos
        else:
            rx, ry = random.uniform(0, W), random.uniform(0, H)

        # Nearest node
        best_i, best_d = 0, 1e9
        for i, n in enumerate(nodes):
            d = math.hypot(n['pos'][0]-rx, n['pos'][1]-ry)
            if d < best_d:
                best_d, best_i = d, i

        near = nodes[best_i]['pos']
        ang  = math.atan2(ry - near[1], rx - near[0])
        nx   = near[0] + math.cos(ang) * RRT_STEP
        ny   = near[1] + math.sin(ang) * RRT_STEP

        if is_wall(nx, ny):
            # Walk back to last free point (wall-grazing node)
            fx, fy = near
            for s in range(1, RRT_STEP + 1):
                tx = near[0] + math.cos(ang) * s
                ty = near[1] + math.sin(ang) * s
                if is_wall(tx, ty):
                    break
                fx, fy = tx, ty
            if math.hypot(fx - near[0], fy - near[1]) < 1:
                continue
            wall_hit = True
            nodes.append({'pos': (fx, fy), 'parent': best_i, 'wall_hit': wall_hit})
            wall_hit_count += 1
        else:
            if not seg_free(near[0], near[1], nx, ny):
                continue
            dw = dist_to_wall_segs(nx, ny)
            wall_hit = dw <= WALL_PROX
            nodes.append({'pos': (nx, ny), 'parent': best_i, 'wall_hit': wall_hit})
            if wall_hit:
                wall_hit_count += 1

    return nodes


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 — SVM TRAINING
# ═══════════════════════════════════════════════════════════════════

def build_svm(tree_a, tree_b):
    """
    Train a LinearSVC on:
      +1 (free)  → non-wall-hit RRT nodes + random free samples
      −1 (wall)  → wall-hit RRT nodes + random wall samples
    Returns a callable score(x, y) → float  (positive = free side).
    """
    free_pts, wall_pts = [], []

    for n in tree_a + tree_b:
        if n['wall_hit']:
            wall_pts.append(n['pos'])
        else:
            free_pts.append(n['pos'])

    # extra random samples
    attempts = 0
    while len(free_pts) < len(free_pts) + EXTRA_FREE_SAMPLES and attempts < 20000:
        attempts += 1
        x, y = random.uniform(5, W-5), random.uniform(5, H-5)
        if not is_wall(x, y):
            free_pts.append((x, y))
        if len(free_pts) >= 200 + EXTRA_FREE_SAMPLES:
            break

    # Make sure we have enough
    rng_free = []
    while len(rng_free) < EXTRA_FREE_SAMPLES:
        x, y = random.uniform(5, W-5), random.uniform(5, H-5)
        if not is_wall(x, y):
            rng_free.append((x, y))
    free_pts.extend(rng_free)

    rng_wall = []
    while len(rng_wall) < EXTRA_WALL_SAMPLES:
        x, y = random.uniform(0, W), random.uniform(0, H)
        if is_wall(x, y):
            rng_wall.append((x, y))
    wall_pts.extend(rng_wall)

    X = np.array(free_pts + wall_pts, dtype=np.float32)
    y = np.array([1]*len(free_pts) + [-1]*len(wall_pts), dtype=np.float32)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    clf = LinearSVC(C=1.0, max_iter=2000)
    clf.fit(Xs, y)

    def score(px, py):
        pt = scaler.transform([[px, py]])
        return float(clf.decision_function(pt)[0])

    return score


# ═══════════════════════════════════════════════════════════════════
#  STEP 3 — MARCHING SQUARES
# ═══════════════════════════════════════════════════════════════════

def marching_squares(score_fn, grid_step=MS_GRID):
    """Extract the score=0 isoline as a list of (p1, p2) segment pairs."""
    cols = math.ceil(W / grid_step)
    rows = math.ceil(H / grid_step)

    # Build grid
    grid = [[score_fn(c*grid_step, r*grid_step) for c in range(cols+1)]
            for r in range(rows+1)]

    segs = []

    def interp(a, b):
        return 0.5 if abs(b-a) < 1e-9 else -a / (b - a)

    for r in range(rows):
        for c in range(cols):
            v = [grid[r][c], grid[r][c+1], grid[r+1][c+1], grid[r+1][c]]
            idx = ((1 if v[0]>0 else 0) << 3 | (1 if v[1]>0 else 0) << 2 |
                   (1 if v[2]>0 else 0) << 1 | (1 if v[3]>0 else 0))

            T = lambda: ((c + interp(v[0], v[1])) * grid_step, r * grid_step)
            R = lambda: ((c+1) * grid_step, (r + interp(v[1], v[2])) * grid_step)
            B = lambda: ((c + interp(v[3], v[2])) * grid_step, (r+1) * grid_step)
            L = lambda: (c * grid_step, (r + interp(v[0], v[3])) * grid_step)

            if idx in (0, 15):
                continue
            elif idx == 5:
                segs += [(T(), R()), (L(), B())]
            elif idx == 10:
                segs += [(T(), L()), (B(), R())]
            else:
                mp = {
                    1:  (L(), B()), 2:  (B(), R()), 3:  (L(), R()),
                    4:  (R(), T()), 6:  (B(), T()), 7:  (L(), T()),
                    8:  (T(), L()), 9:  (T(), B()), 11: (T(), R()),
                    12: (R(), L()), 13: (R(), B()), 14: (B(), L()),
                }
                if idx in mp:
                    segs.append(mp[idx])

    return segs


def chain_segments(segs, eps=8.0):
    """Chain raw (p1,p2) segments into polylines."""
    used  = [False] * len(segs)
    chains = []

    for s in range(len(segs)):
        if used[s]:
            continue
        used[s] = True
        chain = list(segs[s])        # [p0, p1]
        changed = True
        while changed:
            changed = False
            tail = chain[-1]
            for i in range(len(segs)):
                if used[i]:
                    continue
                p0, p1 = segs[i]
                if math.hypot(p0[0]-tail[0], p0[1]-tail[1]) < eps:
                    used[i] = True; chain.append(p1); changed = True; break
                if math.hypot(p1[0]-tail[0], p1[1]-tail[1]) < eps:
                    used[i] = True; chain.append(p0); changed = True; break
        if len(chain) > 3:
            chains.append(chain)

    return chains


# ═══════════════════════════════════════════════════════════════════
#  STEP 4 — ADAPTIVE INFLATION
# ═══════════════════════════════════════════════════════════════════

def offset_chain(pts, d):
    """Offset each point along the averaged outward normal by d pixels."""
    n   = len(pts)
    out = []
    for i in range(n):
        prev = pts[(i-1) % n]
        nxt  = pts[(i+1) % n]
        tx   = nxt[0] - prev[0]
        ty   = nxt[1] - prev[1]
        length = math.hypot(tx, ty) or 1.0
        # outward normal = rotate tangent 90° CCW
        nx_n = -ty / length
        ny_n =  tx / length
        out.append((pts[i][0] + nx_n * d, pts[i][1] + ny_n * d))
    return out


def adaptive_inflate(chain, inf_step, max_inflate, touch_tol):
    """
    Phase A: shrink to -max_inflate (fully inside free space).
    Phase B: inflate step by step.
      • First d where any point touches wall  → first_shell, record indices.
      • Keep going until ALL tracked indices lift off  → lift_shell. Stop.

    Returns (first_shell, lift_shell, contact_pts, first_d, lift_d)
    """
    first_shell   = None
    lift_shell    = None
    contact_pts   = []
    tracked_idx   = None
    first_d_val   = None

    d = -max_inflate
    while d <= max_inflate + inf_step:
        shell    = offset_chain(chain, d)
        touching = [pt_touches_wall(p[0], p[1], touch_tol) for p in shell]

        if first_shell is None:
            if any(touching):
                first_shell  = [p for p in shell]
                tracked_idx  = [i for i, t in enumerate(touching) if t]
                contact_pts  = [shell[i] for i in tracked_idx]
                first_d_val  = d
        else:
            # check if ALL originally-tracked points have lifted off
            all_off = all(
                not pt_touches_wall(shell[i][0], shell[i][1], touch_tol)
                for i in tracked_idx
            )
            if all_off:
                lift_shell = [p for p in shell]
                break

        d += inf_step

    # Fallbacks
    if first_shell is None:
        first_shell = list(chain)
        first_d_val = 0
    if lift_shell is None:
        lift_shell = offset_chain(chain, max_inflate)

    return first_shell, lift_shell, contact_pts, first_d_val


# ═══════════════════════════════════════════════════════════════════
#  STEP 5 — COXETER (A₂*) TRIANGULATION
# ═══════════════════════════════════════════════════════════════════

def coxeter_triangles(step):
    """
    Generate all equilateral triangles of the A₂* Coxeter tiling
    that have at least one vertex inside the canvas.

    Basis vectors:
        e1 = (step, 0)
        e2 = (step/2, step*√3/2)
    """
    sq3h = math.sqrt(3) / 2
    i_max = math.ceil(W / step) + 3
    j_max = math.ceil(H / (step * sq3h)) + 3

    def lattice_to_px(i, j):
        return (i * step + j * step / 2,
                j * step * sq3h)

    tris = []
    for j in range(-1, j_max):
        for i in range(-1, i_max):
            a = lattice_to_px(i,   j)
            b = lattice_to_px(i+1, j)
            c = lattice_to_px(i,   j+1)
            d = lattice_to_px(i+1, j+1)

            # "up" triangle
            if any(0 <= p[0] < W and 0 <= p[1] < H for p in (a, b, c)):
                tris.append((a, b, c))
            # "down" triangle
            if any(0 <= p[0] < W and 0 <= p[1] < H for p in (b, d, c)):
                tris.append((b, d, c))

    return tris


def centroid(pts):
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return cx, cy


def in_band(cx, cy, chain, first_shell, lift_shell):
    """
    True if (cx,cy) lies in the inflation band between first_shell and
    lift_shell.  We compare per-point distances along the chain.
    """
    min_d = 1e9
    best_k = 0
    best_t = 0.0

    for k in range(len(chain) - 1):
        ax, ay = chain[k]
        bx, by = chain[k+1]
        dx, dy = bx-ax, by-ay
        l2 = dx*dx + dy*dy
        if l2 < 1e-9:
            continue
        t  = max(0.0, min(1.0, ((cx-ax)*dx + (cy-ay)*dy) / l2))
        d  = math.hypot(cx - ax - t*dx, cy - ay - t*dy)
        if d < min_d:
            min_d, best_k, best_t = d, k, t

    def interp_chain(ch):
        k = min(best_k, len(ch)-2)
        t = best_t
        return (ch[k][0] + (ch[k+1][0]-ch[k][0])*t,
                ch[k][1] + (ch[k+1][1]-ch[k][1])*t)

    fp  = interp_chain(first_shell)
    lp  = interp_chain(lift_shell)
    band_w = math.hypot(fp[0]-lp[0], fp[1]-lp[1]) + 8.0

    d_first = math.hypot(cx-fp[0], cy-fp[1])
    d_lift  = math.hypot(cx-lp[0], cy-lp[1])

    return min_d < band_w and d_first < d_lift + 4.0


# ═══════════════════════════════════════════════════════════════════
#  STEP 6 — RAY-CASTING TOPOLOGY
# ═══════════════════════════════════════════════════════════════════

def ray_crosses_segment(px, py, ax, ay, bx, by):
    """
    Does the rightward horizontal ray from (px,py) cross segment (a→b)?
    Uses the standard half-open interval [min_y, max_y) convention.
    """
    if ay == by:
        return False
    if not (min(ay, by) <= py < max(ay, by)):
        return False
    t = (py - ay) / (by - ay)
    x_cross = ax + t * (bx - ax)
    return x_cross >= px


def count_crossings(point, chain):
    """Count how many times the rightward ray from point crosses chain."""
    px, py = point
    count  = 0
    n = len(chain)
    for i in range(n - 1):
        ax, ay = chain[i]
        bx, by = chain[i+1]
        if ray_crosses_segment(px, py, ax, ay, bx, by):
            count += 1
    return count


def point_signature(point, chains):
    """
    For each SVM chain, return True (inside) / False (outside).
    Returns list of booleans, one per chain.
    """
    return tuple(count_crossings(point, ch) % 2 == 1 for ch in chains)


def point_in_any_band(point, chains, chain_results):
    """True if the point falls inside ANY chain's inflation band."""
    px, py = point
    for ci, ch in enumerate(chains):
        fs, ls, _, _ = chain_results[ci]
        if in_band(px, py, ch, fs, ls):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_chain_cv(img, chain, color, thickness=1, dash=None):
    pts = [(int(round(p[0])), int(round(p[1]))) for p in chain]
    if dash:
        # simple dashed line
        on, off = dash
        seg_len = on + off
        for i in range(len(pts)-1):
            x0,y0 = pts[i]; x1,y1 = pts[i+1]
            total = math.hypot(x1-x0,y1-y0)
            if total < 1:
                continue
            steps = int(total / (on+off)) + 1
            for s in range(steps):
                t0 = s * seg_len / total
                t1 = min(1.0, (s * seg_len + on) / total)
                xa = int(x0 + t0*(x1-x0)); ya = int(y0 + t0*(y1-y0))
                xb = int(x0 + t1*(x1-x0)); yb = int(y0 + t1*(y1-y0))
                cv2.line(img, (xa,ya), (xb,yb), color, thickness, cv2.LINE_AA)
    else:
        for i in range(len(pts)-1):
            cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)


def draw_walls(img):
    cv2.rectangle(img, (0,0),   (W-1,H-1), (26,26,26), 10)
    cv2.rectangle(img, (80,80), (249,249),  (26,26,26), 10)
    cv2.rectangle(img, (200,250),(299,399), (26,26,26), 10)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    random.seed()   # non-deterministic each run

    # ── Canvas ───────────────────────────────────────────────────
    canvas = np.full((H, W, 3), 240, np.uint8)  # light grey background
    draw_walls(canvas)

    # ── 1. RRT ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1 — Growing RRT trees (target ≥ %d wall-hits each)" % MIN_WALL_HITS)
    tree_a = rrt_wall_hitting(START, MIN_WALL_HITS, goal_pos=GOAL)
    tree_b = rrt_wall_hitting(GOAL,  MIN_WALL_HITS, goal_pos=START)

    hits_a = sum(1 for n in tree_a if n['wall_hit'])
    hits_b = sum(1 for n in tree_b if n['wall_hit'])
    print(f"  Tree A: {len(tree_a)} nodes,  {hits_a} wall-hits")
    print(f"  Tree B: {len(tree_b)} nodes,  {hits_b} wall-hits")

    # ── 2. SVM ───────────────────────────────────────────────────
    print("\nSTEP 2 — Training SVM boundary classifier")
    score_fn = build_svm(tree_a, tree_b)
    print("  SVM trained (LinearSVC, C=1.0)")

    # ── 3. Marching squares ──────────────────────────────────────
    print("\nSTEP 3 — Extracting SVM decision boundary (marching squares)")
    segs   = marching_squares(score_fn, MS_GRID)
    chains = chain_segments(segs, eps=9.0)
    print(f"  Raw segments: {len(segs)}")
    print(f"  Chained polylines: {len(chains)}")

    # ── 4. Adaptive inflation ────────────────────────────────────
    print("\nSTEP 4 — Adaptive inflation per chain")
    chain_results = []
    for ci, ch in enumerate(chains):
        fs, ls, cpts, fd = adaptive_inflate(ch, INFLATE_STEP, MAX_INFLATE, TOUCH_TOL)
        chain_results.append((fs, ls, cpts, fd))
        n_contact = len(cpts)
        print(f"  Chain {ci+1}/{len(chains)}: {len(ch)} pts  "
              f"first-touch d={fd:.0f}  contact_pts={n_contact}")

    total_contact = sum(len(r[2]) for r in chain_results)
    print(f"  Total contact points across all chains: {total_contact}")

    # ── 5. Coxeter triangulation ─────────────────────────────────
    print("\nSTEP 5 — Coxeter (A₂*) triangulation")
    all_tris = coxeter_triangles(TRI_STEP)
    print(f"  Total triangles generated: {len(all_tris)}")

    tri_zone = []   # 'coxeter' | 'intact' | 'wall' | 'off'
    for tri in all_tris:
        cx, cy = centroid(tri)
        if not (0 <= cx < W and 0 <= cy < H):
            tri_zone.append('off')
            continue
        if is_wall(cx, cy):
            tri_zone.append('wall')
            continue
        in_z = False
        for ci, ch in enumerate(chains):
            fs, ls, _, _ = chain_results[ci]
            if in_band(cx, cy, ch, fs, ls):
                in_z = True
                break
        tri_zone.append('coxeter' if in_z else 'intact')

    n_cox    = tri_zone.count('coxeter')
    n_intact = tri_zone.count('intact')
    print(f"  Coxeter-zone triangles : {n_cox}")
    print(f"  Intact triangles       : {n_intact}")

    # ── 6. Ray-casting topology ──────────────────────────────────
    print("\nSTEP 6 — Ray-casting topology check")

    start_sig  = point_signature(START,  chains)
    target_sig = point_signature(GOAL,   chains)

    start_in_band  = point_in_any_band(START, chains, chain_results)
    target_in_band = point_in_any_band(GOAL,  chains, chain_results)

    same_region = (start_sig == target_sig)

    print("\n  === START STATUS ===")
    for ci, (inside, ch) in enumerate(zip(start_sig, chains)):
        crossings = count_crossings(START, ch)
        print(f"    Chain {ci+1}: {'INSIDE ' if inside else 'OUTSIDE'} "
              f"({crossings} crossing{'s' if crossings!=1 else ''})")
    if start_in_band:
        print("    ⚠ Start is inside UNCERTAIN BAND — topology inconclusive")

    print("\n  === GOAL STATUS ===")
    for ci, (inside, ch) in enumerate(zip(target_sig, chains)):
        crossings = count_crossings(GOAL, ch)
        print(f"    Chain {ci+1}: {'INSIDE ' if inside else 'OUTSIDE'} "
              f"({crossings} crossing{'s' if crossings!=1 else ''})")
    if target_in_band:
        print("    ⚠ Goal is inside UNCERTAIN BAND — topology inconclusive")

    print("\n  === SIGNATURES ===")
    print(f"    Start signature : {start_sig}")
    print(f"    Goal  signature : {target_sig}")
    print(f"    Match           : {same_region}")

    print("\n  === VERDICT ===")
    if start_in_band or target_in_band:
        verdict = "UNCERTAIN — one or both points in the inflation band"
        verdict_color = (0, 165, 255)   # orange
    elif same_region:
        verdict = "SAME REGION — a connecting path MAY EXIST"
        verdict_color = (0, 200, 80)    # green
    else:
        verdict = "DIFFERENT REGIONS — path CANNOT EXIST"
        verdict_color = (0, 0, 220)     # red

    print(f"    {verdict}")
    print("=" * 60)
    print(f"  Total time: {time.time()-t0:.2f} s")
    print("=" * 60)

    # ════════════════════════════════════════════════════════════
    #  RENDER
    # ════════════════════════════════════════════════════════════
    img = canvas.copy()

    # -- Intact triangle tint (very subtle green) --
    for ti, tri in enumerate(all_tris):
        if tri_zone[ti] != 'intact':
            continue
        pts_int = np.array([[int(round(p[0])), int(round(p[1]))] for p in tri], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_int], (180, 240, 200))
        cv2.addWeighted(overlay, 0.06, img, 0.94, 0, img)

    # -- Coxeter triangles in band --
    for ti, tri in enumerate(all_tris):
        if tri_zone[ti] != 'coxeter':
            continue
        pts_int = np.array([[int(round(p[0])), int(round(p[1]))] for p in tri], np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts_int], (180, 160, 255))
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)
        cv2.polylines(img, [pts_int], True, (120, 80, 210), 1, cv2.LINE_AA)

    # -- Inflation band fill --
    for ci, ch in enumerate(chains):
        fs, ls, cpts, _ = chain_results[ci]
        if len(fs) > 2 and len(ls) > 2:
            band_pts = ([(int(round(p[0])), int(round(p[1]))) for p in fs] +
                        [(int(round(p[0])), int(round(p[1]))) for p in reversed(ls)])
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(band_pts, np.int32)], (100, 180, 255))
            cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    # -- Lift-off shell (dashed red) --
    for ci, ch in enumerate(chains):
        fs, ls, _, _ = chain_results[ci]
        draw_chain_cv(img, ls, (60, 60, 220), 1, dash=(5, 3))

    # -- First-touch shell (dashed amber) --
    for ci, ch in enumerate(chains):
        fs, _, _, _ = chain_results[ci]
        draw_chain_cv(img, fs, (30, 160, 240), 1, dash=(5, 3))

    # -- SVM boundary (solid amber/yellow) --
    for ch in chains:
        draw_chain_cv(img, ch, (30, 190, 245), 2)

    # -- RRT tree A (green) --
    for n in tree_a:
        if n['parent'] is not None:
            par = tree_a[n['parent']]
            px0, py0 = int(par['pos'][0]), int(par['pos'][1])
            px1, py1 = int(n['pos'][0]),   int(n['pos'][1])
            cv2.line(img, (px0,py0), (px1,py1), (60, 170, 60), 1, cv2.LINE_AA)

    # -- RRT tree B (blue) --
    for n in tree_b:
        if n['parent'] is not None:
            par = tree_b[n['parent']]
            px0, py0 = int(par['pos'][0]), int(par['pos'][1])
            px1, py1 = int(n['pos'][0]),   int(n['pos'][1])
            cv2.line(img, (px0,py0), (px1,py1), (200, 100, 40), 1, cv2.LINE_AA)

    # -- RRT nodes --
    for n in tree_a:
        x, y = int(n['pos'][0]), int(n['pos'][1])
        if n['wall_hit']:
            cv2.circle(img, (x,y), 3, (40, 40, 200), -1)
        else:
            cv2.circle(img, (x,y), 1, (60, 170, 60), -1)

    for n in tree_b:
        x, y = int(n['pos'][0]), int(n['pos'][1])
        if n['wall_hit']:
            cv2.circle(img, (x,y), 3, (40, 40, 200), -1)
        else:
            cv2.circle(img, (x,y), 1, (200, 100, 40), -1)

    # -- Contact points --
    for ci, ch in enumerate(chains):
        _, _, cpts, _ = chain_results[ci]
        for pt in cpts:
            px, py = int(round(pt[0])), int(round(pt[1]))
            cv2.circle(img, (px, py), 5, (30, 130, 255), -1)
            cv2.circle(img, (px, py), 5, (255,255,255), 1)

    # -- Ray from START --
    ray_end_s = (W-5, START[1])
    cv2.arrowedLine(img, START, ray_end_s, (100, 100, 200), 1, tipLength=0.02)
    for ch in chains:
        n = len(ch)
        for i in range(n-1):
            ax,ay = ch[i]; bx,by = ch[i+1]
            if ray_crosses_segment(START[0], START[1], ax, ay, bx, by):
                t = (START[1]-ay)/(by-ay)
                xi = int(ax + t*(bx-ax))
                cv2.circle(img, (xi, START[1]), 5, (0, 80, 255), -1)

    # -- Ray from GOAL --
    ray_end_g = (W-5, GOAL[1])
    cv2.arrowedLine(img, GOAL, ray_end_g, (150, 80, 20), 1, tipLength=0.02)
    for ch in chains:
        n = len(ch)
        for i in range(n-1):
            ax,ay = ch[i]; bx,by = ch[i+1]
            if ray_crosses_segment(GOAL[0], GOAL[1], ax, ay, bx, by):
                t = (GOAL[1]-ay)/(by-ay)
                xi = int(ax + t*(bx-ax))
                cv2.circle(img, (xi, GOAL[1]), 5, (0, 160, 255), -1)

    # -- Redraw walls on top --
    draw_walls(img)

    # -- Start marker --
    cv2.circle(img, START, 9, (40, 200, 60), -1)
    cv2.circle(img, START, 9, (255,255,255), 1)
    cv2.putText(img, "S", (START[0]+12, START[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20,120,40), 2, cv2.LINE_AA)

    # -- Goal marker --
    cv2.circle(img, GOAL, 9, (220, 80, 40), -1)
    cv2.circle(img, GOAL, 9, (255,255,255), 1)
    cv2.putText(img, "G", (GOAL[0]+12, GOAL[1]+5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140,40,20), 2, cv2.LINE_AA)

    # -- Verdict banner --
    cv2.rectangle(img, (0, H-36), (W, H), (30,30,30), -1)
    cv2.putText(img, verdict, (8, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, verdict_color, 1, cv2.LINE_AA)

    # -- Legend --
    legend = [
        ((60,170,60),  "RRT-A edges"),
        ((200,100,40), "RRT-B edges"),
        ((40,40,200),  "Wall-hit nodes"),
        ((30,190,245), "SVM boundary"),
        ((30,160,240), "First-touch shell"),
        ((60,60,220),  "Lift-off shell"),
        ((30,130,255), "Contact pts"),
        ((120,80,210), "Coxeter zone"),
        ((100,200,140),"Intact region"),
    ]
    for idx, (col, label) in enumerate(legend):
        x0 = 8 + (idx % 3) * 158
        y0 = 10 + (idx // 3) * 16
        cv2.circle(img, (x0, y0), 4, col, -1)
        cv2.putText(img, label, (x0+9, y0+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (40,40,40), 1, cv2.LINE_AA)

    cv2.imshow("RRT · SVM · Coxeter · Topology", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()