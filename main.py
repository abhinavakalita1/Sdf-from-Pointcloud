import cv2
import numpy as np
import math
import time

t1 = time.time()
# ════════════════════════════════════════════════════════════════════
#  CANVAS & WALLS
# ════════════════════════════════════════════════════════════════════
W, H = 480, 480
canvas = 255 * np.ones((H, W, 3), np.uint8)
WALL_COLOR = [0, 0, 0]

cv2.rectangle(canvas, (0,   0),   (479, 479), tuple(WALL_COLOR), 10)
cv2.rectangle(canvas, (80,  80),  (250, 250), tuple(WALL_COLOR), 10)
cv2.rectangle(canvas, (200, 250), (300, 400), tuple(WALL_COLOR), 10)

start  = (230, 350)   # (col, row) = (x, y)
target = (100, 100)

gray    = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
obs_map = (gray < 10).astype(np.uint8) * 255   # black pixels = wall

# ════════════════════════════════════════════════════════════════════
#  STEP 1 — COXETER TRIANGULATION
#
#  The A2* Coxeter triangulation tiles the plane with equilateral
#  triangles. Given a side length s, the two basis vectors are:
#    e1 = (s, 0)
#    e2 = (s/2, s*sqrt(3)/2)
#  Every integer combination i*e1 + j*e2 gives a lattice vertex.
#
#  Triangles are formed by the two families:
#    "up"   triangle: (i,j), (i+1,j), (i,j+1)
#    "down" triangle: (i+1,j), (i+1,j+1), (i,j+1)
# ════════════════════════════════════════════════════════════════════
STEP = 28

def lattice_to_pixel(i, j):
    """Convert lattice coords (i,j) to pixel (col, row)."""
    col = int(round(i * STEP + j * STEP / 2))
    row = int(round(j * STEP * math.sqrt(3) / 2))
    return (col, row)

# Compute lattice range to cover the canvas
i_max = int(W / STEP) + 3
j_max = int(H / (STEP * math.sqrt(3) / 2)) + 3

# Build vertex grid
vertices = {}

for j in range(-1, j_max + 1):
    for i in range(-1, i_max + 1):
        px = lattice_to_pixel(i, j)
        vertices[(i, j)] = px

def triangle_pts(i, j, up=True):

    if up:
        return [
            vertices[(i, j)],
            vertices[(i + 1, j)],
            vertices[(i, j + 1)]
        ]
    else:
        return [
            vertices[(i + 1, j)],
            vertices[(i + 1, j + 1)],
            vertices[(i, j + 1)]
        ]

def centroid(pts):

    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts)
    )

def in_canvas(pt):

    return 0 <= pt[0] < W and 0 <= pt[1] < H

def triangle_is_free(pts):
    """True if centroid of triangle is in free space."""

    cx, cy = centroid(pts)

    c = int(round(cx))
    r = int(round(cy))

    if not (0 <= r < H and 0 <= c < W):
        return False

    return obs_map[r, c] == 0

# Collect all triangles visible on canvas
triangles = []

for j in range(-1, j_max):
    for i in range(-1, i_max):

        for up in [True, False]:

            pts = triangle_pts(i, j, up)

            if any(in_canvas(p) for p in pts):
                triangles.append(pts)

# ════════════════════════════════════════════════════════════════════
#  STEP 2 — BOUNDARY TRACING
# ════════════════════════════════════════════════════════════════════

contours_raw, hier = cv2.findContours(
    obs_map,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

boundaries = []

for i, c in enumerate(contours_raw):

    area = cv2.contourArea(c)

    if 5000 < area < 150000:
        boundaries.append(c)

# Sort by area descending so B1 = largest
boundaries.sort(key=cv2.contourArea, reverse=True)

boundary_labels = [f"B{i+1}" for i in range(len(boundaries))]

print(f"Found {len(boundaries)} boundaries: {boundary_labels}")

for i, b in enumerate(boundaries):
    print(f"  {boundary_labels[i]}: area = {cv2.contourArea(b):.0f}")

# ════════════════════════════════════════════════════════════════════
#  STEP 3 — RAY CASTING
# ════════════════════════════════════════════════════════════════════

def ray_intersects_segment(px, py, ax, ay, bx, by):

    if ay == by:
        return False

    if not (min(ay, by) <= py < max(ay, by)):
        return False

    t = (py - ay) / (by - ay)

    x_int = ax + t * (bx - ax)

    return x_int >= px

def count_intersections(point, contour):

    px, py = point

    pts = contour[:, 0, :]

    n = len(pts)

    count = 0

    for i in range(n):

        ax, ay = pts[i]

        bx, by = pts[(i + 1) % n]

        if ray_intersects_segment(px, py, ax, ay, bx, by):
            count += 1

    return count

def point_status(point, boundaries):

    results = []

    for lbl, bnd in zip(boundary_labels, boundaries):

        n = count_intersections(point, bnd)

        results.append((lbl, n % 2 == 1, n))

    return results

start_status  = point_status(start, boundaries)
target_status = point_status(target, boundaries)

# Same region = identical inside/outside signature
start_sig  = tuple(s[1] for s in start_status)
target_sig = tuple(s[1] for s in target_status)

same_region = (start_sig == target_sig)

print(f"\nStart  signature: {start_sig}")
print(f"Target signature: {target_sig}")
print(f"Same connected region: {same_region}")

# ════════════════════════════════════════════════════════════════════
#  DRAWING
# ════════════════════════════════════════════════════════════════════

GRID_COLOR      = (200, 200, 200)
BOUNDARY_COLORS = [
    (0,   0,   220),
    (0,   180, 0),
    (180, 0,   180),
    (0,   180, 180),
]

RAY_COLOR       = (100, 100, 255)
SAME_COLOR      = (0, 200, 0)
DIFF_COLOR      = (0, 0, 220)

img = canvas.copy()

# Draw triangulation
for pts in triangles:

    arr = np.array(pts, dtype=np.int32)

    free = triangle_is_free(pts)

    if free:
        cv2.polylines(
            img,
            [arr],
            isClosed=True,
            color=GRID_COLOR,
            thickness=1
        )

# Draw boundaries
for i, (bnd, lbl) in enumerate(zip(boundaries, boundary_labels)):

    col = BOUNDARY_COLORS[i % len(BOUNDARY_COLORS)]

    cv2.drawContours(img, [bnd], -1, col, 2)

    pts2 = bnd[:, 0, :]

    top_idx = np.argmin(pts2[:, 1])

    tx, ty = pts2[top_idx]

    cv2.putText(
        img,
        lbl,
        (int(tx) + 5, int(ty) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        col,
        2,
        cv2.LINE_AA
    )

# Draw ray from start
ray_end = (W - 5, start[1])

cv2.arrowedLine(
    img,
    start,
    ray_end,
    RAY_COLOR,
    1,
    tipLength=0.02
)

# Mark intersections on start ray
for bnd in boundaries:

    pts2 = bnd[:, 0, :]

    n = len(pts2)

    for k in range(n):

        ax, ay = pts2[k]

        bx, by = pts2[(k + 1) % n]

        px, py = start

        if ray_intersects_segment(px, py, int(ax), int(ay), int(bx), int(by)):

            t = (py - ay) / (by - ay) if ay != by else 0

            x_int = int(ax + t * (bx - ax))

            cv2.circle(img, (x_int, py), 5, (0, 80, 255), -1)

# Draw ray from target
ray_end_t = (W - 5, target[1])

cv2.arrowedLine(
    img,
    target,
    ray_end_t,
    (180, 80, 0),
    1,
    tipLength=0.02
)

for bnd in boundaries:

    pts2 = bnd[:, 0, :]

    n = len(pts2)

    for k in range(n):

        ax, ay = pts2[k]

        bx, by = pts2[(k + 1) % n]

        px, py = target

        if ray_intersects_segment(px, py, int(ax), int(ay), int(bx), int(by)):

            t = (py - ay) / (by - ay) if ay != by else 0

            x_int = int(ax + t * (bx - ax))

            cv2.circle(img, (x_int, py), 5, (0, 160, 255), -1)

# Draw walls on top
cv2.rectangle(img, (0,   0),   (479, 479), tuple(WALL_COLOR), 10)
cv2.rectangle(img, (80,  80),  (250, 250), tuple(WALL_COLOR), 10)
cv2.rectangle(img, (200, 250), (300, 400), tuple(WALL_COLOR), 10)

# Draw start and goal
cv2.circle(img, start,  9, (0, 180, 0), -1)
cv2.circle(img, target, 9, (0, 0, 200), -1)

cv2.putText(
    img,
    "S",
    (start[0] + 12, start[1] + 5),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 100, 0),
    2
)

cv2.putText(
    img,
    "G",
    (target[0] + 12, target[1] + 5),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 0, 120),
    2
)

# Info panel
panel_y = 8

print("\n================ START STATUS ================\n")

for s in start_status:

    txt = f"Start vs {s[0]}: {'INSIDE' if s[1] else 'OUTSIDE'} ({s[2]} intersections)"

    print(txt)

print("\n================ GOAL STATUS =================\n")

for s in target_status:

    txt = f"Goal vs {s[0]}: {'INSIDE' if s[1] else 'OUTSIDE'} ({s[2]} intersections)"

    print(txt)

print("\n================ FINAL VERDICT ===============\n")

verdict = (
    "SAME REGION — PATH MAY EXIST"
    if same_region
    else
    "DIFFERENT REGIONS — PATH CANNOT EXIST"
)

print(verdict)
cv2.imshow("Coxeter Topology", img)

cv2.waitKey(1)

cv2.destroyAllWindows()

print(time.time() - t1)