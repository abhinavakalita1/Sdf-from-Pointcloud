import cv2
import numpy as np
import random
import math
import time

t1 = time.time()
# ════════════════════════════════════════════════════════════════════
#  CANVAS & WALLS
# ════════════════════════════════════════════════════════════════════
W, H   = 480, 480
canvas = 255 * np.ones((H, W, 3), np.uint8)

WALL_COLOR = [0, 0, 0]

cv2.rectangle(canvas, (0, 0),     (479, 479), tuple(WALL_COLOR), 10)
cv2.rectangle(canvas, (80, 80),   (250, 250), tuple(WALL_COLOR), 10)
cv2.rectangle(canvas, (200, 250), (300, 400), tuple(WALL_COLOR), 10)

start  = (230, 350)
target = (100, 100)

# ════════════════════════════════════════════════════════════════════
#  DISTANCE TRANSFORM
# ════════════════════════════════════════════════════════════════════
def make_obs_map(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, obs = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY_INV)
    return obs.astype(np.uint8)

obs_map = make_obs_map(canvas)
dt      = cv2.distanceTransform(1 - obs_map, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

MIN_R = 2.5   # ← minimum allowed radius

def max_radius(col, row):
    r, c = int(round(row)), int(round(col))
    if r < 0 or r >= H or c < 0 or c >= W:
        return 0.0
    return float(dt[r, c])

def is_valid_center(col, row):
    """Free AND can host a circle of at least MIN_R."""
    return max_radius(col, row) >= MIN_R

# ════════════════════════════════════════════════════════════════════
#  CIRCLE STATE
# ════════════════════════════════════════════════════════════════════
circles = []
covered = np.zeros((H, W), dtype=np.uint8)

CIRCLE_COLOR = (180, 210, 255)

def _paint_covered(cx, cy, r):
    cx, cy, r = int(cx), int(cy), int(r)
    Y, X      = np.ogrid[:H, :W]
    covered[(X - cx)**2 + (Y - cy)**2 <= r**2] = 1

def add_circle(col, row):
    r = max_radius(col, row)
    if r < MIN_R:          # ← reject anything below minimum
        return False

    for c in circles:
        d = math.sqrt((col - c['cx'])**2 + (row - c['cy'])**2)
        if d + r <= c['r']:          # new inside existing → absorb
            return False
        if d + c['r'] <= r:          # existing inside new → replace
            c['cx'] = col
            c['cy'] = row
            c['r']  = r
            _paint_covered(col, row, r)
            return False

    circles.append({'cx': col, 'cy': row, 'r': r})
    _paint_covered(col, row, r)
    return True

# ════════════════════════════════════════════════════════════════════
#  CIRCUMFERENCE SAMPLING — only uncovered valid centers
# ════════════════════════════════════════════════════════════════════
def circumference_candidates(c, n_angles=36):
    cx, cy, r = c['cx'], c['cy'], c['r']
    out = []
    for i in range(n_angles):
        theta = 2 * math.pi * i / n_angles
        px    = cx + r * math.cos(theta)
        py    = cy + r * math.sin(theta)
        pc, pr = int(round(px)), int(round(py))

        if not (0 <= pc < W and 0 <= pr < H):
            continue
        if not is_valid_center(pc, pr):   # also enforces MIN_R
            continue
        if covered[pr, pc] == 1:          # already inside a circle
            continue

        out.append((px, py))
    return out

# ════════════════════════════════════════════════════════════════════
#  DRAW
# ════════════════════════════════════════════════════════════════════
def redraw():
    img = canvas.copy()

    for c in circles:
        cx, cy, r = int(c['cx']), int(c['cy']), int(c['r'])
        cv2.circle(img, (cx, cy), r, CIRCLE_COLOR, -1)   # solid, no outline

    # Walls always on top
    cv2.rectangle(img, (0, 0),     (479, 479), tuple(WALL_COLOR), 10)
    cv2.rectangle(img, (80, 80),   (250, 250), tuple(WALL_COLOR), 10)
    cv2.rectangle(img, (200, 250), (300, 400), tuple(WALL_COLOR), 10)

    cv2.circle(img, start,  8, (0, 180, 0),  -1)
    cv2.circle(img, target, 8, (0, 0, 200),  -1)

    cv2.putText(img, f"circles: {len(circles)}  |  any key=expand  ESC=quit",
                (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1)
    return img

# ════════════════════════════════════════════════════════════════════
#  SEED
# ════════════════════════════════════════════════════════════════════
add_circle(*start)
add_circle(*target)

attempts = 0
while len(circles) < 12 and attempts < 3000:
    attempts += 1
    col = random.randint(10, W - 10)
    row = random.randint(10, H - 10)
    if is_valid_center(col, row):
        add_circle(col, row)

cv2.namedWindow("Circle Expansion")
cv2.imshow("Circle Expansion", redraw())
cv2.waitKey(1)

# ════════════════════════════════════════════════════════════════════
#  EXPANSION LOOP
# ════════════════════════════════════════════════════════════════════
N_ANGLES = 24

while True:
    parents    = list(circles)
    added_any  = False

    for c in parents:
        for (px, py) in circumference_candidates(c, n_angles=N_ANGLES):
            if add_circle(px, py):
                added_any = True

    img = redraw()

    if not added_any:
        # ── Termination: no valid circle could be placed ──────────────
        cv2.putText(img, "TERMINATED — no room for new circles (min r = 2.5)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 200), 1)
        cv2.imshow("Circle Expansion", img)
        print("Terminated — free space fully covered or no circle >= 2.5px can be placed.")
        cv2.waitKey(0)
        break

    cv2.imshow("Circle Expansion", img)
    key = cv2.waitKey(0)
    if key == 27:
        break

cv2.destroyAllWindows()
print(time.time() - t1)