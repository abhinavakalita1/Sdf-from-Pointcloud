import cv2
import numpy as np
import math

from main import boundary_labels

W, H = 480, 480
canvas = 255 * np.ones((H, W, 3), np.uint8)
WALL_COLOR = [0, 0, 0]

# Draw obstacles
cv2.rectangle(canvas, (0, 0), (479, 479), tuple(WALL_COLOR), 10)
cv2.rectangle(canvas, (80, 80), (250, 250), tuple(WALL_COLOR), 20)
cv2.rectangle(canvas, (200, 250), (300, 400), tuple(WALL_COLOR), 20)

start = (230, 350)
target = (100, 100)

# Preprocess for boundary detection
gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
obs_map = (gray < 10).astype(np.uint8) * 255

# Find boundary contours
contours_raw, _ = cv2.findContours(obs_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundary_points = []
for c in contours_raw:
    area = cv2.contourArea(c)
    if 5000 < area < 150000:
        boundary_points.extend([tuple(point[0]) for point in c])

# Function to calculate line intersection
def line_intersection(p1, p2, p3):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3;
    a = (x3-x1)/(x2-x1)
    b = (y3-y1)/(y2-y1)
    return True if a>=0 and b>=0 and a<=1 and b>=1 and a<2 and a==b else False


# Find intersections (B1, B2, ...)
intersections = []
for p1 in boundary_points:
    if p1[0]<min(start[0], target[0]) or p1[1]<min(start[1], target[1]) or p1[0]>max(start[0], target[0]) or p1[1]>max(start[1], target[1]):
        continue
    pt = line_intersection(start, target, p1)
    if pt:
        intersections.append(p1)

boundary_labels = [[] for i in intersections]

# STEP 1 — A2* COXETER TRIANGULATION
STEP = 20
sqrt3 = math.sqrt(3)

# Generate A2* lattice points
lattice_points = []
for j in range(int(-H / (STEP * sqrt3 / 2)), int(H / (STEP * sqrt3 / 2)) + 2):
    y = int(j * STEP * sqrt3 / 2)
    for i in range(int(-W / STEP), int(W / STEP) + 2):
        x = int(i * STEP + j * STEP / 2)
        if 0 <= x < W and 0 <= y < H:
            lattice_points.append((x, y))

# Perform Delaunay triangulation on the lattice
rect = (0, 0, W, H)
subdiv = cv2.Subdiv2D(rect)
for p in lattice_points:
    subdiv.insert(p)
triangles = subdiv.getTriangleList()

# Draw triangulation, main line, and intersections
for t in triangles:
    pts = np.array([(t[i], t[i+1]) for i in range(0, 6, 2)], np.int32)
    cv2.polylines(canvas, [pts], True, (255, 0, 0), 1)

cv2.line(canvas, start, target, (0, 0, 0), 1)

for i, pt in enumerate(intersections):
    cv2.circle(canvas, pt, 3, (0, 0, 255), -1)
    cv2.putText(canvas, f'B{i+1}', (pt[0]+5, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

# # Print results
# print(f"Start {start} is {'inside' if start_result > 0 else 'outside' if start_result < 0 else 'on boundary'}")
# print(f"Target {target} is {'inside' if target_result > 0 else 'outside' if target_result < 0 else 'on boundary'}")

cv2.imshow('Canvas', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()