import cv2
import numpy as np
import math

start = [235, 90]
goal = [335, 335]
base = [220, 330]
link1 = 100
link2 = 70
fractions = [0.2,0.4,0.6,0.8]

W, H = 360, 360
STEP = 15
sqrt3 = math.sqrt(3)

canvas = cv2.imread("canvas.png")
cv2.rectangle(canvas, (0, 0), (479, 479), (0, 0, 0), 10)

mask = np.all(canvas == [0, 0, 0], axis=2)
ys, xs = np.where(mask)
obs = list(zip(xs, ys))

foo = 1 * np.ones((480, 480, 3), dtype=np.uint8)
cspace = np.ones((360, 360, 3), dtype=np.uint8) * 255

cv2.circle(cspace, tuple(start), 5, (0, 0, 255), -1)
cv2.circle(cspace, tuple(goal), 5, (0, 255, 0), -1)

cv2.putText(foo, "Theta1", (200, 450), 1, 1, (255,255,255), 1)
cv2.putText(foo, "Theta2", (0, 300), 1, 1, (255,255,255), 1)

def plot_arm(start, goal):
    canvas = cv2.imread("canvas.png")
    cv2.rectangle(canvas, (0, 0), (479, 479), (0, 0, 0), 10)

    pt1 = (int(base[0] + link1 * math.cos(math.radians(goal[0]))), int(base[1] + link1 * math.sin(math.radians(goal[0]))))
    pt2 = (
        int(pt1[0] + link2 * math.cos(math.radians(goal[0] + goal[1]))),
        int(pt1[1] + link2 * math.sin(math.radians(goal[0] + goal[1])))
    )
    cv2.line(canvas, base, pt1, (0,255,0), 10)
    cv2.line(canvas, pt1, pt2, (0,255,0), 10)


    pt1 = (int(base[0] + link1 * math.cos(math.radians(start[0]))), int(base[1] + link1 * math.sin(math.radians(start[0]))))
    pt2 = (
        int(pt1[0] + link2 * math.cos(math.radians(start[0] + start[1]))),
        int(pt1[1] + link2 * math.sin(math.radians(start[0] + start[1])))
    )
    cv2.line(canvas, base, pt1, (0,0,255), 10)
    cv2.line(canvas, pt1, pt2, (0,0,255), 10)

    return canvas, pt1, pt2

def is_colliding(canvas, pt1, pt2):
    roi = canvas[100:300, 100:300]
    red_pixels = np.where((roi[:, :, 2] == 255) & (roi[:, :, 1] == 0) & (roi[:, :, 0] == 0))
    for y, x in zip(*red_pixels):
        y_min, y_max = max(0, y-1), min(roi.shape[0], y+2)
        x_min, x_max = max(0, x-1), min(roi.shape[1], x+2)
        neighborhood = roi[y_min:y_max, x_min:x_max]
        if np.any((neighborhood[:, :, 0] == 0) & (neighborhood[:, :, 1] == 0) & (neighborhood[:, :, 2] == 0)):
            return True
    return False


## Pipeline
cv2.line(cspace, start, goal, (0, 0, 0), 2)

boundaries = []

for t in fractions:

    # interpolate point
    theta1 = int(start[0] + t * (goal[0] - start[0]))
    theta2 = int(start[1] + t * (goal[1] - start[1]))

    pt = (theta1, theta2)
    canvas, pt1, pt2 = plot_arm(pt, goal)

    cv2.circle(cspace, pt, 5, (0, 0, 0), -1)



    if is_colliding(canvas, pt1, pt2):
        if len(boundaries) == 0 or len(boundaries) == 1:
            boundaries.append([theta1, theta2])
        else:
            boundaries[-1] = [theta1, theta2]

lattice_points = []
for j in range(int(-H / (STEP * sqrt3 / 2)), int(H / (STEP * sqrt3 / 2)) + 2):
    y = int(j * STEP * sqrt3 / 2)
    for i in range(int(-W / STEP), int(W / STEP) + 2):
        x = int(i * STEP + j * STEP / 2)
        if 0 <= x < W and 0 <= y < H:
            lattice_points.append((x, y))


rect = (0, 0, W, H)
subdiv = cv2.Subdiv2D(rect)
for p in lattice_points:
    subdiv.insert(p)
triangles = subdiv.getTriangleList()

def find_triangle_for_points(subdiv, points):
    triangles = []
    for pt in points:
        loc, edge, vertex = subdiv.locate(pt)
        triangles.append((loc == cv2.SUBDIV2D_PTLOC_INSIDE, edge, vertex))
    return triangles

t1 = find_triangle_for_points(subdiv, boundaries)

print(t1)

for t in triangles:
    pts = np.array([(t[i], t[i+1]) for i in range(0, 6, 2)], np.int32)
    cv2.line(cspace, tuple(pts[0]), tuple(pts[1]), (0, 0, 0), 1)
    cv2.line(cspace, tuple(pts[1]), tuple(pts[2]), (0, 0, 0), 1)
    cv2.line(cspace, tuple(pts[2]), tuple(pts[0]), (0, 0, 0), 1)


foo[55:415, 55:415] = cspace
img = np.hstack((canvas, foo))
cv2.imshow('Canvas', img)
cv2.waitKey(0)

