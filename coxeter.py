import cv2
import numpy as np
import math

start = [235, 90]
goal = [335, 335]
base = [220, 330]
link1 = 100
link2 = 70
fractions = [0.2, 0.4, 0.6, 0.8]

# triangle side length
STEP = 10
# colors
TRI_COLOR = (180,180,180)
OBS_TRI   = (0,0,0)

H, W = 359, 359

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

def is_colliding(pt):
    canvas, pt1, pt2 = plot_arm(pt, goal)
    for t in fractions:
        x = int(base[0] + t * (pt1[0] - pt1[0]))
        y = int(base[1] + t * (pt1[1] - pt1[1]))
        x_min = max(x - 10, 0)
        x_max = min(x + 10, canvas.shape[1] - 1)

        y_min = max(y - 10, 0)
        y_max = min(y + 10, canvas.shape[0] - 1)

        for cy in range(y_min, y_max + 1):
            for cx in range(x_min, x_max + 1):

                # check if inside circle
                if (cx - x) ** 2 + (cy - y) ** 2 <= 10 ** 2:

                    # check if pixel is black
                    if np.all(canvas[y, x] == [0, 0, 0]):
                        return True

        x = int(pt1[0] + t * (pt2[0] - pt2[0]))
        y = int(pt1[1] + t * (pt2[1] - pt2[1]))
        x_min = max(x - 10, 0)
        x_max = min(x + 10, canvas.shape[1] - 1)

        y_min = max(y - 10, 0)
        y_max = min(y + 10, canvas.shape[0] - 1)

        for cy in range(y_min, y_max + 1):
            for cx in range(x_min, x_max + 1):

                # check if inside circle
                if (cx - x) ** 2 + (cy - y) ** 2 <= 10 ** 2:

                    # check if pixel is black
                    if np.all(canvas[y, x] == [0, 0, 0]):
                        return True

    return False

def lattice_to_pixel(i, j):

    x = int(round(i * STEP + j * STEP / 2))
    y = int(round(j * STEP * math.sqrt(3) / 2))

    return (x, y)

vertices = {}

imax = int(W / STEP) + 5
jmax = int(H / (STEP * math.sqrt(3)/2)) + 5

for j in range(-2, jmax):

    for i in range(-2, imax):

        vertices[(i,j)] = lattice_to_pixel(i,j)

def triangle_pts(i, j, up=True):

    if up:

        return np.array([
            vertices[(i,j)],
            vertices[(i+1,j)],
            vertices[(i,j+1)]
        ], dtype=np.int32)

    else:

        return np.array([
            vertices[(i+1,j)],
            vertices[(i+1,j+1)],
            vertices[(i,j+1)]
        ], dtype=np.int32)

# check if centroid lies in obstacle
def triangle_is_obstacle(pts):

    # bounding box of triangle
    x_min = max(int(np.min(pts[:,0])), 0)
    x_max = min(int(np.max(pts[:,0])), W - 1)

    y_min = max(int(np.min(pts[:,1])), 0)
    y_max = min(int(np.max(pts[:,1])), H - 1)

    # check every pixel inside bounding box
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):

            # point inside triangle?
            inside = cv2.pointPolygonTest(
                pts.astype(np.float32),
                (x, y),
                False
            )

            if inside >= 0:

                # black pixel found
                if np.all(cspace[y, x] == [0,0,0]):

                    return True

    return False

# draw triangulation
for j in range(-1, jmax):

    for i in range(-1, imax):

        for up in [True, False]:

            tri = triangle_pts(i, j, up)

            # skip if triangle fully outside image
            if np.all(tri[:,0] < 0) or np.all(tri[:,0] >= W):
                continue

            if np.all(tri[:,1] < 0) or np.all(tri[:,1] >= H):
                continue

            # obstacle check
            if triangle_is_obstacle(tri):

                cv2.polylines(
                    canvas,
                    [tri],
                    True,
                    OBS_TRI,
                    1
                )

            else:

                cv2.polylines(
                    canvas,
                    [tri],
                    True,
                    TRI_COLOR,
                    1
                )


## Pipeline
cv2.line(cspace, start, goal, (0, 0, 0), 2)

for t in fractions:

    # interpolate point
    theta1 = int(start[0] + t * (goal[0] - start[0]))
    theta2 = int(start[1] + t * (goal[1] - start[1]))

    pt = (theta1, theta2)

    if is_colliding(pt):
        cv2.circle(cspace, pt, 5, (0, 0, 0), -1)









    foo[55:415, 55:415] = cspace
    img = np.hstack((canvas, foo))
    cv2.imshow('Canvas', img)
    cv2.waitKey(0)