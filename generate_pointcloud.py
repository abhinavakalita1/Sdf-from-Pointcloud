"""
generate_pointcloud.py

Camera adjuster mode (uncomment CAMERA_ADJUSTER_MODE = True to use):
  - All 13 camera frames shown simultaneously in a matplotlib grid
  - PyBullet GUI sliders for each camera:
      • Along-axis distance   (how far the camera is from the target)
      • Azimuth angle         (rotate camera around target in horizontal plane)
      • Elevation angle       (raise/lower camera above target)
  - Sliders update live — move a slider, see the camera feed change instantly
  - Press Q or close the window to finish and save config to cameras.json
  - Next run reads cameras.json automatically if it exists

  To go back to pointcloud generation: set CAMERA_ADJUSTER_MODE = False
  (or just comment out the adjuster block at the bottom)
"""

import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import time
from mpl_toolkits.mplot3d import Axes3D


# ── Toggle ────────────────────────────────────────────────────
CAMERA_ADJUSTER_MODE = True    # set False to skip adjuster and go straight to pointcloud
CONFIG_FILE          = "cameras.json"


# ══════════════════════════════════════════════════════════════
# PYBULLET SETUP
# ══════════════════════════════════════════════════════════════

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")


# ══════════════════════════════════════════════════════════════
# OBSTACLE CREATION
# ══════════════════════════════════════════════════════════════

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

# boxId      = create_box(half_extents=[1,1,1],      position=[2,0,1],      orientation=[0.2,1.1,0.4])
# sphereId   = create_sphere(radius=1,               position=[0,2,1])
# cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1],   orientation=[1.3,0,0])

def load_mesh_obstacle(obj_path, position=[0,0,0],
                       orientation=[0,0,0], scale=1.0,
                       color=[0.8, 0.5, 0.2, 1]):
    col  = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    )
    vis  = p.createVisualShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        rgbaColor=color
    )
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(0, col, vis, position, quat)

# glassId   = load_mesh_obstacle("glass.obj",   position=[1, 0.3, 1], scale=0.1)
# bottleId = load_mesh_obstacle("Plastic-Bottle.obj", position=[-1,   0, 0], scale=0.1)
concaveId = load_mesh_obstacle("concave.obj", position=[0,   0.5, 0], scale=0.4)

obstacle_ids   = [concaveId]
obstacle_names = ["Concave"]

# ══════════════════════════════════════════════════════════════
# CAMERA CAPTURE FUNCTION
# ══════════════════════════════════════════════════════════════

TARGET = [0.75, 1.0, 1.0]     # scene centre all cameras look at

def capture(eye, target=TARGET, up=None, sparsity=4,
            width=320, height=240, fov=60, near=0.1, far=10.0):
    """Capture RGB image and sparse pointcloud from one camera position."""
    if up is None:
        # avoid parallel forward/up vectors
        fwd = np.array(target) - np.array(eye)
        if abs(fwd[0]) < 1e-3 and abs(fwd[1]) < 1e-3:
            up = [0, 1, 0]
        else:
            up = [0, 0, 1]

    aspect      = width / height
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    view_matrix = p.computeViewMatrix(eye, target, up)

    img          = p.getCameraImage(width, height,
                                    viewMatrix=view_matrix,
                                    projectionMatrix=proj_matrix,
                                    renderer=p.ER_TINY_RENDERER)
    rgb          = np.array(img[2]).reshape(height, width, 4)[:, :, :3]
    depth_buffer = np.array(img[3]).reshape(height, width)

    proj_np  = np.array(proj_matrix).reshape((4, 4), order='F')
    view_np  = np.array(view_matrix).reshape((4, 4), order='F')
    inv_trans = np.linalg.inv(np.dot(proj_np, view_np))

    pts = []
    for py in range(0, height, sparsity):
        for px in range(0, width, sparsity):
            dv = depth_buffer[py, px]
            if dv < 1.0:
                ndc   = np.array([(2.0*px - width)/width,
                                  -(2.0*py - height)/height,
                                  2.0*dv - 1.0, 1.0])
                world = inv_trans @ ndc
                world /= world[3]
                pts.append(world[:3])

    return rgb, (np.array(pts) if pts else np.zeros((0, 3)))


# ══════════════════════════════════════════════════════════════
# SPHERICAL → CARTESIAN CONVERSION
#
#   Each camera is parameterised in spherical coordinates
#   relative to TARGET:
#     dist      — distance from target (along-axis)
#     azimuth   — angle in XY plane (degrees)  0 = +X axis
#     elevation — angle above XY plane (degrees)
#
#   eye = target + dist * [cos(el)*cos(az), cos(el)*sin(az), sin(el)]
# ══════════════════════════════════════════════════════════════

def spherical_to_eye(dist, azimuth_deg, elevation_deg, target=TARGET):
    az  = np.radians(azimuth_deg)
    el  = np.radians(elevation_deg)
    dx  = dist * np.cos(el) * np.cos(az)
    dy  = dist * np.cos(el) * np.sin(az)
    dz  = dist * np.sin(el)
    return [target[0] + dx, target[1] + dy, target[2] + dz]


def eye_to_spherical(eye, target=TARGET):
    """Convert an XYZ eye position back to (dist, azimuth_deg, elevation_deg)."""
    d    = np.array(eye) - np.array(target)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 5.0, 0.0, 30.0
    az   = float(np.degrees(np.arctan2(d[1], d[0])))
    el   = float(np.degrees(np.arcsin(np.clip(d[2] / dist, -1, 1))))
    return dist, az, el


# ══════════════════════════════════════════════════════════════
# DEFAULT CAMERA PARAMETERS
# ══════════════════════════════════════════════════════════════

DEFAULT_CAMERAS = [
    # horizontal ring (dist, az, el)
    {"dist": 5.0, "az":   0.0, "el":  0.0},
    {"dist": 5.0, "az": 180.0, "el":  0.0},
    {"dist": 5.0, "az":  90.0, "el":  0.0},
    {"dist": 5.0, "az": 270.0, "el":  0.0},
    {"dist": 5.7, "az":  45.0, "el":  0.0},
    {"dist": 5.7, "az": 135.0, "el":  0.0},
    {"dist": 5.7, "az": 315.0, "el":  0.0},
    {"dist": 5.7, "az": 225.0, "el":  0.0},
    # elevated diagonals
    {"dist": 6.7, "az":  45.0, "el": 45.0},
    {"dist": 6.7, "az": 135.0, "el": 45.0},
    {"dist": 6.7, "az": 315.0, "el": 45.0},
    {"dist": 6.7, "az": 225.0, "el": 45.0},
    # top
    {"dist": 8.0, "az":   0.0, "el": 90.0},
]
N_CAMS = len(DEFAULT_CAMERAS)


# ══════════════════════════════════════════════════════════════
# CONFIG SAVE / LOAD
# ══════════════════════════════════════════════════════════════

def save_config(cam_params, path=CONFIG_FILE):
    with open(path, "w") as f:
        json.dump(cam_params, f, indent=2)
    print(f"[INFO] Camera config saved → {path}")


def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded camera config from {path}")
        return cfg
    print(f"[INFO] No config found at {path} — using defaults")
    return None


# ══════════════════════════════════════════════════════════════
# CAMERA ADJUSTER
# ══════════════════════════════════════════════════════════════

def run_camera_adjuster():
    """
    Displays all 13 camera feeds simultaneously in a matplotlib grid.
    PyBullet debug sliders control dist / azimuth / elevation per camera.
    Move a slider → the corresponding camera feed updates live.
    Close the window or press Q to finish and save cameras.json.
    """

    # Load saved config or use defaults
    saved = load_config()
    cam_params = saved if saved else [dict(c) for c in DEFAULT_CAMERAS]

    # ── Create PyBullet debug sliders ─────────────────────────
    # Three sliders per camera: dist, azimuth, elevation
    slider_ids = []
    for i, cp in enumerate(cam_params):
        sid_dist = p.addUserDebugParameter(
            f"Cam{i:02d}_dist",
            rangeMin=1.0, rangeMax=12.0,
            startValue=cp["dist"]
        )
        sid_az = p.addUserDebugParameter(
            f"Cam{i:02d}_azimuth",
            rangeMin=-180.0, rangeMax=180.0,
            startValue=cp["az"]
        )
        sid_el = p.addUserDebugParameter(
            f"Cam{i:02d}_elevation",
            rangeMin=-85.0, rangeMax=85.0,
            startValue=cp["el"]
        )
        slider_ids.append((sid_dist, sid_az, sid_el))

    print(f"\n[INFO] Camera adjuster started.")
    print(f"       Move sliders in the PyBullet GUI to adjust cameras.")
    print(f"       Close the matplotlib window (or press Q) to save and exit.\n")

    # ── Matplotlib grid setup ─────────────────────────────────
    # 13 cameras → 3 rows × 5 cols (last cell empty)
    ROWS, COLS = 3, 5
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle("Camera Adjuster — move sliders in PyBullet GUI",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    gs   = gridspec.GridSpec(ROWS, COLS, figure=fig,
                             hspace=0.08, wspace=0.06,
                             left=0.02, right=0.98,
                             top=0.94, bottom=0.02)
    axes = []
    ims  = []

    for i in range(N_CAMS):
        r, c = divmod(i, COLS)
        ax   = fig.add_subplot(gs[r, c])
        ax.set_facecolor("#1a1a1a")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        # Initial capture
        eye = spherical_to_eye(cam_params[i]["dist"],
                               cam_params[i]["az"],
                               cam_params[i]["el"])
        rgb, _ = capture(eye)
        im = ax.imshow(rgb)
        ax.set_title(f"Cam {i:02d}", color="#aaa", fontsize=8, pad=3)
        axes.append(ax)
        ims.append(im)

    # Hide the unused 15th cell
    if N_CAMS < ROWS * COLS:
        for extra in range(N_CAMS, ROWS * COLS):
            r, c = divmod(extra, COLS)
            fig.add_subplot(gs[r, c]).set_visible(False)

    plt.ion()
    plt.show()

    # ── Live update loop ──────────────────────────────────────
    running = True

    def on_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect("close_event", on_close)

    # Track previous slider values to detect changes
    prev_vals = [(cam_params[i]["dist"],
                  cam_params[i]["az"],
                  cam_params[i]["el"]) for i in range(N_CAMS)]

    while running:
        changed_any = False

        for i, (sid_dist, sid_az, sid_el) in enumerate(slider_ids):
            try:
                dist = p.readUserDebugParameter(sid_dist)
                az   = p.readUserDebugParameter(sid_az)
                el   = p.readUserDebugParameter(sid_el)
            except Exception:
                running = False
                break

            prev = prev_vals[i]
            if (abs(dist - prev[0]) > 0.01 or
                abs(az   - prev[1]) > 0.1  or
                abs(el   - prev[2]) > 0.1):

                # Update stored params
                cam_params[i]["dist"] = dist
                cam_params[i]["az"]   = az
                cam_params[i]["el"]   = el
                prev_vals[i]          = (dist, az, el)

                # Recapture
                eye      = spherical_to_eye(dist, az, el)
                rgb, _   = capture(eye)
                ims[i].set_data(rgb)
                axes[i].set_title(
                    f"Cam {i:02d}  d={dist:.1f} az={az:.0f}° el={el:.0f}°",
                    color="#ccc", fontsize=7, pad=3
                )
                changed_any = True

        if changed_any:
            fig.canvas.draw_idle()

        fig.canvas.flush_events()
        p.stepSimulation()
        time.sleep(0.03)   # ~30 fps

    plt.ioff()

    # ── Save config ───────────────────────────────────────────
    save_config(cam_params)
    print("[INFO] Adjuster closed. Config saved.")
    return cam_params


# ══════════════════════════════════════════════════════════════
# POINTCLOUD GENERATION  (using saved/adjusted cameras)
# ══════════════════════════════════════════════════════════════

def cam_legacy(eye, target=TARGET, sparsity=4,
               width=640, height=480, fov=60, near=0.1, far=10.0):
    """Original cam() function, kept for full-res pointcloud generation."""
    fwd = np.array(target) - np.array(eye)
    up  = [0, 1, 0] if (abs(fwd[0]) < 1e-3 and abs(fwd[1]) < 1e-3) else [0, 0, 1]
    _, pts = capture(eye, target, up, sparsity, width, height, fov, near, far)
    return pts.tolist() if len(pts) else []


def generate_pointcloud(cam_params, sparsity=4):
    """Generate and save pointcloud using the current camera configuration."""
    all_clouds = []
    for i, cp in enumerate(cam_params):
        eye = spherical_to_eye(cp["dist"], cp["az"], cp["el"])
        pts = cam_legacy(eye, sparsity=sparsity)
        all_clouds.append(np.array(pts) if pts else np.zeros((0, 3)))
        print(f"  Cam {i:02d}: {len(pts)} pts  eye={[round(e,2) for e in eye]}")

    points = np.vstack([c for c in all_clouds if len(c) > 0])
    points = remove_floor_ransac(points)
    np.save("points.npy", points)
    print(f"[INFO] Saved {len(points)} points → points.npy")
    return points


def remove_floor_ransac(points, threshold=0.15, iterations=100):
    if len(points) < 100:
        return points
    best_inliers = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        idx    = np.random.choice(len(points), 3, replace=False)
        sample = points[idx]
        v1     = sample[1] - sample[0]
        v2     = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm   = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d       = -np.dot(normal, sample[0])
        dist    = np.abs(points @ normal + d)
        inliers = dist < threshold
        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers.copy()
    clean = points[~best_inliers]
    print(f"[INFO] Floor removal: {len(points)} → {len(clean)} points")
    return clean


def plot(points, squish=0):
    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap='viridis', s=1)
    xl = [points[:, 0].min(), points[:, 0].max()]
    yl = [points[:, 1].min(), points[:, 1].max()]
    zl = [points[:, 2].min(), points[:, 2].max()]
    mr = np.ptp(np.array([xl, yl, zl])).max() / 2.0
    mx, my, mz = np.mean(xl), np.mean(yl), np.mean(zl)
    ax.set_xlim(mx-mr, mx+mr)
    ax.set_ylim(my-mr, my+mr)
    ax.set_zlim(mz-mr+squish, mz+mr-squish)
    plt.show()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if CAMERA_ADJUSTER_MODE:
    # ── Run adjuster, then generate pointcloud from result ────
    cam_params = run_camera_adjuster()
    # Comment out the two lines below to skip pointcloud generation
    points = generate_pointcloud(cam_params, sparsity=7)
    plot(points)

else:
    # ── Skip adjuster: load saved config or use defaults ──────
    saved      = load_config()
    cam_params = saved if saved else [dict(c) for c in DEFAULT_CAMERAS]
    points     = generate_pointcloud(cam_params, sparsity=7)
    plot(points)


p.disconnect()