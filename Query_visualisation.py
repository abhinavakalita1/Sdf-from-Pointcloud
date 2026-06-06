from sys import exception

import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import pybullet as p
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from dataclasses import dataclass, field
from typing import Optional
import json
import os


# ══════════════════════════════════════════════════════════════
# 1. LOADING ENV + HELPERS
# ══════════════════════════════════════════════════════════════

physicsClient    = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId          = p.loadURDF("plane.urdf")

armId = p.loadURDF(
    "arm_5.urdf", [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=(p.URDF_USE_INERTIA_FROM_FILE |
           p.URDF_USE_SELF_COLLISION     |
           p.URDF_USE_IMPLICIT_CYLINDER)
)
NUM_JOINTS = p.getNumJoints(armId)
print(f"[INFO] Arm loaded — {NUM_JOINTS} joints")


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


def draw_line(point_a, point_b):
    # Draw the line
    line_id = p.addUserDebugLine(
        lineFromXYZ=point_a,
        lineToXYZ=point_b,
        lineColorRGB=[1, 0, 0],  # Red (R, G, B) in range 0-1
        lineWidth=2,
        lifeTime=0  # 0 = permanent, >0 = seconds
    )


# ══════════════════════════════════════════════════════════════
# 2. GROUND TRUTH
# ══════════════════════════════════════════════════════════════


def set_config(arm_id: int, config) -> None:
    for j, angle in enumerate(config):
        p.resetJointState(arm_id, j, float(angle))


def gt_min_distance(arm_id: int, obs_ids: list, threshold=10.0) -> float:
    """Ground truth via p.getClosestPoints."""
    min_d = threshold
    p.removeAllUserDebugItems()
    for obs_id in obs_ids:
        contacts = p.getClosestPoints(bodyA=arm_id, bodyB=obs_id, distance=threshold)
        if contacts:
            l = [c[8] for c in contacts]
            d = min(l)
            i = l.index(d)
            draw_line(contacts[i][5], contacts[i][6])

            if d < min_d:
                min_d = d
    return min_d


def query_gt(N_TOTAL, sleep):
    INTERVAL = 1000

    concaveId = load_mesh_obstacle("concave.obj", position=[0, 0.5, 0], scale=0.4)

    obstacle_ids = [concaveId]
    obstacle_names = ["Concave"]

    print(f"\n[INFO] Sampling {N_TOTAL} random configs…\n")

    configs   = np.random.uniform(-np.pi/2, np.pi/2, size=(N_TOTAL, NUM_JOINTS))
    gt_dists  = np.zeros(N_TOTAL)

    t_gt  = 0.0

    for i, cfg in enumerate(configs):
        set_config(armId, cfg)
        p.stepSimulation()
        print(gt_min_distance(armId, obstacle_ids))
        time.sleep(sleep)


# ══════════════════════════════════════════════════════════════
# 2. LOAD POINTS → CLUSTER → BUILD BVH
# ══════════════════════════════════════════════════════════════

def create_box(half_extents=[1,1,1], position=[0,0,0], orientation=[0,0,0], mass=0, color=[1,0,0,1]):
    col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis  = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
    quat = p.getQuaternionFromEuler(orientation)
    return p.createMultiBody(mass, col, vis, position, quat)



bvh_list    = []
bvh_obsid = []



@dataclass
class BVHNode:
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    left:   Optional["BVHNode"] = field(default=None, repr=False)
    right:  Optional["BVHNode"] = field(default=None, repr=False)
    points: Optional[np.ndarray] = field(default=None, repr=False)


def build_bvh(pts: np.ndarray, leaf_size: int = 8) -> BVHNode:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    node   = BVHNode(aabb_min=lo, aabb_max=hi)
    if len(pts) <= leaf_size:
        node.points = pts
        return node
    axis   = int(np.argmax(hi - lo))
    median = np.median(pts[:, axis])
    lmask  = pts[:, axis] <= median
    lpts, rpts = pts[lmask], pts[~lmask]
    if len(lpts) == 0 or len(rpts) == 0:
        node.points = pts
        return node
    node.left  = build_bvh(lpts, leaf_size)
    node.right = build_bvh(rpts, leaf_size)
    return node


def draw_bvh_boxes(node):
    if node is None:
        return

    size = node.aabb_max - node.aabb_min
    center = (node.aabb_max + node.aabb_min) / 2

    bvh_obsid.append(create_box(size.tolist(), center.tolist()))

    draw_bvh_boxes(node.left)
    draw_bvh_boxes(node.right)


def draw_leaf_boxes(node):
    if node is None:
        return

    if node.left is None and node.right is None:
        size = node.aabb_max - node.aabb_min
        center = (node.aabb_max + node.aabb_min) / 2

        bvh_obsid.append(create_box(size.tolist(), center.tolist()))
        bvh_list.append(node)
        return

    draw_leaf_boxes(node.left)
    draw_leaf_boxes(node.right)


def _pt_aabb_sqdist(pt, lo, hi) -> float:
    return float(np.sum((pt - np.clip(pt, lo, hi)) ** 2))


def _bvh_nearest_sq(pt: np.ndarray, node: BVHNode, best: list) -> None:
    if _pt_aabb_sqdist(pt, node.aabb_min, node.aabb_max) >= best[0][0]:
        return
    if node.points is not None:
        sq  = ((node.points - pt) ** 2).sum(axis=1)
        idx = int(np.argmin(sq))
        if sq[idx] < best[0][0]:
            best[0] = (float(sq[idx]), node.points[idx].copy())
        return
    l_sq = _pt_aabb_sqdist(pt, node.left.aabb_min,  node.left.aabb_max)
    r_sq = _pt_aabb_sqdist(pt, node.right.aabb_min, node.right.aabb_max)
    first, second = (node.left, node.right) if l_sq <= r_sq else (node.right, node.left)
    _bvh_nearest_sq(pt, first,  best)
    _bvh_nearest_sq(pt, second, best)


def sdf_single(pt: np.ndarray, bvh: BVHNode) -> float:
    """
    Signed distance to one obstacle BVH.
    Sign: root AABB membership (inside AABB → negative).
    """
    best   = [(np.inf, None)]
    _bvh_nearest_sq(pt, bvh, best)
    dist   = float(np.sqrt(best[0][0]))
    inside = np.all(pt >= bvh.aabb_min) and np.all(pt <= bvh.aabb_max)
    return -dist if inside else dist


def sdf_scene(pt: np.ndarray) -> list:
    """Min signed distance over all obstacle BVHs."""
    l = [sdf_single(pt, bvh) for bvh in bvh_list]
    d = min(l)
    i = l.index(d)
    b = bvh_list[i]
    return [d, b]


CONFIG_FILE = "dbscan_config.json"

def load_config(path=CONFIG_FILE):
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded config from {path}")
        return cfg
    print(f"[INFO] No config found — using defaults")
    return None

params      = load_config()
eps         = params["eps"]         if params else 0.082
min_samples = params["min_samples"] if params else 3


def group(points, labels, min_points_threshold=100):
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_sizes  = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl, cnt in cluster_sizes.items()
                      if lbl != -1 and cnt < min_points_threshold]
    for small_lbl in small_clusters:
        small_mask   = (labels == small_lbl)
        small_points = points[small_mask]
        distances    = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1:
                continue
            centroid = points[labels == lbl].mean(axis=0)
            dist     = np.mean(np.linalg.norm(small_points - centroid, axis=1))
            distances.append((lbl, dist))
        if distances:
            closest_lbl = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest_lbl
            print(f"  Merged cluster {small_lbl} ({len(small_points)} pts) "
                  f"→ cluster {closest_lbl}")
    return labels


def find_clusters(points, e=eps, ms=min_samples):
    db     = DBSCAN(eps=e, min_samples=ms).fit(points)
    labels = db.labels_.copy()
    labels = group(points, labels)
    return labels

def get_link_positions(arm_id: int) -> dict:
    base_pos, _ = p.getBasePositionAndOrientation(arm_id)
    pos = {-1: np.array(base_pos)}
    for j in range(p.getNumJoints(arm_id)):
        state  = p.getLinkState(arm_id, j, computeForwardKinematics=True)
        pos[j] = np.array(state[4])
    return pos


# def bvh_min_distance(arm_id: int) -> float:
#     """Min SDF over all links using AABB-BVH."""
#     l = [sdf_scene(pos) for pos in get_link_positions(arm_id).values()]
#     foo = [x[0] for x in l]
#     d = min(foo)
#     i = foo.index(d)
#     b = l[i][1]
#     try:
#         draw_line((b.aabb_min + b.aabb_max) / 2, get_link_positions(arm_id)[i])
#     except exception as e:
#         pass
#     return d


def bvh_min_distance(arm_id: int) -> float:
    """Min SDF over all links using AABB-BVH."""
    threshold = 10
    contacts = []
    min_d = threshold
    for obs_id in bvh_obsid:
        contacts.extend(p.getClosestPoints(bodyA=arm_id, bodyB=obs_id, distance=threshold))
    if contacts:
        l = [c[8] for c in contacts]
        d = min(l)
        i = l.index(d)
        draw_line(contacts[i][5], contacts[i][6])
        if d < min_d:
            min_d = d
    # return min(sdf_scene(pos) for pos in get_link_positions(arm_id).values())
    return min_d


def query_bvh(N_TOTAL, sleep, leaf):
    points = np.load("points.npy")
    print(f"[INFO] Loaded {len(points)} points")

    labels        = find_clusters(points)
    unique_labels = np.unique(labels[labels >= 0])
    print(f"[INFO] {len(unique_labels)} clusters after merging")

    cluster_pts = []

    for lbl in unique_labels:
        cpts = points[labels == lbl]
        cluster_pts.append(cpts)
        bvh  = build_bvh(cpts, leaf_size=leaf)
        # bvh_list.append(bvh)
        print(f"  Cluster {lbl}: {len(cpts)} pts | "
              f"AABB {bvh.aabb_min.round(3)} → {bvh.aabb_max.round(3)}")

        draw_leaf_boxes(bvh)

    INTERVAL = 1000

    print(f"\n[INFO] Sampling {N_TOTAL} random configs…\n")

    configs   = np.random.uniform(-np.pi/2, np.pi/2, size=(N_TOTAL, NUM_JOINTS))
    bvh_dists = np.zeros(N_TOTAL)

    for i, cfg in enumerate(configs):
        set_config(armId, cfg)
        p.stepSimulation()
        print(bvh_min_distance(armId))
        time.sleep(sleep)


query_bvh(20, 3, 8)