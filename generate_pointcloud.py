import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Connect to physics server
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load environment
plane = p.loadURDF("plane.urdf")

def create_box(half_extents=[1, 1, 1], position=[0, 0, 0], orientation=[0, 0, 0], mass=0, color=[1, 0, 0, 1]):
    collision = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    quat = p.getQuaternionFromEuler(orientation)

    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=quat
    )

    return body

def create_sphere(radius=0.5, position=[0, 0, 0], orientation=[0, 0, 0], mass=0, color=[0, 1, 0, 1]):
    collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius
    )

    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )

    quat = p.getQuaternionFromEuler(orientation)

    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=quat
    )

    return body

def create_cylinder(radius=0.5, height=1.0, position=[0, 0, 0], orientation=[0, 0, 0], mass=0, color=[0, 0, 1, 1]):
    collision = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    quat = p.getQuaternionFromEuler(orientation)

    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=quat
    )

    return body


boxId = create_box(half_extents=[1,1,1], position=[2,0,1], orientation=[0.2,1.1,0.4])

sphereId = create_sphere(radius=1, position=[0,2,1])

cylinderId = create_cylinder(radius=0.3, height=2, position=[-0.5,0,1], orientation=[1.3,0,0])

def cam(cameraEyePosition, cameraTargetPosition, sparsity=10, cameraVector=[0,0,1], width=640, height=480, fov=60, near=0.1, far=10.0):

    # Compute view and projection matrices
    aspect = width / height
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    view_matrix = p.computeViewMatrix(cameraEyePosition=cameraEyePosition,
                                      cameraTargetPosition=cameraTargetPosition,
                                      cameraUpVector=cameraVector)

    # Get depth image
    img = p.getCameraImage(width, height,
                           viewMatrix=view_matrix,
                           projectionMatrix=proj_matrix,
                           renderer=p.ER_TINY_RENDERER)
    depth_buffer = np.array(img[3]).reshape(height, width)

    # Convert matrices to numpy and compute inverse of combined matrix
    proj_np = np.array(proj_matrix).reshape((4, 4), order='F')
    view_np = np.array(view_matrix).reshape((4, 4), order='F')
    inv_trans = np.linalg.inv(np.dot(proj_np, view_np))

    # Generate point cloud
    point_cloud = []
    for y in range(0, height, sparsity):
        for x in range(0, width, sparsity):
            depth_value = depth_buffer[y, x]
            if depth_value < 1.0:  # Valid depth (not sky)
                # Convert to NDC (Normalized Device Coordinates)
                ndc_x = (2.0 * x - width) / width
                ndc_y = -(2.0 * y - height) / height
                ndc_z = 2.0 * depth_value - 1.0
                ndc = np.array([ndc_x, ndc_y, ndc_z, 1.0])

                # Transform to world coordinates
                world = np.dot(inv_trans, ndc)
                world /= world[3]  # Perspective divide
                point_cloud.append(world[:3])

    return point_cloud


clouds = []

cams = [

    # horizontal ring
    [ 5,  0, 1.5],
    [-5,  0, 1.5],
    [ 0,  5, 1.5],
    [ 0, -5, 1.5],

    [ 4,  4, 1.5],
    [-4,  4, 1.5],
    [ 4, -4, 1.5],
    [-4, -4, 1.5],

    # elevated diagonals
    [ 3,  3, 5],
    [-3,  3, 5],
    [ 3, -3, 5],
    [-3, -3, 5],

    # top
    [0,0,8]
]

target = [0.75, 1.0, 1.0]

all_clouds = []

for eye in cams:

    # avoid parallel forward/up vectors
    if eye[0] == target[0] and eye[1] == target[1]:
        up = [0,1,0]
    else:
        up = [0,0,1]

    cloud = cam(
        cameraEyePosition=eye,
        cameraTargetPosition=target,
        cameraVector=up,
        sparsity=4
    )

    all_clouds.append(np.array(cloud))


def remove_floor_ransac(points: np.ndarray, threshold=0.08, iterations=100):
    """Remove dominant floor plane using RANSAC"""
    if len(points) < 100:
        return points

    best_inliers = np.zeros(len(points), dtype=bool)

    for _ in range(iterations):
        # Sample 3 random points
        idx = np.random.choice(len(points), 3, replace=False)
        sample = points[idx]

        vec1 = sample[1] - sample[0]
        vec2 = sample[2] - sample[0]
        normal = np.cross(vec1, vec2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        d = -np.dot(normal, sample[0])

        # Distance to plane
        dist = np.abs(points @ normal + d)
        inliers = dist < threshold

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers.copy()

    clean_points = points[~best_inliers]
    print(f"Floor removal → {len(points)} → {len(clean_points)} points kept")
    return clean_points


# Usage:
points = np.vstack(all_clouds)
points = remove_floor_ransac(points, threshold=0.08)

def plot(points, squish=0):
    # Plot in Matplotlib
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # After creating the scatter plot
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)

    # Calculate the range of your data
    x_limits = [points[:, 0].min(), points[:, 0].max()]
    y_limits = [points[:, 1].min(), points[:, 1].max()]
    z_limits = [points[:, 2].min(), points[:, 2].max()]

    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.ptp(np.array([x_limits, y_limits, z_limits])).max() / 2.0
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range + squish, mid_z + max_range-squish)
    plt.show()

plot(points, squish=0)

np.save("points.npy", points)

# Disconnect
p.disconnect()