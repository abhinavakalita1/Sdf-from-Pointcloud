import pybullet_data
import numpy as np
from sklearn.cluster import DBSCAN
import pybullet as p
import matplotlib.pyplot as plt
import time

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
startPos = [0,0,0]
startOrientation = p.getQuaternionFromEuler([0,0,0])

armId = p.loadURDF(
    "arm_5.urdf",
    startPos,
    startOrientation,
    useFixedBase=True,
    flags=
        p.URDF_USE_INERTIA_FROM_FILE |
        p.URDF_USE_SELF_COLLISION |
        p.URDF_USE_IMPLICIT_CYLINDER

)

points = np.load("points.npy")

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


db = DBSCAN(eps=0.082, min_samples=3).fit(points)
labels = db.labels_
unique_labels, counts = np.unique(labels, return_counts=True)

def group(unique_labels, counts, min_points_threshold=100):
    cluster_sizes = dict(zip(unique_labels, counts))
    small_clusters = [lbl for lbl, cnt in cluster_sizes.items()
                      if lbl != -1 and cnt < min_points_threshold]

    # For each small cluster, assign its points to the nearest big cluster
    for small_lbl in small_clusters:
        small_mask = (labels == small_lbl)
        small_points = points[small_mask]

        # Find distance to all other clusters' centroids
        distances = []
        for lbl in unique_labels:
            if lbl == small_lbl or lbl == -1:
                continue
            cluster_pts = points[labels == lbl]
            centroid = cluster_pts.mean(axis=0)
            dist_to_centroid = np.mean(np.linalg.norm(small_points - centroid, axis=1))
            distances.append((lbl, dist_to_centroid))

        if distances:
            # Assign to closest cluster
            closest_lbl = min(distances, key=lambda x: x[1])[0]
            labels[small_mask] = closest_lbl
            print(f"Merged cluster {small_lbl} ({len(small_points)} pts) → into cluster {closest_lbl}")


group(unique_labels, counts)
mask = labels == 0 # Change this to view clusters
points = points[mask]
def plot_pointcloud_in_pybullet(points: np.ndarray,
                                point_size=3.0,
                                color=[1, 0, 0]):
    """
    Plot raw point cloud in PyBullet using debug points
    """
    if len(points) == 0:
        print("No points to plot")
        return

    # Convert to list of points with color
    point_colors = [color] * len(points)

    point_ids = p.addUserDebugPoints(
        pointPositions=points.tolist(),
        pointColorsRGB=point_colors,
        pointSize=point_size
    )

    print(f"Plotted {len(points)} points in PyBullet")
    return point_ids  # You can use this to remove them later if needed

# Plot the raw point cloud
debug_point_ids = plot_pointcloud_in_pybullet(
    points,
    point_size=4.0,
    color=[1, 0.2, 0.0]   # Orange-red
)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)


p.disconnect()
