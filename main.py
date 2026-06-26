import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
VOXEL_SIZE=0.01

# --------------------------------------------------
# Function: Convert LiDAR frame to (N, 3) point cloud
# --------------------------------------------------

def frame_to_pointcloud(frame):
    """
    Converts a LiDAR frame of shape (3, 64, 1024)
    into a point cloud of shape (N, 3).
    Also removes invalid (0,0,0) points.

    Returns:
        points          : Valid point cloud of shape (N, 3)
        total_points    : Total possible points before filtering
        removed_points  : Number of invalid (0,0,0) points removed
    """

    x = frame[0].flatten()
    y = frame[1].flatten()
    z = frame[2].flatten()

    # Shape becomes (65536, 3)
    points = np.stack((x, y, z), axis=1)

    total_points = points.shape[0]

    # Remove invalid points
    mask = ~(np.all(points == 0, axis=1))
    points = points[mask]

    removed_points = total_points - points.shape[0]

    return points, total_points, removed_points


# --------------------------------------------------
# Function: Visualize one or more point clouds
# --------------------------------------------------

def plot_point_cloud(point_clouds, colors, labels, title):

    fig = plt.figure(figsize=(12, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")

    # Black background
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Remove gray panes
    ax.xaxis.set_pane_color((0, 0, 0, 1))
    ax.yaxis.set_pane_color((0, 0, 0, 1))
    ax.zaxis.set_pane_color((0, 0, 0, 1))

    # Plot each cloud
    for pts, color, label in zip(point_clouds, colors, labels):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=color,
            s=0.08,
            marker=".",
            alpha=0.9,
            linewidths=0,
            label=label
        )

    # Compute limits using all point clouds
    combined = np.vstack(point_clouds)
    limit = np.percentile(np.abs(combined), 99)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-5, 5)

    ax.set_xlabel("X", color="white", fontsize=12)
    ax.set_ylabel("Y", color="white", fontsize=12)
    ax.set_zlabel("Z", color="white", fontsize=12)

    ax.tick_params(colors="white")
    ax.grid(False)

    ax.view_init(elev=20, azim=-60)


    ax.set_title(
        title,
        color="white",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    legend = ax.legend(
    facecolor="black",
    edgecolor="white",
    markerscale=40
)

    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.show()

'''def voxel_downsample(points, voxel_size=VOXEL_SIZE):
    """
    Downsample a point cloud using voxel grid filtering.

    Each voxel is represented by the centroid (mean)
    of all points that fall inside it.

    Parameters:
        points (numpy.ndarray): Input point cloud of shape (N, 3)
        voxel_size (float): Side length of each voxel (in meters)

    Returns:
        numpy.ndarray: Downsampled point cloud
    """

    # Compute voxel index for every point
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Dictionary to store running sums and counts
    voxel_dict = {}

    for voxel, point in zip(map(tuple, voxel_indices), points):

        if voxel not in voxel_dict:
            voxel_dict[voxel] = {
                "sum": np.zeros(3, dtype=np.float64),
                "count": 0
            }

        voxel_dict[voxel]["sum"] += point
        voxel_dict[voxel]["count"] += 1

    # Compute centroid of each voxel
    downsampled_points = []

    for voxel_data in voxel_dict.values():
        centroid = voxel_data["sum"] / voxel_data["count"]
        downsampled_points.append(centroid)

    return np.array(downsampled_points)'''

def find_nearest_neighbors(source_points, target_points):
    """
    For every point in source_points, find the closest point
    in target_points.

    Parameters:
        source_points : (N, 3) NumPy array
        target_points : (M, 3) NumPy array

    Returns:
        matched_target_points : (N, 3)
            Closest point in target for each source point.

        distances : (N,)
            Euclidean distance to the nearest neighbor.

        indices : (N,)
            Index of the matched point in target_points.
    """

    # Build KD-Tree on the target cloud
    tree = cKDTree(target_points)

    # Query nearest neighbor for every source point
    distances, indices = tree.query(source_points)

    # Retrieve matched target points
    matched_target_points = target_points[indices]

    return matched_target_points, distances, indices


def estimate_rigid_transform(source_points, target_points):
    """
    Estimate the optimal rigid transformation (rotation + translation)
    that aligns source_points to target_points using SVD.

    Parameters:
        source_points : (N, 3)
        target_points : (N, 3)

    Returns:
        R : (3, 3) Rotation matrix
        t : (3,)   Translation vector
    """

    # --------------------------------------------------
    # Step 1: Compute centroids
    # --------------------------------------------------
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # --------------------------------------------------
    # Step 2: Center the point clouds
    # --------------------------------------------------
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # --------------------------------------------------
    # Step 3: Compute covariance matrix
    # --------------------------------------------------
    H = source_centered.T @ target_centered

    # --------------------------------------------------
    # Step 4: Singular Value Decomposition
    # --------------------------------------------------
    U, S, Vt = np.linalg.svd(H)

    # --------------------------------------------------
    # Step 5: Compute rotation matrix
    # --------------------------------------------------
    R = Vt.T @ U.T

    # Correct improper rotation (reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # --------------------------------------------------
    # Step 6: Compute translation vector
    # --------------------------------------------------
    t = centroid_target - R @ centroid_source

    return R, t
# --------------------------------------------------
# Function: Apply rigid transformation
# --------------------------------------------------

def apply_transformation(points, R, t):
    """
    Apply the rigid transformation:
        p' = R @ p + t

    Parameters:
        points : (N, 3) point cloud
        R      : (3, 3) rotation matrix
        t      : (3,) translation vector

    Returns:
        transformed_points : (N, 3)
    """

    transformed_points = (R @ points.T).T + t
    return transformed_points

# --------------------------------------------------
# Function: Iterative Closest Point (ICP)
# --------------------------------------------------

def icp(source_points,
        target_points,
        max_iterations=20,
        tolerance=1e-6):
    """
    Align source_points to target_points using the
    Iterative Closest Point (ICP) algorithm.

    Parameters
    ----------
    source_points : (N, 3) ndarray
        Source point cloud.

    target_points : (M, 3) ndarray
        Target point cloud.

    max_iterations : int
        Maximum ICP iterations.

    tolerance : float
        Stop if improvement in mean error is below this value.

    Returns
    -------
    aligned_points : (N, 3)
        Source cloud after ICP alignment.

    R_total : (3, 3)
        Overall rotation matrix.

    t_total : (3,)
        Overall translation vector.

    errors : list
        Mean nearest-neighbor error at each iteration.
    """

    # Work on a copy so the original is unchanged
    aligned_points = source_points.copy()

    # Overall transformation
    R_total = np.eye(3)
    t_total = np.zeros(3)

    errors = []
    previous_error = np.inf

    for iteration in range(max_iterations):

        # ------------------------------------------
        # Step 1: Find nearest neighbors
        # ------------------------------------------
        matched_points, distances, _ = find_nearest_neighbors(
            aligned_points,
            target_points
        )

        mean_error = np.mean(distances)
        errors.append(mean_error)

        print(
            f"Iteration {iteration + 1:2d} | "
            f"Mean Error = {mean_error:.8f}"
        )

        # ------------------------------------------
        # Step 2: Estimate rigid transform
        # ------------------------------------------
        R, t = estimate_rigid_transform(
            aligned_points,
            matched_points
        )

        # ------------------------------------------
        # Step 3: Apply transformation
        # ------------------------------------------
        aligned_points = apply_transformation(
            aligned_points,
            R,
            t
        )

        # ------------------------------------------
        # Step 4: Accumulate total transform
        # ------------------------------------------
        R_total = R @ R_total
        t_total = R @ t_total + t

        # ------------------------------------------
        # Step 5: Check convergence
        # ------------------------------------------
        improvement = previous_error - mean_error

        if improvement >= 0 and improvement < tolerance:
            print("\nICP converged.")
            break

        previous_error = mean_error

    return aligned_points, R_total, t_total, errors
# --------------------------------------------------
# Step 1: Load the dataset
# --------------------------------------------------

data = np.load("data/static/7.npy", allow_pickle=True)

# --------------------------------------------------
# Step 2: Print dataset information
# --------------------------------------------------

num_frames = data.shape[0]
num_channels = data.shape[1]
num_beams = data.shape[2]
num_angles = data.shape[3]

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Type                : {type(data)}")
print(f"Data Type           : {data.dtype}")
print(f"Overall Shape       : {data.shape}")
print(f"Number of Frames    : {num_frames}")
print(f"Coordinate Channels : {num_channels} (x, y, z)")
print(f"Laser Beams         : {num_beams}")
print(f"Horizontal Samples  : {num_angles}")


# --------------------------------------------------
# Step 3: Extract two LiDAR frames
# --------------------------------------------------

frame0 = data[0]
frame1 = data[10]


print("\nFrame Shape :", frame0.shape)

# --------------------------------------------------
# Step 4: Convert to (N, 3) point clouds
# --------------------------------------------------

points0, total0, removed0 = frame_to_pointcloud(frame0)
points1, total1, removed1 = frame_to_pointcloud(frame1)

print("\n" + "=" * 60)
print("POINT CLOUD CONVERSION")
print("=" * 60)
print(f"Original frame shape              : {frame0.shape}")
print(f"Maximum possible points           : {total0} (64 × 1024)")
print(f"Invalid (0,0,0) points removed    : {removed0}")
print(f"Valid points after filtering      : {points0.shape[0]}")
invalid_percentage = 100 * removed0 / total0
print(f"Invalid points removed : {invalid_percentage:.2f}%")

# --------------------------------------------------
# Step 5: Visualize Frame 0
# --------------------------------------------------

plot_point_cloud(
    point_clouds=[points0],
    colors=["#ff1493"],       # Neon pink
    labels=["Frame 0"],
    title="KITTI Sequence 07 - Frame 0 Point Cloud"
)
plot_point_cloud(
    point_clouds=[points1],
    colors=["#00ffff"],       #Cyan
    labels=["Frame 1"],
    title="KITTI Sequence 07 - Frame 1 Point Cloud"
)

# --------------------------------------------------
# Step 6: Visualize Frame 0 and Frame 1 together
# --------------------------------------------------

plot_point_cloud(
    point_clouds=[points0, points1],
    colors=["#ff1493", "#00ffff"],   # Pink and Cyan
    labels=["Frame 0", "Frame 1"],
    title="Overlay of Consecutive LiDAR Frames"
)
#------------------------
#Voxel Downsample
#-------------------------

'''original_points0 = points0.shape[0]
original_points1 = points1.shape[0]

points0 = voxel_downsample(points0)
points1 = voxel_downsample(points1)

reduction0 = 100 * (1 - points0.shape[0] / original_points0)

print("\n" + "=" * 60)
print("VOXEL DOWNSAMPLING")
print("=" * 60)
print(f"Voxel size                        : {VOXEL_SIZE} m")
print(f"Points before downsampling        : {original_points0}")
print(f"Points after downsampling         : {points0.shape[0]}")
print(f"Reduction                         : {reduction0:.2f}%")
'''
# --------------------------------------------------
# Function: Find nearest neighbors using KD-Tree
# --------------------------------------------------


matched_points, distances, indices = find_nearest_neighbors(
    points0,
    points1
)

print("\n" + "=" * 60)
print("KD-TREE NEAREST NEIGHBOR MATCHING")
print("=" * 60)

print(f"Total correspondences             : {len(indices)}")
print(f"Average distance                  : {np.mean(distances):.4f}")
print(f"Median distance                   : {np.median(distances):.4f}")
print(f"Minimum distance                  : {np.min(distances):.4f}")
print(f"Maximum distance                  : {np.max(distances):.4f}")

print("\nSample Correspondences")
print("-" * 60)
print(f"{'Source Index':<15}{'Target Index':<15}{'Distance'}")

for i in range(5):
    print(f"{i:<15}{indices[i]:<15}{distances[i]:.4f}")

R, t = estimate_rigid_transform(
    points0,
    matched_points
)

print("\n" + "=" * 60)
print("RIGID TRANSFORMATION ESTIMATION")
print("=" * 60)

print("\nRotation Matrix (R):")
print(R)

print("\nTranslation Vector (t):")
print(t)

# --------------------------------------------------
# Apply transformation
# --------------------------------------------------

transformed_points0 = apply_transformation(
    points0,
    R,
    t
)



# --------------------------------------------------
# Run ICP
# --------------------------------------------------

aligned_points, R_final, t_final, errors = icp(
    points0,
    points1,
    max_iterations=20,
    tolerance=1e-6
)

print("\n" + "=" * 60)
print("FINAL ICP RESULT")
print("=" * 60)

print("\nFinal Rotation Matrix:")
print(R_final)

print("\nFinal Translation Vector:")
print(t_final)

print(f"\nFinal Mean Error: {errors[-1]:.8f}")

plot_point_cloud(
    point_clouds=[aligned_points, points1],
    colors=["#ff1493", "#00ffff"],
    labels=["ICP Aligned Frame 0", "Frame 1"],
    title="Final ICP Alignment"
)
# ==========================================================
# ICP ODOMETRY OVER FIRST 10 FRAMES
# ==========================================================

NUM_FRAMES = 10

print("\n" + "=" * 60)
print("RUNNING ICP ODOMETRY ON FIRST 10 FRAMES")
print("=" * 60)

# Global pose (world coordinates)
global_R = np.eye(3)
global_t = np.zeros(3)

# Store estimated positions
estimated_trajectory = [global_t.copy()]

for i in range(NUM_FRAMES - 1):

    print("\n" + "-" * 50)
    print(f"Processing Frame {i} -> Frame {i+1}")
    print("-" * 50)

    # Load consecutive frames
    source_frame = data[i]
    target_frame = data[i + 1]

    # Convert to point clouds
    source_points, _, _ = frame_to_pointcloud(source_frame)
    target_points, _, _ = frame_to_pointcloud(target_frame)

    # Run ICP
    aligned_points, R_step, t_step, errors = icp(
        source_points,
        target_points,
        max_iterations=20,
        tolerance=1e-6
    )

    # Update global pose
    global_t = global_t + global_R @ t_step
    global_R = global_R @ R_step

    estimated_trajectory.append(global_t.copy())

    print(f"Final ICP Error : {errors[-1]:.6f}")
    print(f"Translation     : {t_step}")

estimated_trajectory = np.array(estimated_trajectory)

print("\nEstimated trajectory shape:",
      estimated_trajectory.shape)

plt.figure(figsize=(8, 6))

plt.plot(
    estimated_trajectory[:, 0],
    estimated_trajectory[:, 1],
    marker="o",
    linewidth=2,
    label="Estimated ICP Path"
)

plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Estimated Vehicle Trajectory (First 10 Frames)")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()

# ==========================================================
# LOAD GROUND TRUTH
# ==========================================================

ground_truth = np.loadtxt("07.txt")

# Each row is a 3x4 pose matrix
gt_positions = ground_truth[:, [3, 7, 11]]

# Keep only first 10 poses
gt_positions = gt_positions[:NUM_FRAMES]

plt.figure(figsize=(8, 6))

plt.plot(
    estimated_trajectory[:, 0],
    estimated_trajectory[:, 1],
    marker="o",
    linewidth=2,
    label="ICP Estimated"
)

plt.plot(
    gt_positions[:, 0],
    gt_positions[:, 2],      # KITTI typically uses X-Z for top view
    marker="s",
    linewidth=2,
    label="Ground Truth"
)

plt.xlabel("X")
plt.ylabel("Z")
plt.title("ICP vs Ground Truth (First 10 Frames)")
plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()

print(estimated_trajectory)
print(gt_positions[:10])