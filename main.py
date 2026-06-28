#!/usr/bin/env python3
"""
KITTI LiDAR Odometry using ICP (from scratch) – Complete Project
Sequence 07, frames 0-9.
Plots: 3D point clouds (black background) and 2D trajectory (bird's-eye).
Includes calibration to align estimated trajectory with ground truth.
Adds constant‑velocity initial guess to ICP for better convergence.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.linalg import svd
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# 1. I/O Utilities
# ----------------------------------------------------------------------
def load_velodyne_scan(file_path):
    """Load KITTI velodyne .bin file -> Nx3 (x,y,z), removing invalid (0,0,0) points."""
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    # Keep only x, y, z
    points = points[:, :3]
    # Remove points where all coordinates are exactly zero
    mask = ~(np.all(points == 0, axis=1))
    points = points[mask]
    return points


def load_ground_truth_poses(file_path):
    """Load KITTI ground truth poses (3x4 per line) -> list of 4x4."""
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(T)
    return poses

def load_calibration(file_path):
    """Load KITTI calibration file and extract Tr (lidar->camera)."""
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Tr:'):
                parts = line.strip().split()
                vals = list(map(float, parts[1:]))
                Tr = np.eye(4)
                Tr[:3, :] = np.array(vals).reshape(3, 4)
                return Tr
    return np.eye(4)

# ----------------------------------------------------------------------
# 2. Point Cloud Preprocessing
# ----------------------------------------------------------------------
def downsample_voxel(points, voxel_size):
    """Voxel grid downsampling."""
    if voxel_size <= 0 or len(points) == 0:
        return points
    indices = np.floor(points / voxel_size).astype(np.int32)
    voxel_dict = {}
    for i, idx in enumerate(indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    downsampled = [np.mean(pts, axis=0) for pts in voxel_dict.values()]
    return np.array(downsampled)

def estimate_normals(pts, k=20):
    """Estimate normals for each point using PCA on k neighbours."""
    tree = cKDTree(pts)
    normals = np.zeros_like(pts)
    for i, p in enumerate(pts):
        idx = tree.query(p, k=k+1)[1]
        neighbours = pts[idx]
        cov = np.cov(neighbours.T)
        _, _, v = svd(cov)
        normals[i] = v[:, -1]
    return normals

# ----------------------------------------------------------------------
# 3. ICP Core Functions
# ----------------------------------------------------------------------
def estimate_transform_svd(src_pts, tgt_pts):
    """Estimate rigid transform (R, t) via SVD."""
    src_center = np.mean(src_pts, axis=0)
    tgt_center = np.mean(tgt_pts, axis=0)
    src_centered = src_pts - src_center
    tgt_centered = tgt_pts - tgt_center
    H = src_centered.T @ tgt_centered
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_center - R @ src_center
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def icp_point_to_point(source, target, max_iterations=50, tolerance=1e-6,
                       distance_threshold=None, voxel_size=0.2, return_aligned=False):
    """Point‑to‑point ICP."""
    src = source.copy()
    tgt = target.copy()
    T_total = np.eye(4)
    if voxel_size > 0:
        src = downsample_voxel(src, voxel_size)
        tgt = downsample_voxel(tgt, voxel_size)
    prev_error = float('inf')
    for i in range(max_iterations):
        tree = cKDTree(tgt)
        distances, indices = tree.query(src)
        if distance_threshold is None:
            thresh = 3.0 * np.median(distances)
        else:
            thresh = distance_threshold
        valid = distances <= thresh
        src_matched = src[valid]
        tgt_matched = tgt[indices[valid]]
        if len(src_matched) < 10:
            print(f"ICP-P2P iter {i}: too few inliers, stopping.")
            break
        T = estimate_transform_svd(src_matched, tgt_matched)
        src_h = np.hstack((src, np.ones((src.shape[0], 1))))
        src = (T @ src_h.T).T[:, :3]
        T_total = T @ T_total
        if i > 0:
            mean_error = np.mean(distances[valid])
            if abs(prev_error - mean_error) < tolerance:
                print(f"ICP-P2P converged at iter {i}, error={mean_error:.6f}")
                break
            prev_error = mean_error
    if return_aligned:
        src_h = np.hstack((source, np.ones((source.shape[0], 1))))
        aligned = (T_total @ src_h.T).T[:, :3]
        return T_total, aligned
    return T_total

def icp_point_to_plane(source, target, max_iterations=50, tolerance=1e-6,
                       distance_threshold=None, voxel_size=0.2, return_aligned=False):
    """Point‑to‑plane ICP (linearised) – can return aligned points."""
    src = source.copy()
    tgt = target.copy()
    if voxel_size > 0:
        src = downsample_voxel(src, voxel_size)
        tgt = downsample_voxel(tgt, voxel_size)
    normals_tgt = estimate_normals(tgt, k=20)
    T_total = np.eye(4)
    prev_error = float('inf')
    for i in range(max_iterations):
        tree = cKDTree(tgt)
        distances, indices = tree.query(src)
        if distance_threshold is None:
            thresh = 3.0 * np.median(distances)
        else:
            thresh = distance_threshold
        valid = distances <= thresh
        src_matched = src[valid]
        tgt_matched = tgt[indices[valid]]
        normals_matched = normals_tgt[indices[valid]]
        if len(src_matched) < 10:
            print(f"ICP-P2Plane iter {i}: too few inliers, stopping.")
            break
        A_list = []
        b_list = []
        for j in range(len(src_matched)):
            p = src_matched[j]
            q = tgt_matched[j]
            n = normals_matched[j]
            A = np.zeros(6)
            A[0:3] = n
            A[3:6] = np.cross(p, n)
            b = np.dot(n, q - p)
            A_list.append(A)
            b_list.append(b)
        A = np.array(A_list)
        b = np.array(b_list)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        theta = x[3:6]
        R = np.eye(3) + np.array([[0, -theta[2], theta[1]],
                                  [theta[2], 0, -theta[0]],
                                  [-theta[1], theta[0], 0]])
        t = x[0:3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        src_h = np.hstack((src, np.ones((src.shape[0], 1))))
        src = (T @ src_h.T).T[:, :3]
        T_total = T @ T_total
        mean_error = np.mean(distances[valid])
        if abs(prev_error - mean_error) < tolerance:
            print(f"ICP-P2Plane converged at iter {i}, error={mean_error:.6f}")
            break
        prev_error = mean_error
    if return_aligned:
        src_h = np.hstack((source, np.ones((source.shape[0], 1))))
        aligned = (T_total @ src_h.T).T[:, :3]
        return T_total, aligned
    return T_total

# ----------------------------------------------------------------------
# 4. Trajectory Evaluation
# ----------------------------------------------------------------------
def compute_ate(est_poses, gt_poses):
    """Compute ATE RMSE after alignment (translation only)."""
    est_trans = np.array([p[:3, 3] for p in est_poses])
    gt_trans = np.array([p[:3, 3] for p in gt_poses])
    est_mean = np.mean(est_trans, axis=0)
    gt_mean = np.mean(gt_trans, axis=0)
    H = (est_trans - est_mean).T @ (gt_trans - gt_mean)
    U, _, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = gt_mean - R @ est_mean
    aligned = (R @ est_trans.T).T + t
    errors = np.linalg.norm(aligned - gt_trans, axis=1)
    rmse = np.sqrt(np.mean(errors ** 2))
    return rmse, aligned

def compute_rpe(est_poses, gt_poses):
    """Compute mean RPE (translation and rotation) for consecutive pairs."""
    trans_errors = []
    rot_errors = []
    for i in range(len(est_poses)-1):
        est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i+1]
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i+1]
        trans_err = np.linalg.norm(est_rel[:3, 3] - gt_rel[:3, 3])
        trans_errors.append(trans_err)
        R_err = est_rel[:3, :3] @ gt_rel[:3, :3].T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))
    return np.mean(trans_errors), np.mean(rot_errors)

# ----------------------------------------------------------------------
# 5. Visualisation Helpers (3D with black background)
# ----------------------------------------------------------------------
def plot_point_cloud_3d(point_clouds, colors, labels, title, elev=20, azim=-60, point_size=1, alpha=1, max_points=30000):
    """
    Plot one or more point clouds in 3D with a black background.
    If a cloud has more than max_points, it is randomly subsampled.
    Empty clouds are skipped.
    """
    # Filter out empty clouds and subsample if needed
    filtered = []
    for pts, color, label in zip(point_clouds, colors, labels):
        if len(pts) == 0:
            print(f"Warning: empty point cloud for label '{label}', skipping.")
            continue
        '''if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]'''
        filtered.append((pts, color, label))
    if not filtered:
        print("No point clouds to plot.")
        return
    # Unpack
    point_clouds, colors, labels = zip(*filtered)
    
    fig = plt.figure(figsize=(12, 10), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.xaxis.set_pane_color((0,0,0,1))
    ax.yaxis.set_pane_color((0,0,0,1))
    ax.zaxis.set_pane_color((0,0,0,1))
    ax.grid(False)

    for pts, color, label in zip(point_clouds, colors, labels):
        ax.scatter(pts[:,0], pts[:,1], pts[:,2],
                   c=color, s=point_size, marker='.', alpha=alpha,
                   linewidths=0, label=label)

    combined = np.vstack(point_clouds)
    limit = np.percentile(np.abs(combined), 99) if len(combined) > 0 else 10
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X', color='white', fontsize=12)
    ax.set_ylabel('Y', color='white', fontsize=12)
    ax.set_zlabel('Z', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, color='white', fontsize=18, fontweight='bold', pad=20)

    if labels:
        legend = ax.legend(facecolor='black', edgecolor='white', markerscale=10)
        for text in legend.get_texts():
            text.set_color('white')

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 6. Main Program
# ----------------------------------------------------------------------
def main():
    seq = '07'
    base_dir = os.path.join('dataset', 'sequences', seq)
    velo_dir = os.path.join(base_dir, 'velodyne')
    gt_file = os.path.join(base_dir, '07.txt')
    calib_file = os.path.join(base_dir, 'calib.txt')

    Tr = load_calibration(calib_file)
    Tr_inv = np.linalg.inv(Tr)
    print("Calibration Tr (lidar->camera):\n", Tr)

    gt_poses = load_ground_truth_poses(gt_file)

    num_frames = 10
    vis_target = 4
    failure_frame = 19
    frame_indices = list(range(num_frames)) + [failure_frame]

    scans = {}
    for idx in frame_indices:
        fname = os.path.join(velo_dir, f'{idx:06d}.bin')
        scans[idx] = load_velodyne_scan(fname)

    print(f"Loaded frames {list(scans.keys())}")

    # ------------------------------------------------------------------
    # Visualise Frame 0, Frame 4, and their overlay (before ICP)
    # Use alpha=0.5 for overlay to show blending
    # ------------------------------------------------------------------
    plot_point_cloud_3d([scans[0]], ['#ff1493'], ['Frame 0'], 'KITTI Seq 07 – Frame 0')
    plot_point_cloud_3d([scans[vis_target]], ['#00ffff'], [f'Frame {vis_target}'], f'KITTI Seq 07 – Frame {vis_target}')
    plot_point_cloud_3d([scans[0], scans[vis_target]],
                        ['#ff1493', '#00ffff'],
                        ['Frame 0', f'Frame {vis_target}'],
                        f'Overlay before ICP (Frame 0 vs Frame {vis_target})',
                        alpha=0.5)

    # ------------------------------------------------------------------
    # Apply full ICP (point-to-point) on Frame 0 -> Frame 4 (30 iterations)
    # ------------------------------------------------------------------
    T_full, aligned_full = icp_point_to_point(scans[0], scans[vis_target],
                                              max_iterations=30,
                                              voxel_size=0.3,
                                              return_aligned=True)

    plot_point_cloud_3d([aligned_full, scans[vis_target]],
                        ['#ff1493', '#00ffff'],
                        ['Aligned Frame 0', f'Frame {vis_target}'],
                        f'Full ICP Alignment (30 iterations)',
                        alpha=0.5)

    # ------------------------------------------------------------------
    # Full ICP on consecutive pairs (point-to-point) with initial guess
    # ------------------------------------------------------------------
    print("\n--- Running point-to-point ICP on consecutive pairs (with guess) ---")
    T_rels_p2p = []
    prev_T = np.eye(4)

    for i in range(num_frames - 1):
        src_guess = scans[i].copy()
        src_h = np.hstack((src_guess, np.ones((src_guess.shape[0], 1))))
        src_guess = (prev_T @ src_h.T).T[:, :3]

        T_rel = icp_point_to_point(src_guess, scans[i+1],
                                   max_iterations=30, tolerance=1e-5,
                                   distance_threshold=None, voxel_size=0.3,
                                   return_aligned=False)

        T_overall = T_rel @ prev_T
        T_rels_p2p.append(T_overall)
        prev_T = T_overall

    est_poses_p2p = [np.eye(4)]
    for T in T_rels_p2p:
        inv_T = np.linalg.inv(T)
        est_poses_p2p.append(est_poses_p2p[-1] @ inv_T)

    est_poses_cam_p2p = [Tr @ T @ Tr_inv for T in est_poses_p2p]

    # ------------------------------------------------------------------
    # Full ICP on consecutive pairs (point-to-plane) with initial guess
    # ------------------------------------------------------------------
    print("\n--- Running point-to-plane ICP on consecutive pairs (with guess) ---")
    T_rels_p2pl = []
    prev_T = np.eye(4)

    for i in range(num_frames - 1):
        src_guess = scans[i].copy()
        src_h = np.hstack((src_guess, np.ones((src_guess.shape[0], 1))))
        src_guess = (prev_T @ src_h.T).T[:, :3]

        T_rel = icp_point_to_plane(src_guess, scans[i+1],
                                   max_iterations=30, tolerance=1e-5,
                                   distance_threshold=None, voxel_size=0.3,
                                   return_aligned=False)

        T_overall = T_rel @ prev_T
        T_rels_p2pl.append(T_overall)
        prev_T = T_overall

    est_poses_p2pl = [np.eye(4)]
    for T in T_rels_p2pl:
        inv_T = np.linalg.inv(T)
        est_poses_p2pl.append(est_poses_p2pl[-1] @ inv_T)

    est_poses_cam_p2pl = [Tr @ T @ Tr_inv for T in est_poses_p2pl]

    gt_poses_10 = gt_poses[:num_frames]

    # ------------------------------------------------------------------
    # Trajectory plot
    # ------------------------------------------------------------------
    gt_trans = np.array([p[:3, 3] for p in gt_poses_10])
    est_trans_p2p = np.array([p[:3, 3] for p in est_poses_cam_p2p])
    est_trans_p2pl = np.array([p[:3, 3] for p in est_poses_cam_p2pl])

    plt.figure(figsize=(10, 8))
    plt.plot(gt_trans[:, 0], gt_trans[:, 2], 'g-', label='Ground Truth', linewidth=2)
    plt.plot(est_trans_p2p[:, 0], est_trans_p2p[:, 2], 'b--', label='Point‑to‑Point', linewidth=2)
    plt.plot(est_trans_p2pl[:, 0], est_trans_p2pl[:, 2], 'r--', label='Point‑to‑Plane', linewidth=2)
    plt.xlabel('X (camera)')
    plt.ylabel('Z (camera)')
    plt.title('Trajectory Comparison (Bird\'s‑eye, camera frame)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    ate_p2p, _ = compute_ate(est_poses_cam_p2p, gt_poses_10)
    rpe_trans_p2p, rpe_rot_p2p = compute_rpe(est_poses_cam_p2p, gt_poses_10)

    ate_p2pl, _ = compute_ate(est_poses_cam_p2pl, gt_poses_10)
    rpe_trans_p2pl, rpe_rot_p2pl = compute_rpe(est_poses_cam_p2pl, gt_poses_10)

    print("\n--- Evaluation Results ---")
    print(f"Point‑to‑Point: ATE RMSE = {ate_p2p:.4f} m, RPE trans = {rpe_trans_p2p:.4f} m, RPE rot = {rpe_rot_p2p:.2f} deg")
    print(f"Point‑to‑Plane: ATE RMSE = {ate_p2pl:.4f} m, RPE trans = {rpe_trans_p2pl:.4f} m, RPE rot = {rpe_rot_p2pl:.2f} deg")

    # ------------------------------------------------------------------
    # Drift plot (both methods)
    # ------------------------------------------------------------------
    ate_prefix_p2p = []
    ate_prefix_p2pl = []
    for n in range(2, num_frames+1):
        est_pref_p2p = est_poses_cam_p2p[:n]
        est_pref_p2pl = est_poses_cam_p2pl[:n]
        gt_pref = gt_poses_10[:n]
        ate_p2p, _ = compute_ate(est_pref_p2p, gt_pref)
        ate_p2pl, _ = compute_ate(est_pref_p2pl, gt_pref)
        ate_prefix_p2p.append(ate_p2p)
        ate_prefix_p2pl.append(ate_p2pl)

    plt.figure()
    plt.plot(range(2, num_frames+1), ate_prefix_p2p, 'o-', label='ATE RMSE (point‑to‑point)')
    plt.plot(range(2, num_frames+1), ate_prefix_p2pl, 's-', label='ATE RMSE (point‑to‑plane)')
    plt.xlabel('Number of frames')
    plt.ylabel('ATE RMSE (m)')
    plt.title('Drift Growth with Frame Count – Both Methods')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ------------------------------------------------------------------
    # FAILURE CASE: Artificial bad initial guess (rotate source)
    # ------------------------------------------------------------------
    print(f"\n--- FAILURE CASE: Aligning Frame 0 and Frame {failure_frame} with artificial bad guess ---")

    from scipy.spatial.transform import Rotation as R

    # Create a bad initial guess: rotate source by 20° around Z
    rot_angle_deg = 20
    rot_bad = R.from_euler('z', rot_angle_deg, degrees=True).as_matrix()
    T_bad = np.eye(4)
    T_bad[:3, :3] = rot_bad

    # Apply bad guess to source
    src_bad = scans[0].copy()
    src_h = np.hstack((src_bad, np.ones((src_bad.shape[0], 1))))
    src_bad = (T_bad @ src_h.T).T[:, :3]

    # 1) Point-to-point WITHOUT any guess (start from rotated source)
    T_fail, aligned_fail = icp_point_to_point(
        src_bad, scans[failure_frame],
        max_iterations=100, voxel_size=0.3,
        return_aligned=True
    )

    # 2) Point-to-point WITH good initial guess (coarse ICP)
    print("Computing coarse initial guess via ICP with larger voxel...")
    coarse_T = icp_point_to_point(
        scans[0], scans[failure_frame],
        max_iterations=30, voxel_size=0.5,
        return_aligned=False
    )

    src_guess = scans[0].copy()
    src_h = np.hstack((src_guess, np.ones((src_guess.shape[0], 1))))
    src_guess = (coarse_T @ src_h.T).T[:, :3]
    T_refine, aligned_guess = icp_point_to_point(
        src_guess, scans[failure_frame],
        max_iterations=50, voxel_size=0.3,
        return_aligned=True
    )

    # 3) Point-to-plane WITHOUT any guess (start from rotated source)
    T_plane, aligned_plane = icp_point_to_plane(
        src_bad, scans[failure_frame],
        max_iterations=100, voxel_size=0.3,
        return_aligned=True
    )

    # 4) Point-to-plane WITH initial guess (using the same coarse_T)
    src_guess_plane = scans[0].copy()
    src_h = np.hstack((src_guess_plane, np.ones((src_guess_plane.shape[0], 1))))
    src_guess_plane = (coarse_T @ src_h.T).T[:, :3]
    T_plane_refine, aligned_plane_guess = icp_point_to_plane(
        src_guess_plane, scans[failure_frame],
        max_iterations=50, voxel_size=0.3,
        return_aligned=True
    )

    # Compute errors
    tree_target = cKDTree(scans[failure_frame])
    dist_fail, _ = tree_target.query(aligned_fail)
    dist_guess, _ = tree_target.query(aligned_guess)
    dist_plane, _ = tree_target.query(aligned_plane)
    dist_plane_guess, _ = tree_target.query(aligned_plane_guess)

    print(f"Point-to-point WITHOUT guess (bad start): mean dist = {np.mean(dist_fail):.4f} m, median = {np.median(dist_fail):.4f} m")
    print(f"Point-to-point WITH good guess:           mean dist = {np.mean(dist_guess):.4f} m, median = {np.median(dist_guess):.4f} m")
    print(f"Point-to-plane WITHOUT guess (bad start): mean dist = {np.mean(dist_plane):.4f} m, median = {np.median(dist_plane):.4f} m")
    print(f"Point-to-plane WITH good guess:           mean dist = {np.mean(dist_plane_guess):.4f} m, median = {np.median(dist_plane_guess):.4f} m")

    print("\nExplanation: A 20° rotational error makes point‑to‑point ICP without a good initial guess converge to a poor local minimum (large alignment error). A coarse initial guess (from ICP with large voxel) corrects this. Point‑to‑plane is more robust and still works even with the bad guess.")

if __name__ == '__main__':
    main()
