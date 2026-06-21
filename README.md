# Project 10: Drawing a Vehicle's Path from LiDAR Scans (KITTI)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Requirements](#project-requirements)
3. [Data Sources](#data-sources)
4. [Implementation Details](#implementation-details)
5. [Environment & Configuration](#environment--configuration)
6. [Demo & Results](#demo--results)

---

## Project Overview
This project addresses the fundamental challenge of SLAM (Simultaneous Localization and Mapping): determining a vehicle's motion by aligning sequential LiDAR scans. Using the Iterative Closest Point (ICP) algorithm, we reconstruct the vehicle's trajectory and compare it against ground-truth GPS data to analyze drift.

## Project Requirements
This implementation covers both the "Easy" (Open3D-based) and "Hard" (from-scratch implementation) project tracks:

* **Scan Alignment**: Aligning consecutive LiDAR point clouds using the ICP algorithm.
* **Path Reconstruction**: Chaining relative transformations to build a global trajectory.
* **Drift Analysis**: Visualizing the reconstructed path against ground-truth GPS data.
* **Understanding-Checks**: Analysis of ICP performance and ablation studies on voxel downsampling.

## Data Sources
We utilize the KITTI Odometry Dataset for all experiments.

* **Official Source**: [KITTI Vision Benchmark Suite](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)
* **Kaggle Implementation**: [KITTI - Odometry (Kaggle Dataset)](https://www.kaggle.com/datasets/hocop1/kitti-odometry/data)
    * **Why we chose this**: The Kaggle version is pre-organized into the required folder structure (`sequences/XX/velodyne/`). Using this mounted volume eliminates ~80GB of manual data-wrangling and ensures that our path-loading code is compatible with standard community practices.
## Implementation Details
Our project follows a modular design to ensure reproducibility:

1. **Data Loading**: A custom utility parses binary `.bin` files into $N \times 4$ arrays $(x, y, z, \text{intensity})$.
2. **Preprocessing**: LiDAR scans are voxel downsampled to optimize computation without losing critical geometry.
3. **Alignment**: The ICP algorithm is applied to minimize the point-to-point distance between consecutive frames.
4. **Integration**: Relative pose matrices are accumulated to calculate the global path.

## Environment & Configuration
To ensure consistent results, we utilize the Kaggle Notebook environment:

* **Accelerator**: GPU T4 x2 (Enabled for enhanced memory and standardized performance).
* **Internet**: Enabled (for library installation via `pip install open3d`).
* **Repeatability**: The notebook is designed to run top-to-bottom ("Run All").

## Demo & Results

---

*This project was developed as part of the IIT Delhi Internship program.*
