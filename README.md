# Project: Drawing a Vehicle's Path from LiDAR Scans (KITTI)

> A complete implementation of LiDAR odometry using the **Iterative Closest Point (ICP)** algorithm on **KITTI Odometry Sequence 07**, developed from scratch without using Open3D or external ICP libraries.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

# Project Overview

This project estimates the trajectory of a vehicle using consecutive LiDAR scans from the KITTI Odometry dataset.

The complete pipeline includes:

- Loading raw KITTI `.bin` LiDAR scans
- Point cloud preprocessing
- Voxel grid downsampling
- Point-to-Point ICP
- Point-to-Plane ICP
- Pose accumulation
- LiDAR-to-camera calibration
- Trajectory estimation
- Quantitative evaluation using ATE and RPE
- Failure case analysis and visualization

---

# Features

- Pure NumPy + SciPy implementation, No Open3D ICP implementation used
- Point-to-Point ICP
- Point-to-Plane ICP
- Data processing: removal of invalid points, random point selection before plotting, voxel downsampling.
- Adaptive outlier rejection
- Constant-velocity initial guess + coarse initial guess (for failure case)
- Trajectory and point cloud visualisation
- Drift analysis
- Failure case demonstration

---

# Dataset

**Dataset:** KITTI Odometry Sequence 07

Contents:

- Velodyne LiDAR scans (`.bin`)
- Ground-truth poses (`07.txt`)
- Calibration (`calib.txt`)
=> This dataset can be downloaded using download_data.sh file which is available in this repo.
=> The dataset can be downloaded through the official KITTI website:- https://www.cvlibs.net/datasets/kitti/eval_odometry.php.
---

# Methodology

## 1. Preprocessing

- Remove invalid points
- Apply voxel grid downsampling
- (Optional) choosing 30k random points for point cloud plotting.

## 2. Point-to-Point ICP

- KD-tree nearest neighbour search
- Closed-form SVD transformation estimation
- Adaptive outlier rejection, Constant-velocity initial guess
- Iterative refinement until convergence

## 3. Point-to-Plane ICP

- Surface normal estimation using PCA
- Linear least-squares optimization
- Adaptive outlier rejection, Constant-velocity initial guess
- Improved robustness and accuracy

## 4. Pose Estimation and Error Calculation

Relative transforms are accumulated to obtain the complete vehicle trajectory. ATE and RPE are calculated. Drift is analysed.

## 5. Calibration

Estimated poses are transformed into the camera coordinate frame using KITTI calibration.

## 6. Plotting and Visualisation
Point clouds and graphs are plotted for visual demonstration.

---

# Implementation

Main modules include:

- Data loading
- Voxel downsampling
- Normal estimation
- ICP algorithms
- Pose accumulation
- Evaluation
- Visualization

---

# Evaluation Metrics

## Absolute Trajectory Error (ATE)

Measures global trajectory accuracy.

## Relative Pose Error (RPE)

Measures frame-to-frame motion estimation accuracy.

---

# Project Structure

```text
project/
│
├── main.py
├── requirements.txt
├── README.md
├── download_data.sh
│
└── dataset/
    └── sequences/
        └── 07/
            ├── velodyne/
            ├── calib.txt
            └── 07.txt
```

---

# Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Installation

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

Download KITTI Sequence 07 and place it under:

```text
dataset/sequences/07/
```

---

# Usage

Run:

```bash
python main.py
```

The program will:

1. Load LiDAR scans
2. Perform ICP registration
3. Estimate vehicle trajectory
4. Compare against ground truth
5. Display evaluation metrics and plots

---

# Future Improvements

- Loop closure
- Global pose graph optimization
- Scan-to-map registration
- GPU acceleration
- Robust M-estimators
- Dynamic object removal

---
# Results

The proposed LiDAR odometry pipeline was evaluated on **KITTI Odometry Sequence 07** using the first **10 consecutive LiDAR frames (Frames 0–9)**. Both **Point-to-Point ICP** and **Point-to-Plane ICP** were implemented and compared against the KITTI ground-truth trajectory.

## Quantitative Evaluation

| Method | ATE RMSE | RPE Translation | RPE Rotation |
|:-------|---------:|----------------:|-------------:|
| Point-to-Point ICP | **0.1221 m** | **0.0418 m** | **0.06°** |
| Point-to-Plane ICP | **0.0099 m** | **0.0050 m** | **0.26°** |

### Performance Analysis

- **Point-to-Plane ICP** achieved the best overall trajectory estimation accuracy.
- The **Absolute Trajectory Error (ATE)** was reduced from **12.21 cm** to **0.99 cm**, representing more than a **12× improvement** over Point-to-Point ICP.
- The **Relative Pose Error (Translation)** decreased from **4.18 cm** to **0.50 cm**, demonstrating significantly improved frame-to-frame motion estimation.
- Both methods produced low rotational error, with Point-to-Point ICP giving a slightly lower rotational RPE, while Point-to-Plane ICP achieved substantially better translation accuracy.

---

## ICP Convergence

The algorithms converged successfully for every pair of consecutive scans.

### Point-to-Point ICP

| Frame Pair | Iterations | Mean Error |
|------------|-----------:|-----------:|
| 0 → 1 | 12 | 0.090318 |
| 1 → 2 | 3 | 0.061549 |
| 2 → 3 | 5 | 0.075355 |
| 3 → 4 | 6 | 0.088832 |
| 4 → 5 | 5 | 0.093053 |
| 5 → 6 | 3 | 0.090048 |
| 6 → 7 | 9 | 0.082353 |
| 7 → 8 | 6 | 0.091428 |
| 8 → 9 | 3 | 0.090644 |

### Point-to-Plane ICP

| Frame Pair | Iterations | Mean Error |
|------------|-----------:|-----------:|
| 0 → 1 | 9 | 0.093236 |
| 1 → 2 | 5 | 0.061471 |
| 2 → 3 | 8 | 0.074863 |
| 3 → 4 | 6 | 0.086184 |
| 4 → 5 | 6 | 0.091704 |
| 5 → 6 | 6 | 0.085629 |
| 6 → 7 | 8 | 0.079306 |
| 7 → 8 | 5 | 0.085845 |
| 8 → 9 | 6 | 0.083014 |

The relatively low number of iterations required for convergence demonstrates the effectiveness of the **constant-velocity initial guess**, which provides a good starting estimate for ICP optimization.

---

## Failure Case Analysis

To investigate the sensitivity of ICP to initialization, an artificial **20° rotational error** was introduced before registration between **Frame 0** and **Frame 19**.

| Method | Mean Distance | Median Distance |
|:-------|--------------:|----------------:|
| Point-to-Point ICP (No Initial Guess) | **0.7811 m** | **0.4220 m** |
| Point-to-Point ICP (With Coarse Initial Guess) | **0.2329 m** | **0.1323 m** |
| Point-to-Plane ICP (No Initial Guess) | **0.4741 m** | **0.2685 m** |
| Point-to-Plane ICP (With Coarse Initial Guess) | **0.1896 m** | **0.0593 m** |


### Discussion

The experiment highlights the importance of initialization in ICP-based scan registration.

- Point-to-Point ICP is highly susceptible to local minima when initialized with a poor estimate.
- A coarse initial alignment substantially improves convergence.
- Point-to-Plane ICP is inherently more robust to poor initialization because it minimizes the point-to-surface distance rather than the point-to-point Euclidean distance.
- Combining a coarse initial estimate with Point-to-Plane ICP produced the most accurate alignment.

---

## Visual Outputs

The implementation automatically generates the following visualizations:

- Individual LiDAR point cloud visualisation for Frames and Frame 4
- Overlay of two scans before ICP registration
- Overlay after successful ICP alignment
- Estimated trajectory vs. KITTI ground-truth trajectory
- Drift growth (ATE) analysis
- Failure case comparison demonstrating ICP convergence behaviour
  <img width="1280" height="800" alt="Screenshot 2026-06-28 at 9 47 06 PM" src="https://github.com/user-attachments/assets/c198bb9a-e9a7-489d-a126-21df7b2571d0" />

<img width="1280" height="800" alt="Screenshot 2026-06-28 at 11 35 44 PM" src="https://github.com/user-attachments/assets/f3e72e4d-b044-4633-a2f4-c6b17fe6f28d" />
<img width="1280" height="800" alt="Screenshot 2026-06-28 at 11 35 53 PM" src="https://github.com/user-attachments/assets/cabef84c-15fa-4c40-8821-307691ed3bc6" />
<img width="1214" height="694" alt="Screenshot 2026-06-28 at 9 47 49 PM" src="https://github.com/user-attachments/assets/5ce4ace9-7024-449d-b7c1-8c558bcddc1e" />
<img width="609" height="455" alt="Screenshot 2026-06-28 at 9 47 57 PM" src="https://github.com/user-attachments/assets/8a3c76f8-e4ad-4a71-8737-a14b8aa8db9a" />


---

## Terminal Output Summary

```
Point-to-Point ICP
------------------
ATE RMSE          : 0.1221 m
Translation RPE   : 0.0418 m
Rotation RPE      : 0.06°

Point-to-Plane ICP
------------------
ATE RMSE          : 0.0099 m
Translation RPE   : 0.0050 m
Rotation RPE      : 0.26°
```

The experimental results demonstrate that the **Point-to-Plane ICP implementation consistently outperforms the Point-to-Point variant in trajectory estimation accuracy**, achieving **sub-centimetre Absolute Trajectory Error** on KITTI Sequence 07 while maintaining low rotational error.


# Author

**Surat Bhushan**
Under the guidance of Prashant Kumar Sir.

IIT Delhi Internship Project
