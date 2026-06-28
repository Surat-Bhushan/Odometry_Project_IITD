#!/bin/bash
# download_data.sh
# Downloads and extracts KITTI Odometry Velodyne data (Sequence 07 only).

set -e

echo "Starting KITTI Odometry data download..."

# Create dataset directory if it doesn't exist
mkdir -p dataset/sequences/07/velodyne

# URL of the KITTI Odometry Velodyne zip (contains all sequences)
URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
ZIP_FILE="data_odometry_velodyne.zip"

# Check if we already have the zip (avoid re-downloading)
if [ ! -f "$ZIP_FILE" ]; then
    echo "Downloading $ZIP_FILE (≈8 GB) ..."
    if command -v wget &> /dev/null; then
        wget -c "$URL" -O "$ZIP_FILE"
    elif command -v curl &> /dev/null; then
        curl -L -C - -o "$ZIP_FILE" "$URL"
    else
        echo "Error: Neither wget nor curl found. Please install one."
        exit 1
    fi
else
    echo "Zip file already exists. Skipping download."
fi

# Extract only Sequence 07 Velodyne data
echo "Extracting Sequence 07 (velodyne) from zip..."
unzip -j "$ZIP_FILE" "dataset/sequences/07/velodyne/*" -d "dataset/sequences/07/velodyne/"

# Clean up zip (optional)
# rm "$ZIP_FILE"

echo "Done! KITTI Sequence 07 Velodyne data is ready in dataset/sequences/07/velodyne/"
