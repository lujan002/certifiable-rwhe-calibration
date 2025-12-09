import numpy as np
import csv
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import defaultdict
from scipy.spatial.transform import Rotation


def load_poses_with_timestamps(csv_file: str, 
                                timestamp_col: int = 0,
                                pos_cols: Tuple[int, int, int] = (1, 2, 3),
                                quat_cols: Tuple[int, int, int, int] = (4, 5, 6, 7),
                                skip_rows: int = 0,
                                has_header: bool = False) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load poses with timestamps from CSV file.
    
    Args:
        csv_file: Path to CSV file
        timestamp_col: Column index for timestamp
        pos_cols: Column indices for position (x, y, z)
        quat_cols: Column indices for quaternion (qx, qy, qz, qw)
        skip_rows: Number of rows to skip at the beginning
        has_header: Whether the file has a header row
        
    Returns:
        timestamps: Array of timestamps
        poses: List of 4x4 transformation matrices
    """
    timestamps = []
    poses = []
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        
        # Skip header if present
        if has_header:
            next(reader)
        
        # Skip specified number of rows
        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                break
        
        for row in reader:
            if len(row) < max(timestamp_col, max(pos_cols), max(quat_cols)) + 1:
                continue
                
            try:
                # Extract timestamp
                ts = float(row[timestamp_col])
                
                # Extract position
                t = np.array([float(row[pos_cols[0]]), 
                             float(row[pos_cols[1]]), 
                             float(row[pos_cols[2]])])
                
                # Extract quaternion (qx, qy, qz, qw)
                q = [float(row[quat_cols[0]]), 
                     float(row[quat_cols[1]]), 
                     float(row[quat_cols[2]]), 
                     float(row[quat_cols[3]])]
                
                # Convert quaternion to rotation matrix
                R = Rotation.from_quat(q).as_matrix()
                
                # Construct 4x4 transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                
                timestamps.append(ts)
                poses.append(T)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue
    
    print(f"Loaded {len(poses)} poses from {csv_file}")
    return np.array(timestamps), poses


def load_tagged_poses_with_timestamps(csv_file: str,
                                      timestamp_col: int = 0,
                                      tag_col: int = 1,
                                      pos_cols: Tuple[int, int, int] = (6, 7, 8),
                                      quat_cols: Tuple[int, int, int, int] = (2, 3, 4, 5),
                                      skip_rows: int = 0,
                                      has_header: bool = True) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Load tagged poses with timestamps from CSV file.

    Args mirror load_poses_with_timestamps, but additionally:
        tag_col: Column index for the tag id

    Returns:
        timestamps: Array of timestamps
        poses: List of 4x4 transformation matrices
        tags: Array of integer tag ids (rows with missing/invalid tag ids are skipped)
    """
    timestamps: List[float] = []
    poses: List[np.ndarray] = []
    tags: List[int] = []

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)

        if has_header:
            next(reader)

        for _ in range(skip_rows):
            try:
                next(reader)
            except StopIteration:
                break

        for row in reader:
            if len(row) < max(timestamp_col, tag_col, max(pos_cols), max(quat_cols)) + 1:
                continue

            try:
                tag_raw = row[tag_col]
                tag_val = float(tag_raw)
                if np.isnan(tag_val):
                    continue
                tag_id = int(tag_val)
            except (ValueError, TypeError):
                continue

            try:
                ts = float(row[timestamp_col])
                t = np.array([float(row[pos_cols[0]]),
                              float(row[pos_cols[1]]),
                              float(row[pos_cols[2]])])
                q = [float(row[quat_cols[0]]),
                     float(row[quat_cols[1]]),
                     float(row[quat_cols[2]]),
                     float(row[quat_cols[3]])]
                R = Rotation.from_quat(q).as_matrix()

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t

            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping row due to error: {e}")
                continue

            timestamps.append(ts)
            poses.append(T)
            tags.append(tag_id)

    print(f"Loaded {len(poses)} tagged poses from {csv_file}")
    return np.array(timestamps), poses, np.array(tags, dtype=int)


def match_timestamps(timestamps_A: np.ndarray, 
                     timestamps_B: np.ndarray, 
                     threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match timestamps from A and B datasets within a threshold.
    
    Finds globally optimal matching by considering all possible pairs and selecting
    the ones with smallest time differences. Ensures each timestamp is used at most once.
    
    Args:
        timestamps_A: Array of timestamps from dataset A
        timestamps_B: Array of timestamps from dataset B
        threshold: Maximum time difference for a valid match (in same units as timestamps)
        
    Returns:
        matched_indices_A: Indices of matched timestamps in A
        matched_indices_B: Indices of matched timestamps in B
    """
    # Find all valid pairs within threshold
    valid_pairs = []
    
    for idx_A, ts_A in enumerate(timestamps_A):
        for idx_B, ts_B in enumerate(timestamps_B):
            diff = abs(ts_A - ts_B)
            if diff < threshold:
                valid_pairs.append((idx_A, idx_B, diff))
    
    if len(valid_pairs) == 0:
        print(f"No valid pairs found within threshold {threshold}")
        return np.array([]), np.array([])
    
    # Sort by time difference (smallest first) to prioritize best matches
    valid_pairs.sort(key=lambda x: x[2])
    
    # Greedily select matches, ensuring each timestamp is used only once
    matched_indices_A = []
    matched_indices_B = []
    used_A = set()
    used_B = set()
    
    for idx_A, idx_B, diff in valid_pairs:
        if idx_A not in used_A and idx_B not in used_B:
            matched_indices_A.append(idx_A)
            matched_indices_B.append(idx_B)
            used_A.add(idx_A)
            used_B.add(idx_B)
    
    # Sort matches by A indices for consistency
    sorted_order = np.argsort(matched_indices_A)
    matched_indices_A = np.array([matched_indices_A[i] for i in sorted_order])
    matched_indices_B = np.array([matched_indices_B[i] for i in sorted_order])
    
    print(f"Matched {len(matched_indices_A)} pairs out of {len(timestamps_A)} A timestamps "
          f"and {len(timestamps_B)} B timestamps (threshold: {threshold})")
    
    # Print some statistics
    if len(matched_indices_A) > 0:
        diffs = [abs(timestamps_A[i] - timestamps_B[j]) 
                 for i, j in zip(matched_indices_A, matched_indices_B)]
        print(f"  Average time difference: {np.mean(diffs):.6f}s")
        print(f"  Max time difference: {np.max(diffs):.6f}s")
        print(f"  Min time difference: {np.min(diffs):.6f}s")
    
    return matched_indices_A, matched_indices_B


def pose_to_csv_format(pose: np.ndarray) -> np.ndarray:
    """
    Convert a 4x4 transformation matrix to CSV format.
    
    CSV format: [qx, qy, qz, qw, tx, ty, tz]
    (quaternion first 4, translation last 3)
    
    Args:
        pose: 4x4 transformation matrix
        
    Returns:
        Array of 7 values: [qx, qy, qz, qw, tx, ty, tz]
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Convert rotation matrix to quaternion (scipy uses [x, y, z, w] format)
    quat = Rotation.from_matrix(R).as_quat()  # Returns [x, y, z, w]
    
    # Return in format: [qx, qy, qz, qw, tx, ty, tz]
    return np.array([quat[0], quat[1], quat[2], quat[3], t[0], t[1], t[2]])


def construct_synchronized_csvs(poses_A: List[np.ndarray],
                                 poses_B: List[np.ndarray],
                                 matched_indices_A: np.ndarray,
                                 matched_indices_B: np.ndarray,
                                 output_file_A: str,
                                 output_file_B: str):
    """
    Construct synchronized A and B CSV files from matched poses.
    
    Args:
        poses_A: List of pose matrices from dataset A
        poses_B: List of pose matrices from dataset B
        matched_indices_A: Indices of matched poses in A
        matched_indices_B: Indices of matched poses in B
        output_file_A: Path to output CSV file for A
        output_file_B: Path to output CSV file for B
    """
    # Extract matched poses
    matched_poses_A = [poses_A[i] for i in matched_indices_A]
    matched_poses_B = [poses_B[i] for i in matched_indices_B]
    
    # Convert to CSV format
    csv_data_A = np.array([pose_to_csv_format(pose) for pose in matched_poses_A])
    csv_data_B = np.array([pose_to_csv_format(pose) for pose in matched_poses_B])
    
    # Write CSV files
    np.savetxt(output_file_A, csv_data_A, delimiter=',', fmt='%.18e')
    np.savetxt(output_file_B, csv_data_B, delimiter=',', fmt='%.18e')
    
    print(f"Saved {len(matched_poses_A)} synchronized poses to:")
    print(f"  A: {output_file_A}")
    print(f"  B: {output_file_B}")


def synchronize_tagged_camera(poses_A: List[np.ndarray],
                              timestamps_A: np.ndarray,
                              csv_file_B: str,
                              camera_id: int,
                              output_dir: Path,
                              threshold: float,
                              timestamp_col_B: int = 0,
                              tag_col_B: int = 1,
                              pos_cols_B: Tuple[int, int, int] = (6, 7, 8),
                              quat_cols_B: Tuple[int, int, int, int] = (2, 3, 4, 5),
                              skip_rows_B: int = 0,
                              has_header_B: bool = True):
    """
    Synchronize all tags for a single camera B file against dataset A.

    Creates one A/B CSV pair per tag id: tag_<id>_cam<camera_id>_A.csv and tag_<id>_cam<camera_id>_B.csv.
    """
    timestamps_B, poses_B, tags_B = load_tagged_poses_with_timestamps(
        csv_file_B,
        timestamp_col=timestamp_col_B,
        tag_col=tag_col_B,
        pos_cols=pos_cols_B,
        quat_cols=quat_cols_B,
        skip_rows=skip_rows_B,
        has_header=has_header_B,
    )

    tag_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, tag_id in enumerate(tags_B):
        tag_to_indices[tag_id].append(idx)

    if not tag_to_indices:
        print(f"No valid tags found in {csv_file_B}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for tag_id, indices in tag_to_indices.items():
        tag_timestamps = timestamps_B[indices]
        tag_poses = [poses_B[i] for i in indices]

        matched_indices_A, matched_indices_B = match_timestamps(
            timestamps_A, tag_timestamps, threshold
        )

        if len(matched_indices_A) == 0:
            print(f"Skipping tag {tag_id} cam{camera_id}: no matches within threshold.")
            continue

        output_file_A = output_dir / f"tag_{tag_id}_cam_{camera_id}_A.csv"
        output_file_B = output_dir / f"tag_{tag_id}_cam_{camera_id}_B.csv"

        construct_synchronized_csvs(
            poses_A,
            tag_poses,
            matched_indices_A,
            matched_indices_B,
            str(output_file_A),
            str(output_file_B),
        )


def synchronize_and_save(csv_file_A: str,
                         csv_file_B: str,
                         output_file_A: str,
                         output_file_B: str,
                         threshold: float,
                         timestamp_col_A: int = 2,
                         timestamp_col_B: int = 0,
                         pos_cols_A: Tuple[int, int, int] = (7, 8, 9),
                         pos_cols_B: Tuple[int, int, int] = (6, 7, 8),
                         quat_cols_A: Tuple[int, int, int, int] = (3, 4, 5, 6),
                         quat_cols_B: Tuple[int, int, int, int] = (2, 3, 4, 5),
                         skip_rows_A: int = 0,
                         skip_rows_B: int = 1,
                         has_header_A: bool = False,
                         has_header_B: bool = False):
    """
    Main function to synchronize two datasets by timestamp and save synchronized CSV files.
    
    Args:
        csv_file_A: Path to input CSV file for dataset A (with timestamps)
        csv_file_B: Path to input CSV file for dataset B (with timestamps)
        output_file_A: Path to output synchronized CSV file for A
        output_file_B: Path to output synchronized CSV file for B
        threshold: Maximum time difference for matching (in same units as timestamps)
        timestamp_col_A: Column index for timestamp in file A
        timestamp_col_B: Column index for timestamp in file B
        pos_cols_A: Column indices for position in file A (x, y, z)
        pos_cols_B: Column indices for position in file B (x, y, z)
        quat_cols_A: Column indices for quaternion in file A (qx, qy, qz, qw)
        quat_cols_B: Column indices for quaternion in file B (qx, qy, qz, qw)
        skip_rows_A: Number of rows to skip at start of file A
        skip_rows_B: Number of rows to skip at start of file B
        has_header_A: Whether file A has a header row
        has_header_B: Whether file B has a header row
    """
    # Load data
    print("Loading dataset A...")
    timestamps_A, poses_A = load_poses_with_timestamps(
        csv_file_A, timestamp_col_A, pos_cols_A, quat_cols_A, 
        skip_rows_A, has_header_A
    )
    
    print("Loading dataset B...")
    timestamps_B, poses_B = load_poses_with_timestamps(
        csv_file_B, timestamp_col_B, pos_cols_B, quat_cols_B, 
        skip_rows_B, has_header_B
    )
    
    # Match timestamps
    print(f"Matching timestamps with threshold: {threshold}")
    matched_indices_A, matched_indices_B = match_timestamps(
        timestamps_A, timestamps_B, threshold
    )
    
    if len(matched_indices_A) == 0:
        print("ERROR: No matching timestamps found! Check your threshold and data.")
        return
    
    # Construct and save synchronized CSV files
    construct_synchronized_csvs(
        poses_A, poses_B, matched_indices_A, matched_indices_B,
        output_file_A, output_file_B
    )


def synchronize_all_cameras_by_tag(csv_file_A: str,
                                   camera_files: Dict[int, str],
                                   output_dir: str,
                                   threshold: float,
                                   timestamp_col_A: int = 2,
                                   pos_cols_A: Tuple[int, int, int] = (7, 8, 9),
                                   quat_cols_A: Tuple[int, int, int, int] = (3, 4, 5, 6),
                                   skip_rows_A: int = 0,
                                   has_header_A: bool = False,
                                   **camera_kwargs):
    """
    Convenience wrapper: load dataset A once, then synchronize every tag in each
    camera B file. Creates one CSV pair per tag/camera combination.
    """
    print("Loading dataset A...")
    timestamps_A, poses_A = load_poses_with_timestamps(
        csv_file_A,
        timestamp_col_A,
        pos_cols_A,
        quat_cols_A,
        skip_rows_A,
        has_header_A,
    )

    output_root = Path(output_dir)
    # Start fresh: remove existing files in the output directory
    if output_root.exists():
        for item in output_root.iterdir():
            try:
                if item.is_file():
                    item.unlink()
            except Exception as e:
                print(f"Warning: could not delete {item}: {e}")

    for camera_id, csv_file_B in camera_files.items():
        print(f"\nProcessing camera {camera_id} ({csv_file_B}) ...")
        synchronize_tagged_camera(
            poses_A=poses_A,
            timestamps_A=timestamps_A,
            csv_file_B=csv_file_B,
            camera_id=camera_id,
            output_dir=output_root,
            threshold=threshold,
            **camera_kwargs,
        )


# Example usage
if __name__ == "__main__":
    # Example: split every tag for both cameras and save per-tag/camera CSVs
    csv_file_A = "data/high-bay/raw/optitrack_cut.csv"
    camera_files = {
        1: "data/high-bay/raw/Cam1_cut.csv",
        2: "data/high-bay/raw/Cam2_cut.csv",
    }
    output_dir = "data/high-bay/combined"
    threshold = 0.1  # 100ms threshold

    synchronize_all_cameras_by_tag(
        csv_file_A=csv_file_A,
        camera_files=camera_files,
        output_dir=output_dir,
        threshold=threshold,
        # Example overrides for differing CSV layouts:
        # timestamp_col_A=0,
        # pos_cols_A=(1, 2, 3),
        # quat_cols_A=(4, 5, 6, 7),
        # has_header_A=True,
        # timestamp_col_B=0,
        # tag_col_B=1,
        # pos_cols_B=(6, 7, 8),
        # quat_cols_B=(2, 3, 4, 5),
        # has_header_B=True,
    )
