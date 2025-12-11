#!/usr/bin/env python3
"""
Combined visualization of Optitrack poses and AprilTag detections.

Visualizes:
- Camera/rig pose from Optitrack CSV (camera frame position)
- AprilTag poses relative to camera from B matrix CSV
- Allows thumbing through synchronized timestamps

Usage:
    python3 visualize_combined_poses.py \
        --optitrack optitrack.csv \
        --b-csv tag_0_cam_1_B.csv \
        [--a-csv tag_0_cam_1_A.csv]  # Optional: for timestamps if B CSV doesn't have them
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from pathlib import Path
import sys
import cv2
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import yaml

# Import AprilTag detector from construct_B
sys.path.insert(0, str(Path(__file__).parent))
try:
    from construct_B import AprilTagPoseEstimator
except ImportError:
    print("Warning: Could not import AprilTagPoseEstimator from construct_B")
    AprilTagPoseEstimator = None

# Add parent directory to path for SE3 utils if needed
sys.path.insert(0, str(Path(__file__).parent))

def load_optitrack_csv(csv_file, quat_cols=(2, 3, 4, 5), pos_cols=(6, 7, 8), skip_rows=7):
    """Load Optitrack CSV file."""
    df = pd.read_csv(csv_file, skiprows=skip_rows)
    
    # Extract timestamps (column 1 = "Time (Seconds)")
    timestamps = df.iloc[:, 1].values
    
    # Extract quaternion (X, Y, Z, W)
    quat_x = df.iloc[:, quat_cols[0]].values
    quat_y = df.iloc[:, quat_cols[1]].values
    quat_z = df.iloc[:, quat_cols[2]].values
    quat_w = df.iloc[:, quat_cols[3]].values
    
    # Extract position (X, Y, Z) in mm, convert to meters
    pos_x = df.iloc[:, pos_cols[0]].values / 1000.0
    pos_y = df.iloc[:, pos_cols[1]].values / 1000.0
    pos_z = df.iloc[:, pos_cols[2]].values / 1000.0
    
    # Convert to rotation matrices
    rotation_matrices = []
    for x, y, z, w in zip(quat_x, quat_y, quat_z, quat_w):
        r = Rotation.from_quat([x, y, z, w])
        rotation_matrices.append(r.as_matrix())
    
    return {
        'timestamps': timestamps,
        'positions': np.column_stack([pos_x, pos_y, pos_z]),
        'rotation_matrices': np.array(rotation_matrices),
        'quaternions': np.column_stack([quat_x, quat_y, quat_z, quat_w])
    }

def load_b_csv(csv_file, has_timestamps=False):
    """
    Load B matrix CSV file (AprilTag poses relative to camera).
    
    Format options:
    1. Synchronized format: qx, qy, qz, qw, x, y, z (no header, no timestamps)
    2. Detection format: timestamp, tag_id, qx, qy, qz, qw, x, y, z (with header)
    Units: quaternion (normalized), translation in mm
    """
    # Try to detect format by reading first few lines
    with open(csv_file, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip() if first_line else ""
        
        # Check if it has a header (look for column names)
        has_header = any(keyword in first_line.lower() for keyword in 
                        ['timestamp', 'tag_id', 'qx', 'time'])
        
        # Check if first data line has timestamp-like values
        try:
            parts = first_line.split(',') if not has_header else second_line.split(',')
            if len(parts) >= 8:
                # Might be detection format with timestamps
                float(parts[0])  # Try to parse as timestamp
                has_timestamps = True
        except (ValueError, IndexError):
            has_timestamps = False
    
    skip_rows = 1 if has_header else 0
    df = pd.read_csv(csv_file, header=0 if has_header else None, skiprows=skip_rows)
    
    # Detect format based on number of columns and header
    if df.shape[1] == 9 and has_header:
        # Detection format: timestamp, tag_id, qx, qy, qz, qw, x, y, z
        timestamps = df.iloc[:, 0].values
        tag_ids = df.iloc[:, 1].values
        quat_x = df.iloc[:, 2].values
        quat_y = df.iloc[:, 3].values
        quat_z = df.iloc[:, 4].values
        quat_w = df.iloc[:, 5].values
        pos_x = df.iloc[:, 6].values / 1000.0  # mm to m
        pos_y = df.iloc[:, 7].values / 1000.0
        pos_z = df.iloc[:, 8].values / 1000.0
        has_timestamps = True
    elif df.shape[1] == 7:
        # Synchronized format: qx, qy, qz, qw, x, y, z (no timestamps)
        quat_x = df.iloc[:, 0].values
        quat_y = df.iloc[:, 1].values
        quat_z = df.iloc[:, 2].values
        quat_w = df.iloc[:, 3].values
        pos_x = df.iloc[:, 4].values / 1000.0  # mm to m
        pos_y = df.iloc[:, 5].values / 1000.0
        pos_z = df.iloc[:, 6].values / 1000.0
        timestamps = None
        tag_ids = None
    elif df.shape[1] == 8:
        # Format: timestamp, qx, qy, qz, qw, x, y, z
        timestamps = df.iloc[:, 0].values
        quat_x = df.iloc[:, 1].values
        quat_y = df.iloc[:, 2].values
        quat_z = df.iloc[:, 3].values
        quat_w = df.iloc[:, 4].values
        pos_x = df.iloc[:, 5].values / 1000.0
        pos_y = df.iloc[:, 6].values / 1000.0
        pos_z = df.iloc[:, 7].values / 1000.0
        tag_ids = None
        has_timestamps = True
    else:
        raise ValueError(f"Unexpected CSV format: {df.shape[1]} columns (expected 7, 8, or 9)")
    
    # Convert to rotation matrices
    rotation_matrices = []
    for x, y, z, w in zip(quat_x, quat_y, quat_z, quat_w):
        r = Rotation.from_quat([x, y, z, w])
        rotation_matrices.append(r.as_matrix())
    
    result = {
        'positions': np.column_stack([pos_x, pos_y, pos_z]),
        'rotation_matrices': np.array(rotation_matrices),
        'quaternions': np.column_stack([quat_x, quat_y, quat_z, quat_w])
    }
    
    if timestamps is not None:
        result['timestamps'] = timestamps
    if tag_ids is not None:
        result['tag_ids'] = tag_ids
    
    return result

def load_a_csv(csv_file):
    """Load A matrix CSV file (Optitrack poses, synchronized with B)."""
    return load_b_csv(csv_file, has_timestamps=False)  # Same format

def transform_to_world(tag_pos_cam, tag_rot_cam, camera_pos, camera_rot):
    """
    Transform tag pose from camera frame to world frame.
    
    Args:
        tag_pos_cam: Tag position in camera frame (3,)
        tag_rot_cam: Tag rotation matrix in camera frame (3, 3)
        camera_pos: Camera position in world frame (3,)
        camera_rot: Camera rotation matrix in world frame (3, 3)
    
    Returns:
        tag_pos_world: Tag position in world frame (3,)
        tag_rot_world: Tag rotation matrix in world frame (3, 3)
    """
    # Transform position: world_pos = camera_pos + camera_rot @ tag_pos_cam
    tag_pos_world = camera_pos + camera_rot @ tag_pos_cam
    
    # Transform rotation: world_rot = camera_rot @ tag_rot_cam
    tag_rot_world = camera_rot @ tag_rot_cam
    
    return tag_pos_world, tag_rot_world

def draw_coordinate_frame(ax, position, rotation_matrix, scale=0.003, alpha=1.0):
    """Draw a coordinate frame at the given position."""
    # X axis (red)
    x_end = position + rotation_matrix[:, 0] * scale
    ax.plot([position[0], x_end[0]], [position[1], x_end[1]], [position[2], x_end[2]], 
            'r-', linewidth=1.5, alpha=alpha)
    
    # Y axis (green)
    y_end = position + rotation_matrix[:, 1] * scale
    ax.plot([position[0], y_end[0]], [position[1], y_end[1]], [position[2], y_end[2]], 
            'g-', linewidth=1.5, alpha=alpha)
    
    # Z axis (blue)
    z_end = position + rotation_matrix[:, 2] * scale
    ax.plot([position[0], z_end[0]], [position[1], z_end[1]], [position[2], z_end[2]], 
            'b-', linewidth=1.5, alpha=alpha)

def calculate_global_max_range(optitrack_data, all_b_data=None, detection_data=None, use_detection_csv=False, show_in_world_frame=True):
    """
    Calculate the maximum range needed across all frames to keep scale constant.
    Returns the maximum distance from camera to any tag across all timesteps.
    """
    max_range = 0.5  # Minimum fallback
    
    if use_detection_csv and detection_data is not None:
        # For detection CSV mode: find max distance across all timestamps
        max_distances = []
        for timestamp in detection_data['timestamps']:
            detections = detection_data['detections_by_timestamp'][timestamp]
            if len(detections) == 0:
                continue
            
            if show_in_world_frame:
                # Find closest Optitrack pose for this timestamp
                optitrack_pose = find_closest_optitrack_pose(optitrack_data, timestamp)
                camera_pos = optitrack_pose['position']
                camera_rot = optitrack_pose['rotation']
                
                # Calculate distances from camera to all tags at this timestamp (world frame)
                for det in detections:
                    tag_pos_cam = det['position']
                    tag_rot_cam = det['rotation']
                    tag_pos_world, _ = transform_to_world(tag_pos_cam, tag_rot_cam, camera_pos, camera_rot)
                    distance = np.linalg.norm(tag_pos_world - camera_pos)
                    max_distances.append(distance)
            else:
                # Camera frame mode: distance from origin to tag in camera frame
                for det in detections:
                    tag_pos_cam = det['position']
                    distance = np.linalg.norm(tag_pos_cam)
                    max_distances.append(distance)
        
        if len(max_distances) > 0:
            max_range = max(max_distances) * 1.2  # Add 20% padding
    else:
        # Legacy mode: find max distance across all frames
        max_distances = []
        for frame_idx in range(len(optitrack_data['positions'])):
            if show_in_world_frame:
                camera_pos = optitrack_data['positions'][frame_idx]
                camera_rot = optitrack_data['rotation_matrices'][frame_idx]
            
            for b_data in all_b_data.values():
                # Check if B data exists for this frame
                if not b_data.get('has_data', np.ones(len(b_data['positions']), dtype=bool))[frame_idx]:
                    continue
                
                tag_pos_cam = b_data['positions'][frame_idx]
                if np.any(np.isnan(tag_pos_cam)):
                    continue
                
                if show_in_world_frame:
                    # Transform tag to world frame
                    tag_pos_world, _ = transform_to_world(
                        tag_pos_cam, b_data['rotation_matrices'][frame_idx], camera_pos, camera_rot
                    )
                    distance = np.linalg.norm(tag_pos_world - camera_pos)
                else:
                    # Camera frame mode: distance from origin to tag in camera frame
                    distance = np.linalg.norm(tag_pos_cam)
                
                max_distances.append(distance)
        
        if len(max_distances) > 0:
            max_range = max(max_distances) * 1.2  # Add 20% padding
    
    # Ensure minimum range
    if max_range < 0.5:
        max_range = 0.5
    
    return max_range

def get_video_frame_at_timestamp(video_cap, timestamp, video_fps, frame_idx=0):
    """
    Get video frame at a given timestamp.
    
    Args:
        video_cap: cv2.VideoCapture object
        timestamp: Target timestamp in seconds
        video_fps: Video FPS
        
    Returns:
        Video frame (RGB image) or None if not found
    """
    if video_cap is None or video_fps <= 0:
        return None
    
    try:
        # Calculate frame number from timestamp
        frame_number = int(timestamp * video_fps)
        
        # Ensure frame number is valid
        total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_number < 0:
            frame_number = 0
        elif frame_number >= total_frames:
            frame_number = max(0, total_frames - 1)
        
        # Debug on first frame
        if frame_idx == 0:
            print(f"DEBUG get_video_frame_at_timestamp:")
            print(f"  timestamp: {timestamp:.3f}s")
            print(f"  video_fps: {video_fps}")
            print(f"  calculated frame_number: {int(timestamp * video_fps)}")
            print(f"  clamped frame_number: {frame_number}")
            print(f"  total_frames: {total_frames}")
        
        # Set video position to this frame
        success = video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if not success and frame_idx == 0:
            print(f"  Warning: cv2.CAP_PROP_POS_FRAMES set returned False")
            
        ret, frame = video_cap.read()
        
        if frame_idx == 0:
            print(f"  ret: {ret}, frame is None: {frame is None}")
            if frame is not None:
                print(f"  frame.shape: {frame.shape}, frame.size: {frame.size}")
        
        if ret and frame is not None and frame.size > 0:
            # Convert BGR to RGB for matplotlib
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            if frame_idx == 0:
                print(f"  Successfully retrieved video frame!")
            return frame_rgb
        else:
            # Debug: print why frame wasn't read
            if frame_idx == 0:
                print(f"  Failed to read video frame:")
                print(f"    ret: {ret}")
                print(f"    frame is None: {frame is None}")
                if frame is not None:
                    print(f"    frame.size: {frame.size}")
    except Exception as e:
        print(f"  Exception reading video frame at timestamp {timestamp}: {e}")
        import traceback
        traceback.print_exc()
    
    return None

def visualize_frame(ax, optitrack_data, all_b_data, frame_idx, 
                   show_in_world_frame=True, show_camera_frame=True,
                   detection_data=None, use_detection_csv=False,
                   fixed_max_range=None, video_cap=None, video_fps=None,
                   detector=None):
    """
    Visualize a single synchronized frame with all tags.
    
    Args:
        ax: 3D matplotlib axis
        optitrack_data: Optitrack data dict (camera pose)
        all_b_data: Dictionary mapping tag_id -> B matrix data dict (legacy mode)
        frame_idx: Index into synchronized data
        show_in_world_frame: If True, show tags in world frame; if False, show relative to camera
        show_camera_frame: If True, draw camera coordinate frame
        detection_data: Detection CSV data dict (if using detection CSV mode)
        use_detection_csv: If True, use detection CSV mode
    """
    ax.clear()
    
    # Update video feed in separate OpenCV window if available
    if video_cap is not None and video_fps is not None and video_fps > 0:
        # Get current timestamp
        if use_detection_csv and detection_data is not None:
            timestamp = detection_data['timestamps'][frame_idx]
            detections = detection_data['detections_by_timestamp'][timestamp]
        else:
            timestamp = optitrack_data['timestamps'][frame_idx]
            detections = []
        
        # Get video frame at this timestamp
        video_frame = get_video_frame_at_timestamp(video_cap, timestamp, video_fps, frame_idx)
        if video_frame is not None:
            # Convert RGB back to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            
            # Draw tag detections using same logic as construct_B.py
            if detector is not None:
                try:
                    # Try to detect tags in the current video frame
                    detections_video = detector.detect(frame_bgr)
                    
                    if detections_video and len(detections_video) > 0:
                        # Handle different return formats from detector
                        for detection in detections_video:
                            if len(detection) == 4:
                                # Format: (tag_id, pose, corners, orientation)
                                tag_id, pose, corners, _orientation = detection
                            elif len(detection) == 3:
                                # Format: (tag_id, pose, corners)
                                tag_id, pose, corners = detection
                            elif len(detection) == 2:
                                # Format: (tag_id, corners)
                                tag_id, corners = detection
                                pose = None
                            else:
                                # Unknown format, skip
                                continue
                            
                            # Draw green rectangle around tag (same as construct_B.py)
                            corners_int = corners.astype(int)
                            cv2.polylines(frame_bgr, [corners_int], True, (0, 255, 0), 2)
                            
                            # Draw tag ID
                            center = corners_int.mean(axis=0).astype(int)
                            cv2.putText(frame_bgr, f"ID:{tag_id}", tuple(center),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            
                            # Draw distance if pose is available (same as construct_B.py)
                            if pose is not None and hasattr(pose, 't'):
                                distance = np.linalg.norm(pose.t)
                                cv2.putText(frame_bgr, f"{distance:.2f}m", 
                                           (center[0], center[1] + 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception as e:
                    print(f"Error during tag detection in video frame: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Add text overlay with timestamp and frame info
            cv2.putText(frame_bgr, f'Time: {timestamp:.3f}s', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame_bgr, f'Frame: {frame_idx}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if use_detection_csv:
                tag_count = len(detections)
                cv2.putText(frame_bgr, f'Tags: {tag_count}', (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display in separate window
            try:
                cv2.imshow('Video Feed', frame_bgr)
                cv2.waitKey(1)  # Non-blocking wait to update display
            except Exception as e:
                print(f"Error displaying video frame: {e}")
        else:
            # Debug: print why frame is None
            if frame_idx == 0:
                print(f"DEBUG: video_frame is None at timestamp {timestamp:.3f}s, frame_idx {frame_idx}")
                print(f"  video_cap is None: {video_cap is None}")
                print(f"  video_fps: {video_fps}")
                print(f"  video_fps > 0: {video_fps > 0 if video_fps else False}")
    
    # Plot full Optitrack trajectory as semi-transparent line (visible across all steps)
    if len(optitrack_data['positions']) > 1:
        ax.plot(optitrack_data['positions'][:, 0], 
                optitrack_data['positions'][:, 1], 
                optitrack_data['positions'][:, 2],
                'gray', linewidth=1, alpha=0.3, linestyle='-', label='Optitrack Trajectory', zorder=1)
    
    if use_detection_csv and detection_data is not None:
        # Get timestamp from detection CSV
        timestamp = detection_data['timestamps'][frame_idx]
        
        # Find closest Optitrack pose
        optitrack_pose = find_closest_optitrack_pose(optitrack_data, timestamp)
        camera_pos = optitrack_pose['position']
        camera_rot = optitrack_pose['rotation']
        
        # Get detections for this timestamp
        detections = detection_data['detections_by_timestamp'][timestamp]
        
    else:
        # Legacy mode: Get camera pose from Optitrack
        camera_pos = optitrack_data['positions'][frame_idx]
        camera_rot = optitrack_data['rotation_matrices'][frame_idx]
        detections = None
    
    if use_detection_csv and detections is not None:
        # Use detection CSV mode
        if len(detections) == 0:
            # No tags detected at this timestamp
            tag_ids_detected = []
        else:
            tag_ids_detected = [det['tag_id'] for det in detections]
        
        # Color map for different tags
        max_tag_id = max(tag_ids_detected) if tag_ids_detected else 0
        colors = plt.cm.tab20(np.linspace(0, 1, max(max_tag_id + 1, 20)))
        
        if show_in_world_frame:
            # Draw camera frame
            if show_camera_frame:
                draw_coordinate_frame(ax, camera_pos, camera_rot, scale=0.005, alpha=0.7)
                ax.scatter(*camera_pos, c='blue', s=50, marker='o', label='Camera', alpha=0.7, zorder=10)
            
            # Draw all detected tags in world frame
            all_tag_positions = []
            for det in detections:
                tag_id = det['tag_id']
                tag_pos_cam = det['position']
                tag_rot_cam = det['rotation']
                color = colors[tag_id % len(colors)]
                
                # Transform tag to world frame
                tag_pos_world, tag_rot_world = transform_to_world(
                    tag_pos_cam, tag_rot_cam, camera_pos, camera_rot
                )
                all_tag_positions.append(tag_pos_world)
                
                # Draw tag frame in world coordinates
                draw_coordinate_frame(ax, tag_pos_world, tag_rot_world, scale=0.003, alpha=0.8)
                ax.scatter(*tag_pos_world, c=[color], s=80, marker='^', 
                          label=f'Tag {tag_id}', alpha=0.8, zorder=5)
                
                # Draw line from camera to tag
                ax.plot([camera_pos[0], tag_pos_world[0]], 
                       [camera_pos[1], tag_pos_world[1]], 
                       [camera_pos[2], tag_pos_world[2]], 
                       'k--', linewidth=1, alpha=0.3)
            
            # Set axis limits for detection CSV world frame (center on camera, fixed scale)
            max_range = fixed_max_range if fixed_max_range is not None else 0.5
            
            # Center view on camera position with uniform spacing
            ax.set_xlim(camera_pos[0] - max_range, camera_pos[0] + max_range)
            ax.set_ylim(camera_pos[1] - max_range, camera_pos[1] + max_range)
            ax.set_zlim(camera_pos[2] - max_range, camera_pos[2] + max_range)
        
        else:
            # Camera frame mode
            ax.scatter(0, 0, 0, c='blue', s=50, marker='o', label='Camera', alpha=0.7, zorder=10)
            draw_coordinate_frame(ax, np.zeros(3), np.eye(3), scale=0.005, alpha=0.7)
            
            for det in detections:
                tag_id = det['tag_id']
                tag_pos_cam = det['position']
                tag_rot_cam = det['rotation']
                color = colors[tag_id % len(colors)]
                
                draw_coordinate_frame(ax, tag_pos_cam, tag_rot_cam, scale=0.003, alpha=0.8)
                ax.scatter(*tag_pos_cam, c=[color], s=80, marker='^', 
                          label=f'Tag {tag_id}', alpha=0.8, zorder=5)
                
                ax.plot([0, tag_pos_cam[0]], [0, tag_pos_cam[1]], [0, tag_pos_cam[2]], 
                       'k--', linewidth=1, alpha=0.3)
            
            # Set axis limits for detection CSV camera frame (fixed scale)
            max_range = fixed_max_range if fixed_max_range is not None else 0.5
            
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
    
    else:
        # Legacy mode: Use B CSV data
        # Color map for different tags
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_b_data)))
        
        if show_in_world_frame:
            # Draw camera frame
            if show_camera_frame:
                draw_coordinate_frame(ax, camera_pos, camera_rot, scale=0.005, alpha=0.7)
                ax.scatter(*camera_pos, c='blue', s=50, marker='o', label='Camera', alpha=0.7, zorder=10)
            
            # Draw all tags in world frame (only if B data exists for this frame)
            all_tag_positions = []
            for (tag_id, b_data), color in zip(all_b_data.items(), colors):
                # Check if B data exists for this frame
                if not b_data.get('has_data', np.ones(len(b_data['positions']), dtype=bool))[frame_idx]:
                    continue  # Skip this tag for this frame
                
                # Get tag pose from B matrix (relative to camera)
                tag_pos_cam = b_data['positions'][frame_idx]
                tag_rot_cam = b_data['rotation_matrices'][frame_idx]
                
                # Check for NaN (no data)
                if np.any(np.isnan(tag_pos_cam)) or np.any(np.isnan(tag_rot_cam)):
                    continue
                
                # Transform tag to world frame
                tag_pos_world, tag_rot_world = transform_to_world(
                    tag_pos_cam, tag_rot_cam, camera_pos, camera_rot
                )
                all_tag_positions.append(tag_pos_world)
                
                # Draw tag frame in world coordinates
                draw_coordinate_frame(ax, tag_pos_world, tag_rot_world, scale=0.003, alpha=0.8)
                ax.scatter(*tag_pos_world, c=[color], s=80, marker='^', 
                          label=f'Tag {tag_id}', alpha=0.8, zorder=5)
                
                # Draw line from camera to tag
                ax.plot([camera_pos[0], tag_pos_world[0]], 
                       [camera_pos[1], tag_pos_world[1]], 
                       [camera_pos[2], tag_pos_world[2]], 
                       'k--', linewidth=1, alpha=0.3)
            
            # Set axis limits for legacy mode world frame (center on camera, fixed scale)
            max_range = fixed_max_range if fixed_max_range is not None else 0.5
            
            # Center view on camera position with uniform spacing
            ax.set_xlim(camera_pos[0] - max_range, camera_pos[0] + max_range)
            ax.set_ylim(camera_pos[1] - max_range, camera_pos[1] + max_range)
            ax.set_zlim(camera_pos[2] - max_range, camera_pos[2] + max_range)
        
        else:
            # Legacy mode camera frame
            # Show in camera frame (camera at origin)
            ax.scatter(0, 0, 0, c='blue', s=50, marker='o', label='Camera', alpha=0.7, zorder=10)
            draw_coordinate_frame(ax, np.zeros(3), np.eye(3), scale=0.005, alpha=0.7)
            
            # Draw all tags relative to camera (only if B data exists for this frame)
            all_tag_positions_cam = []
            for (tag_id, b_data), color in zip(all_b_data.items(), colors):
                # Check if B data exists for this frame
                if not b_data.get('has_data', np.ones(len(b_data['positions']), dtype=bool))[frame_idx]:
                    continue  # Skip this tag for this frame
                
                tag_pos_cam = b_data['positions'][frame_idx]
                tag_rot_cam = b_data['rotation_matrices'][frame_idx]
                
                # Check for NaN (no data)
                if np.any(np.isnan(tag_pos_cam)) or np.any(np.isnan(tag_rot_cam)):
                    continue
                
                all_tag_positions_cam.append(tag_pos_cam)
                
                draw_coordinate_frame(ax, tag_pos_cam, tag_rot_cam, scale=0.003, alpha=0.8)
                ax.scatter(*tag_pos_cam, c=[color], s=80, marker='^', 
                          label=f'Tag {tag_id}', alpha=0.8, zorder=5)
                
                # Draw line from origin to tag
                ax.plot([0, tag_pos_cam[0]], [0, tag_pos_cam[1]], [0, tag_pos_cam[2]], 
                       'k--', linewidth=1, alpha=0.3)
            
            # Set axis limits for legacy mode camera frame (fixed scale)
            max_range = fixed_max_range if fixed_max_range is not None else 0.5
            
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)
    
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    
    # Create legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    if unique:
        ax.legend(*zip(*unique), loc='upper right')
    else:
        ax.legend(loc='upper right')
    
    ax.grid(True, alpha=0.3)

def match_timestamps(timestamps_A, timestamps_B, threshold=0.02):
    """
    Match timestamps between two datasets.
    Returns indices for matched pairs.
    """
    matched_A = []
    matched_B = []
    
    for idx_B, ts_B in enumerate(timestamps_B):
        # Find closest timestamp in A
        diffs = np.abs(timestamps_A - ts_B)
        closest_idx = np.argmin(diffs)
        if diffs[closest_idx] < threshold:
            matched_A.append(closest_idx)
            matched_B.append(idx_B)
    
    return np.array(matched_A), np.array(matched_B)

def load_all_b_csvs_for_camera(b_csv_dir, camera_id):
    """
    Load all B CSV files for a given camera.
    
    Args:
        b_csv_dir: Directory containing B CSV files
        camera_id: Camera ID to load
    
    Returns:
        Dictionary mapping tag_id -> b_data dict
    """
    b_csv_dir = Path(b_csv_dir)
    pattern = f"tag_*_cam_{camera_id}_B.csv"
    b_files = list(b_csv_dir.glob(pattern))
    
    if len(b_files) == 0:
        # Try alternative pattern
        pattern = f"tag_*_cam_{camera_id}_B.csv"
        b_files = list(b_csv_dir.glob(pattern))
    
    if len(b_files) == 0:
        raise ValueError(f"No B CSV files found for camera {camera_id} in {b_csv_dir}")
    
    print(f"Found {len(b_files)} B CSV files for camera {camera_id}")
    
    all_b_data = {}
    for b_file in sorted(b_files):
        # Extract tag ID from filename
        parts = b_file.stem.split('_')
        if 'tag' in parts:
            tag_idx = parts.index('tag')
            if tag_idx + 1 < len(parts):
                try:
                    tag_id = int(parts[tag_idx + 1])
                    print(f"  Loading tag {tag_id} from {b_file.name}...")
                    b_data = load_b_csv(str(b_file))
                    all_b_data[tag_id] = b_data
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not extract tag ID from {b_file.name}: {e}")
    
    return all_b_data

def load_detection_csv(csv_file):
    """
    Load AprilTag detection CSV file.
    
    Format: timestamp, tag_id, orientation, qx, qy, qz, qw, x, y, z
    Multiple rows can have the same timestamp (one per tag detected)
    NaN rows indicate no tags detected at that timestamp
    """
    df = pd.read_csv(csv_file)
    
    # Debug: Print column names to see what we're working with
    print(f"DEBUG: CSV columns: {list(df.columns)}")
    print(f"DEBUG: First row: {df.iloc[0] if len(df) > 0 else 'Empty'}")
    
    # Group by timestamp
    timestamps = []
    detections_by_timestamp = {}
    
    for _, row in df.iterrows():
        ts = row['timestamp']
        
        # Check if tag_id is NaN (no detection)
        if pd.isna(row['tag_id']):
            # Still record this timestamp, but with no detections
            if ts not in detections_by_timestamp:
                detections_by_timestamp[ts] = []
                timestamps.append(ts)
        else:
            tag_id = int(row['tag_id'])
            
            # Extract pose data - skip the 'orientation' column
            # Assuming columns: timestamp, tag_id, orientation, qx, qy, qz, qw, x, y, z
            qx = row['qx']
            qy = row['qy']
            qz = row['qz']
            qw = row['qw']
            x = row['x'] / 1000.0  # mm to m
            y = row['y'] / 1000.0
            z = row['z'] / 1000.0
            
            # Convert to rotation matrix
            r = Rotation.from_quat([qx, qy, qz, qw])
            R = r.as_matrix()
            
            if ts not in detections_by_timestamp:
                detections_by_timestamp[ts] = []
                timestamps.append(ts)
            
            detections_by_timestamp[ts].append({
                'tag_id': tag_id,
                'position': np.array([x, y, z]),
                'rotation': R,
                'quaternion': np.array([qx, qy, qz, qw])
            })
    
    # Sort timestamps
    timestamps = sorted(set(timestamps))
    
    print(f"DEBUG: Loaded {len(timestamps)} unique timestamps")
    print(f"DEBUG: First few timestamps: {timestamps[:5] if len(timestamps) > 5 else timestamps}")
    
    return {
        'timestamps': np.array(timestamps),
        'detections_by_timestamp': detections_by_timestamp
    }

def find_closest_optitrack_pose(optitrack_data, target_timestamp):
    """Find closest Optitrack pose to target timestamp."""
    optitrack_timestamps = optitrack_data['timestamps']
    
    # Find closest index
    idx = np.argmin(np.abs(optitrack_timestamps - target_timestamp))
    
    return {
        'position': optitrack_data['positions'][idx],
        'rotation': optitrack_data['rotation_matrices'][idx],
        'timestamp': optitrack_timestamps[idx]
    }

def main():
    parser = argparse.ArgumentParser(description="Visualize Optitrack and AprilTag poses together")
    parser.add_argument("--optitrack", required=True, help="Optitrack CSV file")
    parser.add_argument("--detection-csv", default=None, help="AprilTag detection CSV file (e.g., Left_success1_cut.csv)")
    parser.add_argument("--b-csv", default=None, help="Single B matrix CSV file (tag poses relative to camera) - alternative to detection-csv")
    parser.add_argument("--b-csv-dir", default=None, help="Directory containing B CSV files (will load all tags for a camera) - alternative to detection-csv")
    parser.add_argument("--camera-id", type=int, default=None, help="Camera ID (required if using --b-csv-dir)")
    parser.add_argument("--camera-frame", action="store_true", help="Show in camera frame (camera at origin)")
    parser.add_argument("--start-frame", type=int, default=0, help="Starting frame index")
    parser.add_argument("--video", default=None, help="Path to video file to display in corner (synchronized with visualization)")
    parser.add_argument("--camera-matrix", default=None, help="Path to camera calibration file (.yaml or .npz format) for tag detection")
    parser.add_argument("--tag-size", type=float, default=0.15, help="Tag size in meters (default: 0.15)")
    parser.add_argument("--tag-family", type=str, default='tag36h11', help="Tag family (default: tag36h11)")
    args = parser.parse_args()
    
    print(f"Loading Optitrack data from {args.optitrack}...")
    optitrack_data = load_optitrack_csv(args.optitrack)
    print(f"  Loaded {len(optitrack_data['timestamps'])} poses")
    print(f"  Time range: {optitrack_data['timestamps'][0]:.3f} to {optitrack_data['timestamps'][-1]:.3f} s")
    
    # Load detection data
    if args.detection_csv:
        # Use detection CSV as primary timeline
        print(f"\nLoading detection data from {args.detection_csv}...")
        detection_data = load_detection_csv(args.detection_csv)
        n_frames = len(detection_data['timestamps'])
        print(f"  Loaded {n_frames} unique timestamps")
        print(f"  Time range: {detection_data['timestamps'][0]:.3f} to {detection_data['timestamps'][-1]:.3f} s")
        
        # Count detections per timestamp
        detections_per_frame = [len(detection_data['detections_by_timestamp'][ts]) 
                               for ts in detection_data['timestamps']]
        frames_with_detections = sum(1 for count in detections_per_frame if count > 0)
        print(f"  Frames with detections: {frames_with_detections}/{n_frames}")
        
        use_detection_csv = True
        
    elif args.b_csv_dir and args.camera_id is not None:
        use_detection_csv = False
        # Load all tags for this camera
        print(f"\nLoading all B matrix files for camera {args.camera_id} from {args.b_csv_dir}...")
        all_b_data = load_all_b_csvs_for_camera(args.b_csv_dir, args.camera_id)
        
        if len(all_b_data) == 0:
            raise ValueError(f"No B CSV files loaded for camera {args.camera_id}")
        
        # Check that all have same length
        lengths = [len(b_data['positions']) for b_data in all_b_data.values()]
        if len(set(lengths)) > 1:
            print(f"  Warning: B CSV files have different lengths: {lengths}")
            min_length = min(lengths)
            print(f"  Truncating all to minimum length: {min_length}")
            for tag_id in all_b_data:
                all_b_data[tag_id]['positions'] = all_b_data[tag_id]['positions'][:min_length]
                all_b_data[tag_id]['rotation_matrices'] = all_b_data[tag_id]['rotation_matrices'][:min_length]
        
        n_b = lengths[0] if len(set(lengths)) == 1 else min(lengths)
        
    elif args.b_csv:
        # Load single B CSV file
        print(f"\nLoading B matrix data from {args.b_csv}...")
        b_data = load_b_csv(args.b_csv)
        print(f"  Loaded {len(b_data['positions'])} poses")
        # Convert to dict format for consistency
        tag_id = args.camera_id if args.camera_id is not None else 0
        all_b_data = {tag_id: b_data}
        n_b = len(b_data['positions'])
    else:
        raise ValueError("Must provide either --detection-csv or --b-csv or --b-csv-dir with --camera-id")
    
    if use_detection_csv:
        # Use detection CSV as primary timeline
        n_frames = len(detection_data['timestamps'])
        print(f"\nVisualization will show {n_frames} frames from detection CSV")
    else:
        # Legacy mode: Use Optitrack as primary timeline
        n_optitrack = len(optitrack_data['timestamps'])
        
        print(f"\nData lengths:")
        print(f"  Optitrack: {n_optitrack} frames (full timeline)")
        print(f"  B matrix: {n_b} frames (synchronized subset)")
        
        if n_optitrack != n_b:
            print(f"\nNote: B CSV has fewer frames than Optitrack.")
            print(f"  Will show all {n_optitrack} Optitrack frames.")
            print(f"  Tags will only appear for the {n_b} frames with B data.")
            
            # Extend B data to match Optitrack length by padding with NaN
            for tag_id in all_b_data:
                b_positions = all_b_data[tag_id]['positions']
                b_rotations = all_b_data[tag_id]['rotation_matrices']
                
                extended_positions = np.full((n_optitrack, 3), np.nan)
                extended_rotations = np.full((n_optitrack, 3, 3), np.nan)
                
                if n_b <= n_optitrack:
                    extended_positions[:n_b] = b_positions
                    extended_rotations[:n_b] = b_rotations
                else:
                    extended_positions = b_positions[:n_optitrack]
                    extended_rotations = b_rotations[:n_optitrack]
                
                all_b_data[tag_id]['positions'] = extended_positions
                all_b_data[tag_id]['rotation_matrices'] = extended_rotations
                all_b_data[tag_id]['has_data'] = ~np.isnan(extended_positions[:, 0])
        else:
            for tag_id in all_b_data:
                all_b_data[tag_id]['has_data'] = np.ones(n_optitrack, dtype=bool)
        
        n_frames = n_optitrack
        print(f"\nVisualization will show {n_frames} frames")
        print(f"  Tags loaded: {sorted(all_b_data.keys())}")
        print(f"  Tags visible in: {n_b} frames")
    
    # Calculate global fixed max range for constant scale
    print("\nCalculating global scale range...")
    show_in_world_frame = not args.camera_frame
    fixed_max_range = calculate_global_max_range(
        optitrack_data, 
        all_b_data if not use_detection_csv else None,
        detection_data if use_detection_csv else None,
        use_detection_csv,
        show_in_world_frame=show_in_world_frame
    )
    print(f"  Fixed scale range: {fixed_max_range:.3f} m")
    
    # Initialize AprilTag detector if video is provided
    detector = None
    if args.video and AprilTagPoseEstimator is not None:
        # Load camera matrix for tag detection
        if args.camera_matrix:
            calib_path = Path(args.camera_matrix)
            if calib_path.suffix.lower() == '.yaml' or calib_path.suffix.lower() == '.yml':
                print(f"\nLoading camera matrix from {args.camera_matrix}...")
                with open(calib_path, 'r') as f:
                    calib_data = yaml.safe_load(f)
                K = np.array(calib_data['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
                dist = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float32)
            else:
                # Load from npz file
                calib_data = np.load(args.camera_matrix)
                K = calib_data['K'].astype(np.float32)
                dist = calib_data['dist'].astype(np.float32)
        else:
            # Default camera matrix (will be less accurate)
            print("Warning: No camera matrix provided. Using default camera matrix for tag detection.")
            print("  For accurate tag detection, provide --camera-matrix")
            h, w = 480, 640  # Default video dimensions
            K = np.array([
                [w * 0.8, 0, w / 2],
                [0, w * 0.8, h / 2],
                [0, 0, 1]
            ], dtype=np.float32)
            dist = np.zeros(5)
        
        # Initialize detector (same as construct_B.py)
        detector = AprilTagPoseEstimator(
            K,
            dist,
            tag_size=args.tag_size,
            family=args.tag_family
        )
        print(f"  AprilTag detector initialized (tag_size={args.tag_size}m, family={args.tag_family})")
    elif args.video and AprilTagPoseEstimator is None:
        print("Warning: Could not initialize AprilTag detector. Video will be shown without tag detection.")
    
    # Load video if provided
    video_cap = None
    video_fps = None
    if args.video:
        video_path = Path(args.video)
        # Try to resolve relative paths from script directory
        if not video_path.exists() and not video_path.is_absolute():
            script_dir = Path(__file__).parent
            video_path = script_dir / video_path
        if not video_path.exists():
            print(f"Warning: Video file not found: {args.video}")
            print(f"  Tried: {Path(args.video)}")
            print(f"  Tried: {script_dir / Path(args.video)}")
        else:
            print(f"\nLoading video: {args.video}")
            video_cap = cv2.VideoCapture(str(video_path))
            if not video_cap.isOpened():
                print(f"Warning: Could not open video file: {args.video}")
                video_cap = None
            else:
                video_fps = video_cap.get(cv2.CAP_PROP_FPS)
                total_video_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_duration = total_video_frames / video_fps if video_fps > 0 else 0
                print(f"  Video FPS: {video_fps:.2f}, Total frames: {total_video_frames}, Duration: {video_duration:.2f}s")
    
    # Create interactive visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Video will be displayed in separate OpenCV window
    if video_cap is not None and video_fps is not None and video_fps > 0:
        print(f"  Video will be displayed in separate OpenCV window")
        print(f"  Video window will sync with visualization timestamps")
    
    current_frame = args.start_frame
    show_in_world_frame = not args.camera_frame
    
    def update_plot():
        visualize_frame(ax, optitrack_data, 
                       all_b_data if not use_detection_csv else None, 
                       current_frame, 
                       show_in_world_frame=show_in_world_frame,
                       detection_data=detection_data if use_detection_csv else None,
                       use_detection_csv=use_detection_csv,
                       fixed_max_range=fixed_max_range,
                       video_cap=video_cap,
                       video_fps=video_fps,
                       detector=detector)
        
        # Add frame info
        if use_detection_csv:
            timestamp = detection_data['timestamps'][current_frame]
            detections = detection_data['detections_by_timestamp'][timestamp]
            tag_ids_detected = sorted([det['tag_id'] for det in detections])
            frame_info = f"Frame: {current_frame}/{n_frames-1} | Time: {timestamp:.3f}s | Tags: {tag_ids_detected}"
        else:
            timestamp = optitrack_data['timestamps'][current_frame]
            tags_with_data = []
            for tag_id, b_data in all_b_data.items():
                if b_data.get('has_data', np.ones(len(b_data['positions']), dtype=bool))[current_frame]:
                    if not np.any(np.isnan(b_data['positions'][current_frame])):
                        tags_with_data.append(tag_id)
            frame_info = f"Frame: {current_frame}/{n_frames-1} | Time: {timestamp:.3f}s | Tags visible: {sorted(tags_with_data)}"
        
        ax.set_title(frame_info, fontsize=12, fontweight='bold')
        plt.draw()  # Force immediate redraw
    
    def on_key(event):
        nonlocal current_frame, show_in_world_frame
        
        if event.key == 'right' or event.key == 'd':
            current_frame = min(current_frame + 1, n_frames - 1)
            update_plot()
        elif event.key == 'left' or event.key == 'a':
            current_frame = max(current_frame - 1, 0)
            update_plot()
        elif event.key == 'home' or event.key == 'h':
            current_frame = 0
            update_plot()
        elif event.key == 'end' or event.key == 'e':
            current_frame = n_frames - 1
            update_plot()
        elif event.key == ' ':
            # Toggle frame reference
            show_in_world_frame = not show_in_world_frame
            update_plot()
        elif event.key == 'q':
            plt.close()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial plot
    update_plot()
    
    print(f"\n{'='*60}")
    print("Interactive Controls:")
    print(f"{'='*60}")
    print("  LEFT ARROW / 'a'  - Previous frame")
    print("  RIGHT ARROW / 'd' - Next frame")
    print("  HOME / 'h'        - First frame")
    print("  END / 'e'         - Last frame")
    print("  SPACE             - Toggle world/camera frame")
    print("  'q'               - Quit")
    print(f"{'='*60}\n")
    
    plt.show()
    
    # Clean up video capture and OpenCV windows
    if video_cap is not None:
        video_cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()