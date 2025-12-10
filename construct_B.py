"""
AprilTag detection and pose estimation.

Detects AprilTags in images and estimates their 6-DoF pose
relative to the camera frame (B_i in the RWHEC formulation).
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import yaml

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.se3_utils import SE3


class AprilTagDetector:
    """
    Detect AprilTags and estimate their poses in camera frame.
    
    This computes B_i = T_{tag}^{camera} for the RWHEC equation A_i X = Y B_i
    """
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size: float,
        family: str,
    ):
        """
        Initialize AprilTag detector.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3)
            tag_size: Physical size of tag in meters
            family: AprilTag family ('tag36h11', 'tag25h9', etc.)
        """
        self.K = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.family = family
        
        # Try to import apriltag library
        try:
            import apriltag
            self.detector = apriltag.Detector(
                apriltag.DetectorOptions(families=family)
            )
            self._use_apriltag = True
        except ImportError:
            # Fall back to OpenCV ArUco if apriltag not available
            print("Warning: apriltag library not found, using OpenCV ArUco")
            self._use_apriltag = False
            self._setup_aruco()
        
        # 3D coordinates of tag corners in tag frame
        # Tag frame: origin at center, X right, Y up, Z out of tag
        s = tag_size / 2
        self.object_points = np.array([
            [-s, -s, 0],  # Bottom-left
            [ s, -s, 0],  # Bottom-right
            [ s,  s, 0],  # Top-right
            [-s,  s, 0],  # Top-left
        ], dtype=np.float32)
    
    def _setup_aruco(self):
        """Set up OpenCV ArUco detector as fallback."""
        # Map AprilTag families to ArUco dictionaries
        aruco_dict_map = {
            'tag36h11': cv2.aruco.DICT_APRILTAG_36h11,
            'tag25h9': cv2.aruco.DICT_APRILTAG_25h9,
            'tag16h5': cv2.aruco.DICT_APRILTAG_16h5,
        }
        
        dict_id = aruco_dict_map.get(self.family, cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def detect(self, image: np.ndarray) -> List[Tuple[int, SE3, np.ndarray]]:
        """
        Detect AprilTags in image and estimate their poses.
        
        Args:
            image: Grayscale or BGR image
            
        Returns:
            List of (tag_id, pose, corners) tuples where:
                - tag_id: Integer ID of the detected tag
                - pose: SE3 transformation from tag frame to camera frame
                - corners: 4x2 array of corner pixel coordinates
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self._use_apriltag:
            return self._detect_apriltag(gray)
        else:
            return self._detect_aruco(gray)
    
    def _detect_apriltag(self, gray: np.ndarray) -> List[Tuple[int, SE3, np.ndarray]]:
        """Detect using apriltag library."""
        import apriltag
        
        detections = self.detector.detect(gray)
        results = []
        
        for det in detections:
            # Get corner points (already in correct order)
            corners = det.corners.astype(np.float32)
            
            # Solve PnP for pose estimation
            pose = self._estimate_pose(corners)
            
            if pose is not None:
                results.append((det.tag_id, pose, corners))
        
        return results
    
    def _detect_aruco(self, gray: np.ndarray) -> List[Tuple[int, SE3, np.ndarray]]:
        """Detect using OpenCV ArUco."""
        corners_list, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        results = []
        
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                corners = corners_list[i].reshape(4, 2).astype(np.float32)
                
                # Solve PnP for pose estimation
                pose = self._estimate_pose(corners)
                
                if pose is not None:
                    results.append((int(tag_id), pose, corners))
        
        return results
    
    def _estimate_pose(self, corners: np.ndarray) -> Optional[SE3]:
        """
        Estimate tag pose from corner detections using PnP.
        
        Args:
            corners: 4x2 array of corner pixel coordinates
            
        Returns:
            SE3 pose of tag in camera frame, or None if PnP fails
        """
        # Use IPPE_SQUARE for planar targets (better for squares)
        success, rvec, tvec = cv2.solvePnP(
            self.object_points,
            corners,
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if not success:
            # Try iterative method as fallback
            success, rvec, tvec = cv2.solvePnP(
                self.object_points,
                corners,
                self.K,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return SE3(R, tvec.flatten())
        
        return None
    
    def detect_board(
        self,
        image: np.ndarray,
        board_config: Dict[int, np.ndarray]
    ) -> Optional[SE3]:
        """
        Detect AprilTag board and estimate its pose using all visible tags.
        
        Args:
            image: Grayscale or BGR image
            board_config: Dictionary mapping tag_id -> SE3 offset from board center
                         Each entry defines where that tag is relative to board frame
                         
        Returns:
            Board pose in camera frame (SE3), or None if detection fails
        """
        detections = self.detect(image)
        
        if len(detections) == 0:
            return None
        
        # Collect all object points (in board frame) and image points
        object_points = []
        image_points = []
        
        for tag_id, _, corners in detections:
            if tag_id in board_config:
                tag_offset = board_config[tag_id]  # SE3 from board to tag
                
                # Transform tag corners to board frame
                for i, local_corner in enumerate(self.object_points):
                    # Corner in board frame = tag_offset @ corner_in_tag_frame
                    if isinstance(tag_offset, SE3):
                        board_corner = tag_offset.transform_point(local_corner)
                    else:
                        # Assume it's a translation offset
                        board_corner = local_corner + np.array(tag_offset)
                    
                    object_points.append(board_corner)
                    image_points.append(corners[i])
        
        if len(object_points) < 4:
            return None
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP with all points
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            R, _ = cv2.Rodrigues(rvec)
            return SE3(R, tvec.flatten())
        
        return None


def create_board_config(
    n_rows: int,
    n_cols: int,
    tag_size: float,
    tag_spacing: float,
    start_id: int = 0
) -> Dict[int, np.ndarray]:
    """
    Create board configuration for a regular grid of AprilTags.
    
    Args:
        n_rows: Number of rows
        n_cols: Number of columns
        tag_size: Size of each tag in meters
        tag_spacing: Spacing between tag centers in meters
        start_id: ID of the first tag (top-left)
        
    Returns:
        Dictionary mapping tag_id -> (x, y, z) offset from board center
    """
    config = {}
    
    # Board center
    center_x = (n_cols - 1) * tag_spacing / 2
    center_y = (n_rows - 1) * tag_spacing / 2
    
    tag_id = start_id
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tag_spacing - center_x
            y = (n_rows - 1 - row) * tag_spacing - center_y  # Y increases upward
            z = 0.0
            
            config[tag_id] = np.array([x, y, z])
            tag_id += 1
    
    return config


if __name__ == '__main__':
    """Process video file and extract AprilTag poses to CSV."""
    import argparse
    import csv
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Extract AprilTag poses from video to CSV')
    parser.add_argument('video', type=str, help='Path to video file (.mov, .mp4, .webm, .mkv, etc)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file path (default: video_name_poses.csv)')
    parser.add_argument('--tag-size', type=float, default=0.1, help='Tag size in meters')
    parser.add_argument('--tag-id', type=int, default=None, help='Specific tag ID to track (default: use first detected tag)')
    parser.add_argument('--camera-matrix', type=str, default=None, help='Path to camera calibration file (.yaml or .npz format)')
    parser.add_argument('--num-samples', type=int, default=None, help='Total number of frames to sample from the video (e.g., 30 to sample 30 frames evenly spaced). If None, processes all frames.')
    parser.add_argument('--show-visualization', action='store_true', help='Show visualization playback with detected AprilTags overlaid')
    parser.add_argument('--tag-family', type=str, default='tag36h11', help='Tag family (default: tag36h11)')
    args = parser.parse_args()
    
    # Load camera matrix
    if args.camera_matrix:
        calib_path = Path(args.camera_matrix)
        if calib_path.suffix.lower() == '.yaml' or calib_path.suffix.lower() == '.yml':
            # Load from YAML file
            print("Using camera matrix from YAML file: ", calib_path)
            with open(calib_path, 'r') as f:
                calib_data = yaml.safe_load(f)
            print("Camera matrix data: ", calib_data)
            # Extract camera matrix (3x3) from YAML format
            cm_data = calib_data['camera_matrix']['data']
            K = np.array(cm_data, dtype=np.float32).reshape(3, 3)
            # Extract distortion coefficients
            dist = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float32)
        else:
            # Load from npz file (backwards compatibility)
            calib_data = np.load(args.camera_matrix)
            K = calib_data['K'].astype(np.float32)
            dist = calib_data['dist'].astype(np.float32)
    else:
        # Default camera matrix (adjust for your camera)
        print("Warning: Using default camera matrix. For accurate results, provide --camera-matrix")
        K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        dist = np.zeros(5)
    
    # Initialize detector
    detector = AprilTagDetector(
        K,
        dist,
        tag_size=args.tag_size,
        family=args.tag_family
    )
    
    # Open video file
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        # Check if OpenCV has FFmpeg support
        build_info = cv2.getBuildInformation()
        has_ffmpeg = 'FFMPEG: YES' in build_info or 'ffmpeg: YES' in build_info.lower()
        
        error_msg = f"Could not open video file: {args.video}\n"
        if not has_ffmpeg:
            error_msg += "Note: OpenCV may not be built with FFmpeg support.\n"
        error_msg += f"Supported formats depend on your OpenCV build. Common formats: .mp4, .mov, .webm, .mkv, .avi"
        raise ValueError(error_msg)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps if fps > 0 else 0
    
    # Determine frame skipping
    if args.num_samples is not None:
        if total_frames > 0 and args.num_samples > 0:
            frame_skip = max(1, int(total_frames / args.num_samples))
            expected_samples = total_frames // frame_skip
            print(f"Frame skipping: processing 1 frame every {frame_skip} frames")
            print(f"  Total frames in video: {total_frames}")
            print(f"  Target number of samples: {args.num_samples}")
            print(f"  Calculation: {total_frames} / {args.num_samples} = {total_frames/args.num_samples:.2f} -> frame_skip = {frame_skip}")
            print(f"  Expected samples: ~{expected_samples} frames")
        else:
            frame_skip = 1
            print(f"Warning: Total frames is {total_frames}, cannot calculate frame skip. Processing all frames.")
            if args.num_samples <= 0:
                print(f"Warning: Invalid num_samples value: {args.num_samples}")
    else:
        frame_skip = 1
        print("Processing all frames (no frame skipping - --num-samples not specified)")
    
    # Determine output CSV path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = "data/high-bay/raw/" + f"{video_path.stem}.csv"
    
    # Prepare CSV data
    csv_data = []
    frame_count = 0
    processed_count = 0
    
    # Store detections for visualization: {frame_number: [(tag_id, corners, pose), ...]}
    detections_by_frame = {}
    # Store which frames were actually processed (for playback)
    processed_frames = []
    
    print(f"Processing video: {args.video}")
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {video_duration:.2f}s")
    print(f"Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")
    print(f"Output CSV: {output_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # Check if we've actually reached the end or if OpenCV stopped early
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos < total_frames - 1:
                print(f"\nWarning: OpenCV stopped reading at frame {int(current_pos)}/{total_frames}")
                print("This can happen due to:")
                print("  - Codec issues (especially with .webm files)")
                print("  - Corrupted frames in the video")
                print("  - OpenCV's video decoder limitations")
                print(f"Processed {processed_count} frames before stopping.")
            break
        
        # Get current frame number before any skipping
        current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        actual_frame_num = int(current_frame_pos) - 1  # 0-indexed frame number
        
        # Skip frames if sampling
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Track this frame as processed
        if args.show_visualization:
            processed_frames.append(actual_frame_num)
        
        # Get timestamp (in seconds) - use actual frame position for accuracy
        timestamp = current_frame_pos / fps if fps > 0 else frame_count * 0.033  # fallback to ~30fps
        
        # Detect tags
        detections = detector.detect(frame)
        
        # Store detections for visualization (only for frames we process)
        if args.show_visualization:
            detections_by_frame[actual_frame_num] = detections  # Store all detections (empty list if none)
        
        # Filter detections if a specific tag is requested
        if args.tag_id is not None:
            detections = [det for det in detections if det[0] == args.tag_id]
        
        # Append one row per detected tag. If none detected, write NaN row.
        if len(detections) == 0:
            csv_data.append({
                'timestamp': timestamp,
                'tag_id': np.nan,
                'qx': np.nan,
                'qy': np.nan,
                'qz': np.nan,
                'qw': np.nan,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
        else:
            # Convert pose translations from meters (solvePnP output) to millimeters
            # to match the OptiTrack dataset and downstream expectations.
            meter_to_mm = 1000.0
            for tag_id, pose, _ in detections:
                q, t = pose.to_quaternion_translation()
                # q is [w, x, y, z], but we need [qx, qy, qz, qw]
                qx, qy, qz, qw = q[1], q[2], q[3], q[0]
                x, y, z = t[0] * meter_to_mm, t[1] * meter_to_mm, t[2] * meter_to_mm
                
                csv_data.append({
                    'timestamp': timestamp,
                    'tag_id': tag_id,
                    'qx': qx,
                    'qy': qy,
                    'qz': qz,
                    'qw': qw,
                    'x': x,
                    'y': y,
                    'z': z
                })
        
        frame_count += 1
        processed_count += 1
        if processed_count % 100 == 0:
            current_time = timestamp
            print(f"Processed {processed_count} frames (time: {current_time:.2f}s / {video_duration:.2f}s)...")
    
    cap.release()
    
    # Write CSV file
    print(f"\nProcessed {processed_count} frames total")
    print(f"Writing {len(csv_data)} rows to CSV...")
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'tag_id', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Done! Output saved to: {output_path}")
    
    # Visualization playback
    if args.show_visualization:
        print("\nStarting visualization playback...")
        print(f"Playing back {len(processed_frames)} sampled frames")
        print("Controls:")
        print("  'q' - quit")
        print("  SPACE - pause/resume")
        print("  LEFT ARROW or 'a' - previous frame")
        print("  RIGHT ARROW or 'd' - next frame")
        print("  HOME or 'h' - first frame")
        print("  END or 'e' - last frame")
        
        # Reopen video for playback
        cap_viz = cv2.VideoCapture(str(video_path))
        if not cap_viz.isOpened():
            print("Warning: Could not reopen video for visualization")
        else:
            paused = True  # Start paused so user can navigate
            # Use a reasonable delay for sampled frames (e.g., 100ms per frame)
            frame_delay = 100  # milliseconds per frame
            frame_index = 0  # Index into processed_frames list
            last_frame_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000  # For auto-advance timing
            
            def display_frame(idx):
                """Display frame at given index."""
                if idx < 0 or idx >= len(processed_frames):
                    return None
                
                target_frame = processed_frames[idx]
                cap_viz.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap_viz.read()
                
                if not ret:
                    return None
                
                current_frame = target_frame
                current_time = current_frame / fps if fps > 0 else current_frame * 0.033
                
                # Get detections for this frame
                detections = detections_by_frame.get(current_frame, [])
                
                # Draw detections if any
                if len(detections) > 0:
                    for tag_id, pose, corners in detections:
                        # Draw green rectangle around tag
                        corners_int = corners.astype(int)
                        cv2.polylines(frame, [corners_int], True, (0, 255, 0), 2)
                        
                        # Draw tag ID
                        center = corners_int.mean(axis=0).astype(int)
                        cv2.putText(frame, f"ID:{tag_id}", tuple(center),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Draw distance
                        distance = np.linalg.norm(pose.t)
                        cv2.putText(frame, f"{distance:.2f}m", 
                                   (center[0], center[1] + 25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw timestamp and frame info
                cv2.putText(frame, f"Time: {current_time:.2f}s", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {current_frame}/{total_frames}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sample: {idx + 1}/{len(processed_frames)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show detection status
                if len(detections) > 0:
                    cv2.putText(frame, "TAG DETECTED", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No tag", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # Show pause status
                if paused:
                    cv2.putText(frame, "PAUSED", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "PLAYING", (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('AprilTag Detection Visualization', frame)
                return frame
            
            # Display first frame
            display_frame(frame_index)
            
            while True:
                current_time_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000
                
                # Auto-advance if playing
                if not paused:
                    if current_time_ms - last_frame_time >= frame_delay:
                        frame_index += 1
                        if frame_index >= len(processed_frames):
                            # Loop back to beginning
                            frame_index = 0
                        display_frame(frame_index)
                        last_frame_time = current_time_ms
                
                # Handle keyboard input
                wait_time = 10 if paused else max(1, int(frame_delay - (current_time_ms - last_frame_time)))
                key_code = cv2.waitKey(wait_time)
                key = key_code & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    if not paused:
                        last_frame_time = current_time_ms
                elif key == ord('a') or key_code == 81 or key_code == 65361:  # 'a' or left arrow
                    # Previous frame
                    paused = True
                    frame_index = max(0, frame_index - 1)
                    display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('d') or key_code == 83 or key_code == 65363:  # 'd' or right arrow
                    # Next frame
                    paused = True
                    frame_index = min(len(processed_frames) - 1, frame_index + 1)
                    display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('h') or key_code == 80 or key_code == 65360:  # 'h' or Home key
                    # First frame
                    paused = True
                    frame_index = 0
                    display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('e') or key_code == 87 or key_code == 65367:  # 'e' or End key
                    # Last frame
                    paused = True
                    frame_index = len(processed_frames) - 1
                    display_frame(frame_index)
                    last_frame_time = current_time_ms
            
            cap_viz.release()
            cv2.destroyAllWindows()
            print("Visualization closed.")
