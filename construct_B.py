"""
AprilTag detection and pose estimation 

Detects AprilTags in images and estimates their 6-DoF pose
relative to the camera frame (B_i in the RWHEC formulation).
Correctly handles both ground (horizontal) and wall (vertical) tags.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
import yaml
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# Add parent directory to path for SE3 import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.utils.se3_utils import SE3


class AprilTagPoseEstimator:
    """
    Robust AprilTag pose estimator that handles both ground and wall tags.
    
    Key improvements:
    1. Multiple 3D models for different tag orientations
    2. Automatic orientation detection
    3. Comprehensive validation and debugging
    4. Consistent coordinate system for all tags
    """
    
    # Camera coordinate system (OpenCV convention):
    #   X: right, Y: down, Z: forward (into scene)
    #
    # Tag coordinate system (standard for ALL tags):
    #   Origin: center of tag
    #   X: right along tag
    #   Y: up along tag  
    #   Z: out of tag (normal to tag surface)
    
    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
        tag_size: float,
        family: str,
        verbose: bool = False
    ):
        """
        Initialize the pose estimator.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            tag_size: Physical tag size in meters
            family: AprilTag family ('tag36h11', etc.)
            verbose: Print debug information
        """
        self.K = camera_matrix.astype(np.float32)
        self.dist_coeffs = dist_coeffs.astype(np.float32)
        self.tag_size = tag_size
        self.family = family
        self.verbose = verbose
        
        # Try to import apriltag library
        try:
            import apriltag
            self.detector = apriltag.Detector(
                apriltag.DetectorOptions(families=family)
            )
            self._use_apriltag = True
            if verbose:
                print("Using apriltag library for detection")
        except ImportError:
            # Fall back to OpenCV ArUco
            print("Warning: apriltag library not found, using OpenCV ArUco")
            self._use_apriltag = False
            self._setup_aruco()
        
        # Precompute 3D model points for different tag orientations
        self._setup_3d_models()
        
    def _setup_aruco(self):
        """Set up OpenCV ArUco detector as fallback."""
        aruco_dict_map = {
            'tag36h11': cv2.aruco.DICT_APRILTAG_36h11,
            'tag25h9': cv2.aruco.DICT_APRILTAG_25h9,
            'tag16h5': cv2.aruco.DICT_APRILTAG_16h5,
        }
        
        dict_id = aruco_dict_map.get(self.family, cv2.aruco.DICT_APRILTAG_36h11)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def _setup_3d_models(self):
        """Precompute 3D model points for different tag orientations."""
        s = self.tag_size / 2.0
        
        # We define 4 possible orientations:
        # 1. Ground (horizontal): tag in X-Y plane, Z=0, Z points up
        # 2. Wall front: tag in Y-Z plane, X=0, X points out of wall (toward camera)
        # 3. Wall right: tag in X-Z plane, Y=0, Y points out of wall (to the right)
        # 4. Wall left: tag in X-Z plane, Y=0, Y points out of wall (to the left)
        
        self.models = {
            'ground': np.array([
                [-s, -s, 0],  # bottom-left
                [ s, -s, 0],  # bottom-right
                [ s,  s, 0],  # top-right
                [-s,  s, 0],  # top-left
            ], dtype=np.float32),
            
            'wall_front': np.array([
                [0, -s, -s],  # bottom-left (X=0, Y=-s, Z=-s)
                [0, -s,  s],  # bottom-right
                [0,  s,  s],  # top-right
                [0,  s, -s],  # top-left
            ], dtype=np.float32),
            
            'wall_right': np.array([
                [-s, 0, -s],  # bottom-left
                [ s, 0, -s],  # bottom-right
                [ s, 0,  s],  # top-right
                [-s, 0,  s],  # top-left
            ], dtype=np.float32),
            
            'wall_left': np.array([
                [-s, 0,  s],  # bottom-left
                [ s, 0,  s],  # bottom-right
                [ s, 0, -s],  # top-right
                [-s, 0, -s],  # top-left
            ], dtype=np.float32),
        }
        
        # Transformation matrices to convert from each model's coordinate system
        # to our standard tag frame (X-right, Y-up, Z-out)
        self.model_transforms = {
            'ground': np.eye(4, dtype=np.float32),  # Already in standard frame
            
            'wall_front': np.array([
                [0, 0, 1, 0],   # New X = old Z (out of wall)
                [0, 1, 0, 0],   # New Y = old Y (up)
                [-1, 0, 0, 0],  # New Z = -old X (toward wall)
                [0, 0, 0, 1]
            ], dtype=np.float32),
            
            'wall_right': np.array([
                [1, 0, 0, 0],   # New X = old X (right)
                [0, 0, 1, 0],   # New Y = old Z (up)
                [0, -1, 0, 0],  # New Z = -old Y (out of wall)
                [0, 0, 0, 1]
            ], dtype=np.float32),
            
            'wall_left': np.array([
                [1, 0, 0, 0],   # New X = old X (right)
                [0, 0, -1, 0],  # New Y = -old Z (up)
                [0, 1, 0, 0],   # New Z = old Y (out of wall)
                [0, 0, 0, 1]
            ], dtype=np.float32),
        }
    
    def detect(self, image: np.ndarray, frame_num: int = None, timestamp: float = None) -> List[Tuple[int, SE3, np.ndarray, str]]:
        """
        Detect AprilTags and estimate their poses.
        
        Returns:
            List of (tag_id, pose, corners, orientation) tuples
            orientation: 'ground', 'wall_front', 'wall_right', or 'wall_left'
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect tags
        if self._use_apriltag:
            detections = self._detect_apriltag(gray)
        else:
            detections = self._detect_aruco(gray)
        
        results = []
        for tag_id, corners in detections:
            pose, orientation = self._estimate_pose_with_orientation(corners, tag_id)
            if pose is not None:
                # FIX: Ensure tag is in front of camera
                pose = self._ensure_pose_in_front_of_camera(pose, tag_id)
                if pose is not None:
                    results.append((tag_id, pose, corners, orientation))
        
        # Print debug info
        if self.verbose and results:
            print(f"\nFrame {frame_num} (t={timestamp:.2f}s) - Found {len(results)} tags:")
            for tag_id, pose, corners, orientation in results:
                self._print_debug_info(tag_id, pose, corners, orientation)
        
        return results
    
    def _detect_apriltag(self, gray: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Detect using apriltag library."""
        import apriltag
        
        detections = self.detector.detect(gray)
        results = []
        
        for det in detections:
            # Get corner points (should already be in consistent order)
            corners = det.corners.astype(np.float32)
            results.append((det.tag_id, corners))
        
        return results
    
    def _detect_aruco(self, gray: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        """Detect using OpenCV ArUco."""
        corners_list, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        results = []
        
        if ids is not None:
            for i, tag_id in enumerate(ids.flatten()):
                corners = corners_list[i].reshape(4, 2).astype(np.float32)
                
                # Aruco corners might be in different order, ensure consistency
                corners = self._reorder_corners_clockwise(corners)
                results.append((int(tag_id), corners))
        
        return results
    
    def _reorder_corners_clockwise(self, corners: np.ndarray) -> np.ndarray:
        """Reorder corners to be clockwise starting from top-left."""
        # Compute center
        center = corners.mean(axis=0)
        
        # Compute angles from center
        angles = []
        for corner in corners:
            dx = corner[0] - center[0]
            dy = corner[1] - center[1]
            angles.append(np.arctan2(dy, dx))
        
        # Sort by angle (clockwise)
        sorted_indices = np.argsort(angles)
        
        # Reorder to start from smallest angle (top-left quadrant)
        # Find index with angle closest to -135 degrees (-3π/4)
        target_angle = -3 * np.pi / 4
        angle_diffs = [abs(angles[i] - target_angle) for i in sorted_indices]
        start_idx = np.argmin(angle_diffs)
        
        reordered = np.roll(corners[sorted_indices], -start_idx, axis=0)
        return reordered
    
    def _ensure_pose_in_front_of_camera(self, pose: SE3, tag_id: int = None) -> Optional[SE3]:
        """
        CRITICAL FIX: Ensure the tag is in front of the camera (positive Z).
        Fixes the issue where tags appear behind the camera in the CSV data.
        """
        if pose is None:
            return None
        
        t = pose.t
        R = pose.R
        
        # Tag must be in front of camera (Z > 0 in OpenCV convention)
        if t[2] < 0:
            if self.verbose:
                print(f"  Tag {tag_id}: Flipping Z from {t[2]:.3f}m to positive (tag was behind camera)")
            
            # Flip the Z coordinate to positive
            t_new = t.copy()
            t_new[2] = abs(t[2])
            
            # Also flip the Z column in the rotation matrix to keep tag facing camera
            R_new = R.copy()
            R_new[:, 2] = -R_new[:, 2]  # Flip Z axis direction
            
            # Create new pose
            pose = SE3(R_new, t_new)
        
        # Additional check: tag should not be too close
        if t[2] < 0.1:  # Less than 10cm
            if self.verbose:
                print(f"  Tag {tag_id}: Too close to camera ({t[2]:.3f}m), rejecting")
            return None
        
        return pose
    
    def _estimate_pose_with_orientation(self, corners: np.ndarray, tag_id: int = None) -> Tuple[Optional[SE3], str]:
        """
        Estimate pose by trying all possible orientations and selecting the best.
        
        Returns:
            Tuple of (pose, orientation_type) or (None, None) if no valid pose found
        """
        best_pose = None
        best_orientation = None
        best_error = float('inf')
        best_distance = None
        
        # Try each orientation model
        for orientation, obj_points in self.models.items():
            # Try multiple PnP methods
            pose, error, distance = self._solve_pnp_with_fallbacks(corners, obj_points, tag_id)
            
            if pose is None:
                continue
            
            # Validate pose
            is_valid, validation_error = self._validate_pose(pose, corners, obj_points, orientation)
            
            if not is_valid:
                continue
            
            # Combined error metric
            total_error = error + validation_error
            
            # Prefer orientations that give more plausible distances
            # (indoor tags should typically be 0.5m to 5m away)
            distance_penalty = 0
            if distance < 0.3:
                distance_penalty = 5.0  # Penalize too close
            elif distance > 8.0:
                distance_penalty = 3.0  # Penalize too far
            
            score = total_error + distance_penalty
            
            if score < best_error:
                best_error = score
                best_pose = pose
                best_orientation = orientation
                best_distance = distance
        
        # Apply coordinate system transformation if needed
        if best_pose is not None and best_orientation != 'ground':
            # Transform from model-specific frame to standard tag frame
            transform = self.model_transforms[best_orientation]
            R_transform = transform[:3, :3]
            t_transform = transform[:3, 3]
            
            # Apply transformation: T_standard = T_model @ T_transform
            R_new = best_pose.R @ R_transform
            t_new = best_pose.R @ t_transform + best_pose.t
            
            best_pose = SE3(R_new, t_new)
        
        if self.verbose and tag_id is not None:
            if best_pose is not None:
                print(f"  Tag {tag_id}: Selected {best_orientation} orientation (error: {best_error:.3f})")
                print(f"    Distance: {best_distance:.3f}m")
                # Check if tag would be behind camera
                if best_pose.t[2] < 0:
                    print(f"    WARNING: Tag would be behind camera! (Z={best_pose.t[2]:.3f}m)")
            else:
                print(f"  Tag {tag_id}: No valid pose found")
        
        return best_pose, best_orientation
    
    def _solve_pnp_with_fallbacks(self, corners: np.ndarray, obj_points: np.ndarray, tag_id: int = None) -> Tuple[Optional[SE3], float, float]:
        """Solve PnP with multiple fallback methods."""
        methods = [
            ('IPPE_SQUARE', cv2.SOLVEPNP_IPPE_SQUARE),
            ('ITERATIVE', cv2.SOLVEPNP_ITERATIVE),
            ('EPNP', cv2.SOLVEPNP_EPNP),
            ('SQPNP', cv2.SOLVEPNP_SQPNP),
        ]
        
        best_pose = None
        best_error = float('inf')
        best_distance = None
        
        for method_name, method_flag in methods:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    corners,
                    self.K,
                    self.dist_coeffs,
                    flags=method_flag
                )
                
                if not success:
                    continue
                
                # Optional refinement
                try:
                    rvec, tvec = cv2.solvePnPRefineLM(
                        obj_points,
                        corners,
                        self.K,
                        self.dist_coeffs,
                        rvec,
                        tvec
                    )
                except:
                    pass  # Skip refinement if it fails
                
                # Convert to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                pose = SE3(R, tvec.flatten())
                
                # Calculate reprojection error
                projected, _ = cv2.projectPoints(obj_points, rvec, tvec, self.K, self.dist_coeffs)
                error = np.mean(np.linalg.norm(corners - projected.reshape(4, 2), axis=1))
                
                # Calculate distance
                distance = np.linalg.norm(pose.t)
                
                if error < best_error:
                    best_error = error
                    best_pose = pose
                    best_distance = distance
                    
                    if self.verbose and tag_id is not None and error < 10.0:  # Only print if error is reasonable
                        print(f"    {method_name}: error={error:.3f}px, distance={distance:.3f}m")
                        if pose.t[2] < 0:
                            print(f"      WARNING: Negative Z detected! (Z={pose.t[2]:.3f}m)")
                        
            except Exception as e:
                if self.verbose and tag_id is not None:
                    print(f"    {method_name} failed: {e}")
                continue
        
        return best_pose, best_error, best_distance
    
    def _validate_pose(self, pose: SE3, corners: np.ndarray, obj_points: np.ndarray, orientation: str) -> Tuple[bool, float]:
        """Validate if the estimated pose is physically plausible."""
        validation_error = 0.0
        
        # 1. Check reprojection error (already done, but include in validation)
        rvec, _ = cv2.Rodrigues(pose.R)
        tvec = pose.t.reshape(3, 1)
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, self.K, self.dist_coeffs)
        reproj_error = np.mean(np.linalg.norm(corners - projected.reshape(4, 2), axis=1))
        
        if reproj_error > 10.0:  # More than 10 pixels error is suspicious
            return False, 100.0
        
        validation_error += reproj_error
        
        # 2. Check distance (indoor tags should be reasonable)
        distance = np.linalg.norm(pose.t)
        if distance < 0.1 or distance > 15.0:
            return False, 100.0
        
        # 3. Check tag orientation vs expected orientation
        # Get tag normal (Z-axis in tag frame)
        tag_normal = pose.R @ np.array([0, 0, 1])
        
        if orientation == 'ground':
            # Ground tag: normal should be mostly vertical (large Y component in camera frame)
            # In camera frame: Y points down, so ground tag normal should have |Y| > 0.7
            if abs(tag_normal[1]) < 0.5:
                validation_error += 5.0  # Penalize but don't reject
        else:
            # Wall tag: normal should be mostly horizontal (small Y component)
            if abs(tag_normal[1]) > 0.7:
                validation_error += 5.0  # Penalize but don't reject
        
        # 4. Check if tag is facing camera (Z component of normal should be negative in camera frame)
        # Tag Z points out of tag. If tag_normal.z < 0, tag is facing away from camera
        # This is not necessarily wrong (tags can be at an angle), but extreme values are suspicious
        if tag_normal[2] < -0.95:  # Almost directly facing away
            validation_error += 3.0
        
        return True, validation_error
    
    def _print_debug_info(self, tag_id: int, pose: SE3, corners: np.ndarray, orientation: str):
        """Print detailed debug information about a tag."""
        t = pose.t
        R = pose.R
        distance = np.linalg.norm(t)
        
        # Calculate tag normal in camera frame
        tag_normal = R @ np.array([0, 0, 1])
        
        # Calculate reprojection error
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.reshape(3, 1)
        obj_points = self.models['ground']  # Use ground model for reprojection (after transformation)
        projected, _ = cv2.projectPoints(obj_points, rvec, tvec, self.K, self.dist_coeffs)
        reproj_error = np.mean(np.linalg.norm(corners - projected.reshape(4, 2), axis=1))
        
        # Calculate apparent size distance estimate
        focal_length = self.K[0, 0]
        tag_width_px = np.linalg.norm(corners[1] - corners[0])
        expected_distance = (focal_length * self.tag_size) / tag_width_px
        
        print(f"  Tag {tag_id} ({orientation}):")
        print(f"    Distance: {distance:.3f}m (expected: {expected_distance:.3f}m, ratio: {distance/expected_distance:.3f})")
        print(f"    Translation: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        print(f"    Normal vector: [{tag_normal[0]:.3f}, {tag_normal[1]:.3f}, {tag_normal[2]:.3f}]")
        print(f"    Reprojection error: {reproj_error:.2f} pixels")
        
        # Check for potential issues
        if abs(distance/expected_distance - 1.0) > 0.5:
            print(f"    WARNING: Distance mismatch > 50%!")
        if reproj_error > 5.0:
            print(f"    WARNING: High reprojection error!")
        if t[2] < 0:
            print(f"    WARNING: Tag is behind camera! (Z={t[2]:.3f}m)")
    
    def draw_detections(self, image: np.ndarray, detections: List[Tuple[int, SE3, np.ndarray, str]]) -> np.ndarray:
        """
        Draw tag detections and pose information on image.
        
        Args:
            image: Input image
            detections: List of (tag_id, pose, corners, orientation)
            
        Returns:
            Image with visualizations
        """
        output = image.copy()
        img_height, img_width = output.shape[:2]
        
        for tag_id, pose, corners, orientation in detections:
            try:
                # Draw tag boundary
                corners_int = corners.astype(int)
                cv2.polylines(output, [corners_int], True, (0, 255, 0), 2)
                
                # Draw tag ID and orientation
                cv2.putText(output, f"ID:{tag_id} ({orientation})", 
                           tuple(corners_int[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw center point
                center = corners_int.mean(axis=0).astype(int)
                cv2.circle(output, tuple(center), 4, (255, 0, 0), -1)
                
                # Get pose information
                t = pose.t
                distance = np.linalg.norm(t)
                
                # Draw coordinate axes (10cm length)
                axis_length = 0.1
                axis_points = np.array([
                    [0, 0, 0],
                    [axis_length, 0, 0],  # X axis (red)
                    [0, axis_length, 0],  # Y axis (green)
                    [0, 0, axis_length]   # Z axis (blue)
                ], dtype=np.float32)
                
                # Project axes
                R = pose.R
                rvec, _ = cv2.Rodrigues(R)
                tvec = t.reshape(3, 1)
                projected_axes, _ = cv2.projectPoints(axis_points, rvec, tvec, self.K, self.dist_coeffs)
                
                # Reshape and convert to integers
                projected_axes = projected_axes.reshape(4, 2).astype(int)
                origin = tuple(projected_axes[0])
                
                # Draw axes if origin is within image bounds
                if (0 <= origin[0] < img_width and 0 <= origin[1] < img_height):
                    # X axis (red)
                    if len(projected_axes) > 1:
                        x_end = tuple(projected_axes[1])
                        cv2.arrowedLine(output, origin, x_end, (0, 0, 255), 2)
                        cv2.putText(output, "X", x_end, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Y axis (green)
                    if len(projected_axes) > 2:
                        y_end = tuple(projected_axes[2])
                        cv2.arrowedLine(output, origin, y_end, (0, 255, 0), 2)
                        cv2.putText(output, "Y", y_end, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Z axis (blue) - points out of tag
                    if len(projected_axes) > 3:
                        z_end = tuple(projected_axes[3])
                        cv2.arrowedLine(output, origin, z_end, (255, 0, 0), 2)
                        cv2.putText(output, "Z", z_end, 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
                # Draw pose information
                info_lines = [
                    f"Dist: {distance:.2f}m",
                    f"t: [{t[0]:.2f},{t[1]:.2f},{t[2]:.2f}]",
                ]
                
                # Get quaternion for display
                try:
                    q, _ = pose.to_quaternion_translation()
                    info_lines.append(f"q: [{q[0]:.2f},{q[1]:.2f},{q[2]:.2f},{q[3]:.2f}]")
                except:
                    pass
                
                # Draw info lines
                text_y = center[1] + 30
                for i, line in enumerate(info_lines):
                    cv2.putText(output, line, 
                               (center[0], text_y + i * 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
                # Draw line from camera to tag (in image coordinates)
                # Camera center projects to principal point
                camera_center = (int(self.K[0, 2]), int(self.K[1, 2]))
                cv2.line(output, camera_center, tuple(center), (255, 255, 255), 1, cv2.LINE_AA)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error drawing tag {tag_id}: {e}")
        
        return output


class AprilTag3DVisualizer:
    """3D visualization of camera and tag positions."""
    
    def __init__(self):
        """Initialize the 3D visualizer."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels
        self.ax.set_xlabel('X (right, m)')
        self.ax.set_ylabel('Y (down, m)')
        self.ax.set_zlabel('Z (forward, m)')
        self.ax.set_title('Camera and Tag Positions')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set view angle for better visualization
        self.ax.view_init(elev=20, azim=-60)
        
        # Initialize collections for animation
        self.scatters = {}
        self.lines = {}
        self.quivers = {}
        
        # Store all tag positions for history
        self.all_tag_positions = {}  # tag_id -> list of positions
        self.camera_positions = []   # List of camera positions (should be [0,0,0] always)
        
    def update_plot(self, detections: List[Tuple[int, SE3, np.ndarray, str]], frame_num: int, timestamp: float):
        """
        Update the 3D plot with new detections.
        
        Args:
            detections: List of (tag_id, pose, corners, orientation)
            frame_num: Current frame number
            timestamp: Current timestamp
        """
        # Clear previous frame's plot
        self.ax.clear()
        
        # Set labels and title
        self.ax.set_xlabel('X (right, m)')
        self.ax.set_ylabel('Y (down, m)')
        self.ax.set_zlabel('Z (forward, m)')
        self.ax.set_title(f'Camera and Tag Positions - Frame {frame_num} (t={timestamp:.2f}s)')
        
        # Plot camera at origin (0, 0, 0)
        camera_pos = np.array([0, 0, 0])
        self.ax.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
                       c='red', s=100, marker='o', label='Camera')
        
        # Draw camera coordinate system
        axis_length = 0.2  # 20cm
        # Camera coordinate system: X-right, Y-down, Z-forward
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', label='Camera X (right)', 
                      arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', label='Camera Y (down)', 
                      arrow_length_ratio=0.1, linewidth=2)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', label='Camera Z (forward)', 
                      arrow_length_ratio=0.1, linewidth=2)
        
        # Draw a line showing camera "front" (Z-axis direction)
        front_line_length = 0.5  # 50cm
        self.ax.plot([0, 0], [0, 0], [0, front_line_length], 
                    color='cyan', linewidth=3, linestyle='--', label='Camera Front')
        
        # Plot each detected tag
        colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color map for different tags
        
        for i, (tag_id, pose, corners, orientation) in enumerate(detections):
            # Get tag position
            tag_pos = pose.t
            
            # Store for history
            if tag_id not in self.all_tag_positions:
                self.all_tag_positions[tag_id] = []
            self.all_tag_positions[tag_id].append(tag_pos)
            
            # Plot current position
            color = colors[tag_id % len(colors)]
            self.ax.scatter([tag_pos[0]], [tag_pos[1]], [tag_pos[2]], 
                           c=[color], s=50, marker='^', label=f'Tag {tag_id} ({orientation})')
            
            # Draw line from camera to tag
            self.ax.plot([0, tag_pos[0]], [0, tag_pos[1]], [0, tag_pos[2]], 
                        color=color, alpha=0.5, linestyle=':')
            
            # Draw tag coordinate axes
            axis_length = 0.1  # 10cm
            R = pose.R
            # Tag axes in camera frame: R * axis_vector
            x_axis = R @ np.array([axis_length, 0, 0])
            y_axis = R @ np.array([0, axis_length, 0])
            z_axis = R @ np.array([0, 0, axis_length])
            
            self.ax.quiver(tag_pos[0], tag_pos[1], tag_pos[2], 
                          x_axis[0], x_axis[1], x_axis[2], 
                          color='red', alpha=0.5, arrow_length_ratio=0.1)
            self.ax.quiver(tag_pos[0], tag_pos[1], tag_pos[2], 
                          y_axis[0], y_axis[1], y_axis[2], 
                          color='green', alpha=0.5, arrow_length_ratio=0.1)
            self.ax.quiver(tag_pos[0], tag_pos[1], tag_pos[2], 
                          z_axis[0], z_axis[1], z_axis[2], 
                          color='blue', alpha=0.5, arrow_length_ratio=0.1)
            
            # Add text label near tag
            self.ax.text(tag_pos[0], tag_pos[1], tag_pos[2], 
                        f'{tag_id}', fontsize=8, color=color)
        
        # Plot tag position history (trajectories)
        for tag_id, positions in self.all_tag_positions.items():
            if len(positions) > 1:
                positions_array = np.array(positions)
                color = colors[tag_id % len(colors)]
                self.ax.plot(positions_array[:, 0], positions_array[:, 1], positions_array[:, 2],
                           color=color, alpha=0.3, linewidth=1, linestyle='--')
        
        # Set plot limits
        max_range = 3.0  # 3 meters
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([0, max_range])  # Z is forward, typically positive
        
        # Add grid and legend
        self.ax.grid(True)
        self.ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize='small')
        
        # Draw the plot
        plt.draw()
        plt.pause(0.001)  # Small pause to update the plot
    
    def save_figure(self, filename: str):
        """Save the current 3D plot to a file."""
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved 3D plot to {filename}")


# The rest of the file remains similar but uses the new class
def create_board_config(
    n_rows: int,
    n_cols: int,
    tag_size: float,
    tag_spacing: float,
    start_id: int = 0
) -> Dict[int, np.ndarray]:
    """Create board configuration for a regular grid of AprilTags."""
    config = {}
    
    # Board center
    center_x = (n_cols - 1) * tag_spacing / 2
    center_y = (n_rows - 1) * tag_spacing / 2
    
    tag_id = start_id
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * tag_spacing - center_x
            y = (n_rows - 1 - row) * tag_spacing - center_y
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
    parser.add_argument('--verbose', action='store_true', help='Print detailed pose information for debugging')
    args = parser.parse_args()
    
    # Load camera matrix
    if args.camera_matrix:
        calib_path = Path(args.camera_matrix)
        if calib_path.suffix.lower() in ['.yaml', '.yml']:
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
            # Load from npz file
            calib_data = np.load(args.camera_matrix)
            K = calib_data['K'].astype(np.float32)
            dist = calib_data['dist'].astype(np.float32)
    else:
        # Default camera matrix
        print("Warning: Using default camera matrix. For accurate results, provide --camera-matrix")
        K = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        dist = np.zeros(5)
    args.tag_size = float(0.15)  # hardcoded tag size for this context
    # Print camera calibration info
    print("\n=== CAMERA CALIBRATION INFO ===")
    print(f"Camera matrix K:\n{K}")
    print(f"\nDistortion coefficients: {dist}")
    print(f"Focal length: fx={K[0,0]:.2f}, fy={K[1,1]:.2f}")
    print(f"Principal point: cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")
    print(f"Tag size: {args.tag_size}m")
    print("==============================\n")
    
    # Initialize detector
    detector = AprilTagPoseEstimator(
        K,
        dist,
        tag_size=args.tag_size,
        family=args.tag_family,
        verbose=args.verbose
    )
    
    # ALWAYS initialize 3D visualizer
    visualizer_3d = AprilTag3DVisualizer()
    print("3D visualization enabled (always on)")
    
    # Open video file
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {args.video}")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Determine frame skipping
    if args.num_samples is not None and total_frames > 0 and args.num_samples > 0:
        frame_skip = max(1, int(total_frames / args.num_samples))
        expected_samples = total_frames // frame_skip
        print(f"Frame skipping: processing 1 frame every {frame_skip} frames")
        print(f"  Total frames in video: {total_frames}")
        print(f"  Target number of samples: {args.num_samples}")
        print(f"  Expected samples: ~{expected_samples} frames")
    else:
        frame_skip = 1
        print("Processing all frames")
    
    # Determine output CSV path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("data/high-bay/raw") / f"{video_path.stem}.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV data
    csv_data = []
    frame_count = 0
    processed_count = 0
    
    # Store detections for visualization
    detections_by_frame = {}
    processed_frames = []
    
    print(f"\n=== PROCESSING VIDEO ===")
    print(f"Video: {args.video}")
    print(f"Resolution: {width}x{height}")
    print(f"Video FPS: {fps:.2f}, Total frames: {total_frames}, Duration: {video_duration:.2f}s")
    print(f"Frame skip: {frame_skip} (processing every {frame_skip} frame(s))")
    print(f"Output CSV: {output_path}")
    print("3D visualization: Always on")
    print("=======================\n")
    
    # Process video
    while True:
        ret, frame = cap.read()
        if not ret:
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos < total_frames - 1:
                print(f"\nWarning: OpenCV stopped reading at frame {int(current_pos)}/{total_frames}")
            break
        
        # Get current frame number
        current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        actual_frame_num = int(current_frame_pos) - 1
        
        # Skip frames if sampling
        if frame_count % frame_skip != 0:
            frame_count += 1
            continue
        
        # Track this frame as processed
        if args.show_visualization:
            processed_frames.append(actual_frame_num)
        
        # Get timestamp
        timestamp = current_frame_pos / fps if fps > 0 else frame_count * 0.033
        
        # Detect tags
        detections = detector.detect(frame, frame_num=actual_frame_num, timestamp=timestamp)
        
        # ALWAYS update 3D plot
        visualizer_3d.update_plot(detections, actual_frame_num, timestamp)
        
        # Draw visualizations for video playback
        if args.show_visualization:
            vis_frame = detector.draw_detections(frame, detections)
            detections_by_frame[actual_frame_num] = (detections, vis_frame)
        else:
            detections_by_frame[actual_frame_num] = (detections, None)
        
        # Filter detections if a specific tag is requested
        if args.tag_id is not None:
            detections = [det for det in detections if det[0] == args.tag_id]
        
        # Append to CSV data
        if len(detections) == 0:
            csv_data.append({
                'timestamp': timestamp,
                'tag_id': np.nan,
                'orientation': np.nan,
                'qx': np.nan,
                'qy': np.nan,
                'qz': np.nan,
                'qw': np.nan,
                'x': np.nan,
                'y': np.nan,
                'z': np.nan
            })
        else:
            # Convert pose translations from meters to millimeters
            meter_to_mm = 1000.0
            for tag_id, pose, corners, orientation in detections:
                q, t = pose.to_quaternion_translation()
                # q is [w, x, y, z], but we need [qx, qy, qz, qw] for CSV
                qx, qy, qz, qw = q[1], q[2], q[3], q[0]
                x, y, z = t[0] * meter_to_mm, t[1] * meter_to_mm, t[2] * meter_to_mm
                
                csv_data.append({
                    'timestamp': timestamp,
                    'tag_id': tag_id,
                    'orientation': orientation,
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
        
        # Progress update
        if processed_count % 100 == 0 and not args.verbose:
            current_time = timestamp
            print(f"Processed {processed_count} frames (time: {current_time:.2f}s / {video_duration:.2f}s)...")
    
    cap.release()
    
    # Write CSV file
    print(f"\nProcessed {processed_count} frames total")
    print(f"Writing {len(csv_data)} rows to CSV...")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'tag_id', 'orientation', 'qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Done! Output saved to: {output_path}")
    
    # Save the final 3D plot as left_success3_cut_visualization.png
    save_path = output_path.parent / "left_success3_cut_visualization.png"
    visualizer_3d.save_figure(str(save_path))
    
    print(f"Saved 3D visualization to: {save_path}")
    
    # Visualization playback (if requested)
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
        print("  's' - save current frame as image")
        
        # Reopen video for playback
        cap_viz = cv2.VideoCapture(str(video_path))
        if not cap_viz.isOpened():
            print("Warning: Could not reopen video for visualization")
        else:
            paused = True
            frame_delay = 100  # milliseconds per frame
            frame_index = 0
            last_frame_time = cv2.getTickCount() / cv2.getTickFrequency() * 1000
            
            def display_frame(idx):
                if idx < 0 or idx >= len(processed_frames):
                    return None
                
                target_frame = processed_frames[idx]
                cap_viz.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                ret, frame = cap_viz.read()
                
                if not ret:
                    return None
                
                current_frame = target_frame
                current_time = current_frame / fps if fps > 0 else current_frame * 0.033
                
                # Get detections and visualization frame
                detections, vis_frame = detections_by_frame.get(current_frame, ([], None))
                
                if vis_frame is not None:
                    display_frame_img = vis_frame.copy()
                else:
                    display_frame_img = frame.copy()
                    # Draw basic info
                    cv2.putText(display_frame_img, f"Frame: {current_frame}/{total_frames}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame_img, f"Time: {current_time:.2f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add navigation info
                cv2.putText(display_frame_img, f"Sample: {idx + 1}/{len(processed_frames)}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show detection status
                if len(detections) > 0:
                    cv2.putText(display_frame_img, f"Tags detected: {len(detections)}", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame_img, "No tags detected", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
                
                # Show pause status
                if paused:
                    cv2.putText(display_frame_img, "PAUSED", (10, display_frame_img.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame_img, "PLAYING", (10, display_frame_img.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add help text
                cv2.putText(display_frame_img, "Space: pause/play | Arrows: navigate | q: quit | s: save", 
                           (10, display_frame_img.shape[0] - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('AprilTag Detection Visualization', display_frame_img)
                return display_frame_img
            
            # Display first frame
            current_display = display_frame(frame_index)
            
            while True:
                current_time_ms = cv2.getTickCount() / cv2.getTickFrequency() * 1000
                
                # Auto-advance if playing
                if not paused:
                    if current_time_ms - last_frame_time >= frame_delay:
                        frame_index += 1
                        if frame_index >= len(processed_frames):
                            frame_index = 0
                        current_display = display_frame(frame_index)
                        last_frame_time = current_time_ms
                
                # Handle keyboard input
                wait_time = 10 if paused else max(1, int(frame_delay - (current_time_ms - last_frame_time)))
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    if not paused:
                        last_frame_time = current_time_ms
                elif key == ord('a'):
                    paused = True
                    frame_index = max(0, frame_index - 1)
                    current_display = display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('d'):
                    paused = True
                    frame_index = min(len(processed_frames) - 1, frame_index + 1)
                    current_display = display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('h'):
                    paused = True
                    frame_index = 0
                    current_display = display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('e'):
                    paused = True
                    frame_index = len(processed_frames) - 1
                    current_display = display_frame(frame_index)
                    last_frame_time = current_time_ms
                elif key == ord('s'):
                    if current_display is not None:
                        save_path = f"frame_{processed_frames[frame_index]:05d}.png"
                        cv2.imwrite(save_path, current_display)
                        print(f"Saved frame to {save_path}")
            
            cap_viz.release()
            cv2.destroyAllWindows()
            print("Visualization closed.")
    
    # Keep the 3D plot open until user closes it
    print("\n3D plot displayed. Close the window to exit...")
    plt.show()