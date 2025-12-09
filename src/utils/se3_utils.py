"""
SE(3) utilities for rigid body transformations.

Convention:
- SE(3) represented as 4x4 homogeneous matrix:
  T = [R  t]
      [0  1]
- T transforms points from frame B to frame A: p_A = T_AB @ p_B
"""

import numpy as np
from typing import Tuple, Optional

from .rotation_utils import (
    random_rotation,
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
    rotation_error_deg,
    project_to_SO3,
    is_valid_rotation
)


class SE3:
    """
    Rigid body transformation in SE(3).
    
    Represents a pose as rotation R ∈ SO(3) and translation t ∈ ℝ³.
    
    Attributes:
        R: 3x3 rotation matrix
        t: 3-element translation vector
    """
    
    def __init__(
        self, 
        R: Optional[np.ndarray] = None, 
        t: Optional[np.ndarray] = None, 
        T: Optional[np.ndarray] = None
    ):
        """
        Initialize SE(3) transformation.
        
        Args:
            R: 3x3 rotation matrix (default: identity)
            t: 3-element translation vector (default: zero)
            T: 4x4 homogeneous transformation matrix (overrides R, t)
        """
        if T is not None:
            self.R = T[:3, :3].copy()
            self.t = T[:3, 3].copy()
        else:
            self.R = R.copy() if R is not None else np.eye(3)
            self.t = t.copy().flatten() if t is not None else np.zeros(3)
    
    def matrix(self) -> np.ndarray:
        """
        Return 4x4 homogeneous transformation matrix.
        
        Returns:
            T = [R  t]
                [0  1]
        """
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T
    
    def inverse(self) -> 'SE3':
        """
        Return inverse transformation.
        
        T^{-1} = [R^T  -R^T @ t]
                 [0        1   ]
        
        Returns:
            SE3 inverse
        """
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return SE3(R_inv, t_inv)
    
    def __matmul__(self, other: 'SE3') -> 'SE3':
        """
        Compose transformations: T_AC = T_AB @ T_BC
        
        Args:
            other: SE3 transformation to compose
            
        Returns:
            Composed SE3 transformation
        """
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return SE3(R_new, t_new)
    
    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """
        Transform a 3D point.
        
        p' = R @ p + t
        
        Args:
            p: 3-element point
            
        Returns:
            Transformed 3-element point
        """
        p = np.asarray(p).flatten()
        return self.R @ p + self.t
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transform multiple 3D points.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Nx3 array of transformed points
        """
        return (self.R @ points.T).T + self.t
    
    @staticmethod
    def random(trans_scale: float = 1.0) -> 'SE3':
        """
        Generate random SE(3) transformation.
        
        Args:
            trans_scale: Standard deviation of translation components
            
        Returns:
            Random SE3 transformation
        """
        R = random_rotation()
        t = np.random.randn(3) * trans_scale
        return SE3(R, t)
    
    @staticmethod
    def identity() -> 'SE3':
        """Return identity transformation."""
        return SE3()
    
    @staticmethod
    def from_Rt(R: np.ndarray, t: np.ndarray) -> 'SE3':
        """Construct from rotation matrix and translation vector."""
        return SE3(R, t)
    
    @staticmethod
    def from_matrix(T: np.ndarray) -> 'SE3':
        """Construct from 4x4 homogeneous matrix."""
        return SE3(T=T)
    
    @staticmethod
    def from_quaternion_translation(q: np.ndarray, t: np.ndarray) -> 'SE3':
        """
        Construct from quaternion [w,x,y,z] and translation.
        
        Args:
            q: Quaternion [w, x, y, z]
            t: Translation vector
            
        Returns:
            SE3 transformation
        """
        R = quaternion_to_rotation_matrix(q)
        return SE3(R, t)
    
    def to_quaternion_translation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to quaternion and translation.
        
        Returns:
            q: Quaternion [w, x, y, z]
            t: Translation vector
        """
        q = rotation_matrix_to_quaternion(self.R)
        return q, self.t.copy()
    
    def copy(self) -> 'SE3':
        """Return a deep copy."""
        return SE3(self.R.copy(), self.t.copy())
    
    def __repr__(self) -> str:
        return f"SE3(R=\n{self.R},\nt={self.t})"
    
    def __str__(self) -> str:
        q, t = self.to_quaternion_translation()
        return f"SE3(t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}], q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}])"


def SE3_error(T1: SE3, T2: SE3) -> Tuple[float, float]:
    """
    Compute error between two SE(3) transformations.
    
    Args:
        T1, T2: SE3 transformations
        
    Returns:
        trans_error: Translation error in meters (Euclidean distance)
        rot_error: Rotation error in degrees
    """
    trans_error = np.linalg.norm(T1.t - T2.t)
    rot_error = rotation_error_deg(T1.R, T2.R)
    return trans_error, rot_error


def interpolate_SE3(T1: SE3, T2: SE3, alpha: float) -> SE3:
    """
    Linear interpolation between two SE(3) transformations.
    
    Uses SLERP for rotation, linear interpolation for translation.
    
    Args:
        T1, T2: SE3 transformations (endpoints)
        alpha: Interpolation parameter [0, 1]
               alpha=0 returns T1, alpha=1 returns T2
               
    Returns:
        Interpolated SE3 transformation
    """
    # Linear interpolation for translation
    t_interp = (1 - alpha) * T1.t + alpha * T2.t
    
    # SLERP for rotation via axis-angle
    from .rotation_utils import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
    
    R_diff = T1.R.T @ T2.R
    axis, angle = rotation_matrix_to_axis_angle(R_diff)
    
    if angle < 1e-10:
        R_interp = T1.R.copy()
    else:
        R_delta = axis_angle_to_rotation_matrix(axis, alpha * angle)
        R_interp = T1.R @ R_delta
    
    return SE3(R_interp, t_interp)


def add_SE3_noise(T: SE3, trans_std: float, rot_std: float) -> SE3:
    """
    Add Gaussian noise to SE(3) transformation.
    
    Args:
        T: SE3 transformation
        trans_std: Standard deviation of translation noise (meters)
        rot_std: Standard deviation of rotation noise (radians)
        
    Returns:
        Noisy SE3 transformation
    """
    from .rotation_utils import axis_angle_to_rotation_matrix
    
    # Translation noise
    t_noisy = T.t + np.random.normal(0, trans_std, 3)
    
    # Rotation noise (axis-angle perturbation)
    if rot_std > 0:
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        angle = np.random.normal(0, rot_std)
        R_noise = axis_angle_to_rotation_matrix(axis, angle)
        R_noisy = R_noise @ T.R
    else:
        R_noisy = T.R.copy()
    
    return SE3(R_noisy, t_noisy)


def is_valid_SE3(T: SE3, tol: float = 1e-6) -> bool:
    """
    Check if transformation is a valid SE(3) element.
    
    Args:
        T: SE3 to check
        tol: Tolerance for rotation validity
        
    Returns:
        True if valid
    """
    return is_valid_rotation(T.R, tol)
