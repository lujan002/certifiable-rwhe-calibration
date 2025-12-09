"""
Rotation utilities for SO(3) operations.

Mathematical conventions:
- Rotation matrices R ∈ SO(3) act on column vectors: v' = R @ v
- Quaternions use [w, x, y, z] convention (scalar first)
- Axis-angle: R = exp([ω*θ]_×) where ω is unit axis, θ is angle
"""

import numpy as np
from typing import Tuple


def skew(v: np.ndarray) -> np.ndarray:
    """
    Convert 3-vector to skew-symmetric matrix.
    
    [v]_× = [  0   -v_z   v_y ]
            [ v_z    0   -v_x ]
            [-v_y   v_x    0  ]
    
    Property: [v]_× @ u = v × u (cross product)
    
    Args:
        v: 3-element array
        
    Returns:
        3x3 skew-symmetric matrix
    """
    v = np.asarray(v).flatten()
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def vee(S: np.ndarray) -> np.ndarray:
    """
    Extract 3-vector from skew-symmetric matrix.
    
    Inverse of skew(): vee(skew(v)) = v
    
    Args:
        S: 3x3 skew-symmetric matrix
        
    Returns:
        3-element array
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def axis_angle_to_rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    R = I + sin(θ)[ω]_× + (1 - cos(θ))[ω]_×²
    
    Args:
        axis: 3-element unit vector (rotation axis ω)
        angle: Rotation angle θ in radians
        
    Returns:
        3x3 rotation matrix
    """
    axis = np.asarray(axis).flatten()
    axis = axis / np.linalg.norm(axis)  # Ensure unit vector
    
    K = skew(axis)
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def rotation_matrix_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Extract axis-angle representation from rotation matrix.
    
    Uses: θ = arccos((trace(R) - 1) / 2)
          ω = (1/(2*sin(θ))) * [R32-R23, R13-R31, R21-R12]^T
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        axis: Unit vector (3,) - rotation axis
        angle: Rotation angle in radians [0, π]
    """
    # Angle from trace
    trace_R = np.trace(R)
    angle = np.arccos(np.clip((trace_R - 1) / 2, -1, 1))
    
    # Handle edge cases
    if angle < 1e-10:
        # Near-identity: return arbitrary axis with zero angle
        return np.array([0.0, 0.0, 1.0]), 0.0
    
    if np.abs(angle - np.pi) < 1e-6:
        # Near 180°: axis from eigenvector corresponding to eigenvalue 1
        # R @ v = v for the rotation axis v
        eigenvalues, eigenvectors = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigenvalues - 1))
        axis = np.real(eigenvectors[:, idx])
        return axis / np.linalg.norm(axis), angle
    
    # General case: axis from skew-symmetric part
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]
    ])
    axis = axis / (2 * np.sin(angle))
    
    return axis / np.linalg.norm(axis), angle


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    
    Uses Shepperd's method for numerical stability.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z] with ||q|| = 1
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2 * np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2 * np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to rotation matrix.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    q = np.asarray(q).flatten()
    q = q / np.linalg.norm(q)  # Normalize
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def project_to_SO3(M: np.ndarray) -> np.ndarray:
    """
    Project arbitrary 3x3 matrix to nearest rotation matrix.
    
    Uses SVD: R = U @ V^T, with det correction for proper rotation.
    
    This is the solution to: argmin_R ||R - M||_F  s.t. R ∈ SO(3)
    
    Args:
        M: 3x3 matrix
        
    Returns:
        3x3 rotation matrix (nearest to M in Frobenius norm)
    """
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    # Ensure det(R) = +1 (proper rotation, not reflection)
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    return R


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """
    Compute rotation error in degrees between two rotation matrices.
    
    Error = arccos((trace(R1^T @ R2) - 1) / 2)
    
    Args:
        R1, R2: 3x3 rotation matrices
        
    Returns:
        Rotation error in degrees [0, 180]
    """
    R_diff = R1.T @ R2
    _, angle = rotation_matrix_to_axis_angle(R_diff)
    return np.rad2deg(angle)


def random_rotation() -> np.ndarray:
    """
    Generate a uniformly random rotation matrix.
    
    Uses QR decomposition of random Gaussian matrix.
    
    Returns:
        3x3 rotation matrix sampled uniformly from SO(3)
    """
    # Random matrix with Gaussian entries
    M = np.random.randn(3, 3)
    # QR decomposition gives orthogonal Q
    Q, R = np.linalg.qr(M)
    # Ensure proper rotation (det = +1)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def average_rotations(R_list: list) -> np.ndarray:
    """
    Compute average rotation from list of rotation matrices.
    
    Uses iterative algorithm on SO(3) manifold (Karcher mean).
    
    Args:
        R_list: List of 3x3 rotation matrices
        
    Returns:
        3x3 average rotation matrix
    """
    if len(R_list) == 0:
        return np.eye(3)
    
    R_avg = R_list[0].copy()
    
    for _ in range(10):  # Iterate until convergence
        # Compute mean in tangent space
        log_sum = np.zeros((3, 3))
        for R in R_list:
            R_diff = R_avg.T @ R
            axis, angle = rotation_matrix_to_axis_angle(R_diff)
            log_sum += angle * skew(axis)
        
        log_mean = log_sum / len(R_list)
        
        # Check convergence
        if np.linalg.norm(log_mean, 'fro') < 1e-10:
            break
        
        # Update average
        axis = vee(log_mean)
        angle = np.linalg.norm(axis)
        if angle > 1e-10:
            axis = axis / angle
            R_update = axis_angle_to_rotation_matrix(axis, angle)
            R_avg = R_avg @ R_update
    
    return R_avg


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if matrix is a valid rotation matrix.
    
    Checks: R^T @ R ≈ I and det(R) ≈ 1
    
    Args:
        R: Matrix to check
        tol: Tolerance for numerical comparison
        
    Returns:
        True if R ∈ SO(3)
    """
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality
    if not np.allclose(R.T @ R, np.eye(3), atol=tol):
        return False
    
    # Check determinant
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False
    
    return True
