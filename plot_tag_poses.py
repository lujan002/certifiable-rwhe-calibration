#!/usr/bin/env python3
"""
3D visualization of tag poses from CSV file using matplotlib.
Can optionally overlay OptiTrack trajectory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import sys
import os

def load_first_A_pose_from_optitrack(optitrack_csv_file):
    """Load the first A pose from OptiTrack CSV file.
    
    Args:
        optitrack_csv_file: Path to OptiTrack CSV file
        
    Returns:
        A_first: 4x4 transformation matrix of the first A pose, or None if not found
    """
    if not os.path.exists(optitrack_csv_file):
        print(f"Warning: OptiTrack file '{optitrack_csv_file}' not found. Cannot transform poses.")
        return None
    
    try:
        # OptiTrack CSV has 7 header rows, then:
        # Frame, Time, Quat_X, Quat_Y, Quat_Z, Quat_W, Pos_X, Pos_Y, Pos_Z (in mm)
        df = pd.read_csv(optitrack_csv_file, skiprows=7)
        
        if len(df) == 0:
            print(f"Warning: No data rows in '{optitrack_csv_file}'. Cannot transform poses.")
            return None
        
        # Extract first row data
        # Columns: Frame, Time, Quat_X(2), Quat_Y(3), Quat_Z(4), Quat_W(5), Pos_X(6), Pos_Y(7), Pos_Z(8)
        first_row = df.iloc[0]
        
        # Extract quaternion (qx, qy, qz, qw) - columns 2, 3, 4, 5
        qx = first_row.iloc[2]
        qy = first_row.iloc[3]
        qz = first_row.iloc[4]
        qw = first_row.iloc[5]
        
        # Extract position (in mm) - columns 6, 7, 8
        tx = first_row.iloc[6]
        ty = first_row.iloc[7]
        tz = first_row.iloc[8]
        
        # Convert quaternion to rotation matrix
        # scipy uses [x, y, z, w] format
        quat = [qx, qy, qz, qw]
        R = Rotation.from_quat(quat).as_matrix()
        
        # Construct 4x4 transformation matrix
        A_first = np.eye(4)
        A_first[:3, :3] = R
        A_first[:3, 3] = [tx, ty, tz]
        
        print(f"First A pose loaded from OptiTrack: translation = [{tx:.2f}, {ty:.2f}, {tz:.2f}] mm")
        return A_first
        
    except Exception as e:
        print(f"Warning: Error loading first A pose from OptiTrack: {e}. Cannot transform poses.")
        return None

def transform_poses_by_inv_A(translations, rotation_matrices, A_first):
    """Transform tag poses by the inverse of the first A pose.
    
    Args:
        translations: Nx3 array of tag positions
        rotation_matrices: Nx3x3 array of rotation matrices
        A_first: 4x4 transformation matrix of the first A pose
        
    Returns:
        transformed_translations: Nx3 array of transformed positions
        transformed_rotation_matrices: Nx3x3 array of transformed rotations
    """
    if A_first is None:
        return translations, rotation_matrices
    
    # Compute inverse of A_first
    A_first_inv = np.linalg.inv(A_first)
    
    n_tags = len(translations)
    transformed_translations = np.zeros_like(translations)
    transformed_rotation_matrices = np.zeros_like(rotation_matrices)
    
    for i in range(n_tags):
        # Construct 4x4 transformation matrix for this tag
        T_tag = np.eye(4)
        T_tag[:3, :3] = rotation_matrices[i]
        T_tag[:3, 3] = translations[i]
        
        # Transform: T_tag_transformed = A_first_inv @ T_tag
        T_tag_transformed = A_first_inv @ T_tag
        
        # Extract transformed translation and rotation
        transformed_translations[i] = T_tag_transformed[:3, 3]
        transformed_rotation_matrices[i] = T_tag_transformed[:3, :3]
    
    return transformed_translations, transformed_rotation_matrices

def load_poses_from_csv(csv_filename="tag_poses.csv"):
    """Load tag poses from CSV file."""
    if not os.path.exists(csv_filename):
        print(f"Error: File '{csv_filename}' not found.")
        sys.exit(1)
    
    df = pd.read_csv(csv_filename)
    
    tag_ids = df['tag_id'].values
    translations = df[['tx', 'ty', 'tz']].values
    rotations = df[['r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33']].values
    
    # Reshape rotations to 3x3 matrices
    rotation_matrices = rotations.reshape(-1, 3, 3)
    
    return tag_ids, translations, rotation_matrices

def load_optitrack_trajectory(optitrack_csv_file):
    """Load OptiTrack trajectory from CSV file.
    
    Expected format: OptiTrack CSV with 7 header rows, then:
        Frame, Time, Quat_X, Quat_Y, Quat_Z, Quat_W, Pos_X, Pos_Y, Pos_Z (in mm)
    
    Returns:
        positions: Nx3 array of positions in mm
    """
    if not os.path.exists(optitrack_csv_file):
        print(f"Warning: OptiTrack file '{optitrack_csv_file}' not found.")
        return None
    
    # Skip first 7 rows (header information)
    df = pd.read_csv(optitrack_csv_file, skiprows=7)
    
    # Extract position columns (columns 6, 7, 8 are X, Y, Z in mm)
    pos_x = df.iloc[:, 6].values  # Already in mm
    pos_y = df.iloc[:, 7].values
    pos_z = df.iloc[:, 8].values
    
    positions = np.column_stack([pos_x, pos_y, pos_z])
    
    print(f"Loaded {len(positions)} OptiTrack trajectory points")
    return positions

def plot_poses_interactive(tag_ids, translations, rotation_matrices, optitrack_positions=None, arrow_scale=0.2):
    """Create 3D plot of tag poses using matplotlib.
    
    Args:
        tag_ids: Array of tag IDs
        translations: Nx3 array of tag positions
        rotation_matrices: Nx3x3 array of rotation matrices
        optitrack_positions: Optional Nx3 array of OptiTrack trajectory positions
        arrow_scale: Scale factor for pose arrows
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    xs = translations[:, 0]
    ys = translations[:, 1]
    zs = translations[:, 2]
    
    # Extract z-axis directions from rotation matrices (third column)
    z_axes = rotation_matrices[:, :, 2]  # Shape: (n_tags, 3)
    
    # Calculate arrow endpoints
    arrow_xs = xs + arrow_scale * z_axes[:, 0]
    arrow_ys = ys + arrow_scale * z_axes[:, 1]
    arrow_zs = zs + arrow_scale * z_axes[:, 2]
    
    # Plot OptiTrack trajectory if provided
    if optitrack_positions is not None:
        ax.plot(optitrack_positions[:, 0], optitrack_positions[:, 1], optitrack_positions[:, 2],
                'b-', linewidth=2, alpha=0.7, label='OptiTrack Trajectory')
        # Mark start and end points
        ax.scatter(optitrack_positions[0, 0], optitrack_positions[0, 1], optitrack_positions[0, 2],
                  c='green', s=100, marker='o', label='Trajectory Start', zorder=5)
        ax.scatter(optitrack_positions[-1, 0], optitrack_positions[-1, 1], optitrack_positions[-1, 2],
                  c='orange', s=100, marker='s', label='Trajectory End', zorder=5)
    
    # Plot tag origins
    scatter = ax.scatter(xs, ys, zs, c=tag_ids, s=100, cmap='viridis', 
                         label='Tag Origins', edgecolors='black', linewidths=0.5)
    
    # Add text labels for each tag
    for i in range(len(tag_ids)):
        ax.text(xs[i], ys[i], zs[i], f"tag {int(tag_ids[i])}", 
                fontsize=8, ha='left', va='bottom')
    
    # Plot pose arrows (z-axis)
    for i in range(len(tag_ids)):
        ax.plot([xs[i], arrow_xs[i]], [ys[i], arrow_ys[i]], [zs[i], arrow_zs[i]], 
                'r-', linewidth=2, label='Z-axis' if i == 0 else '')
        # Arrowhead
        ax.scatter([arrow_xs[i]], [arrow_ys[i]], [arrow_zs[i]], 
                  c='red', s=50, marker='^')
    
    # Set labels and title
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    title = 'Tag Poses Visualization'
    if optitrack_positions is not None:
        title += ' with OptiTrack Trajectory'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Tag ID', fontsize=10)
    
    # Set equal aspect ratio - consider both tag poses and trajectory
    all_x = xs.copy()
    all_y = ys.copy()
    all_z = zs.copy()
    if optitrack_positions is not None:
        all_x = np.concatenate([all_x, optitrack_positions[:, 0]])
        all_y = np.concatenate([all_y, optitrack_positions[:, 1]])
        all_z = np.concatenate([all_z, optitrack_positions[:, 2]])
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set initial viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add legend (only show unique labels)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    csv_filename = "tag_poses.csv"
    optitrack_csv = None
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    if len(sys.argv) > 2:
        optitrack_csv = sys.argv[2]
    
    print(f"Loading poses from '{csv_filename}'...")
    tag_ids, translations, rotation_matrices = load_poses_from_csv(csv_filename)
    
    print(f"Loaded {len(tag_ids)} tag poses")
    print(f"Tag IDs: {tag_ids}")
    
    # Load first A pose from OptiTrack and transform tag poses
    A_first = None
    if optitrack_csv:
        print(f"\nLoading first A pose from OptiTrack file '{optitrack_csv}'...")
        A_first = load_first_A_pose_from_optitrack(optitrack_csv)
        
        if A_first is not None:
            print("Transforming tag poses to be relative to first A pose...")
            translations, rotation_matrices = transform_poses_by_inv_A(
                translations, rotation_matrices, A_first
            )
            print("Tag poses transformed successfully.")
        else:
            print("Warning: Could not load first A pose. Plotting untransformed poses.")
    
    # Load OptiTrack trajectory if provided
    optitrack_positions = None
    if optitrack_csv:
        print(f"\nLoading OptiTrack trajectory from '{optitrack_csv}'...")
        optitrack_positions = load_optitrack_trajectory(optitrack_csv)
    
    print("\nCreating plot...")
    fig = plot_poses_interactive(tag_ids, translations, rotation_matrices, optitrack_positions)
    
    # Save as PNG
    png_filename = csv_filename.replace('.csv', '_plot.png')
    if optitrack_positions is not None:
        png_filename = csv_filename.replace('.csv', '_with_trajectory_plot.png')
    fig.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved to '{png_filename}'")
    
    print("Displaying plot (close window to exit)...")
    plt.show()

if __name__ == "__main__":
    main()
