#!/usr/bin/env python3
"""
Visualize Optitrack pose data from CSV file.
Creates 3D trajectory plot and time-series plots of position and orientation.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

def quaternion_to_euler(x, y, z, w):
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians."""
    r = Rotation.from_quat([x, y, z, w])
    euler = r.as_euler('xyz', degrees=False)  # Returns [roll, pitch, yaw]
    return euler

def load_optitrack_data(csv_file):
    """Load Optitrack data from CSV file."""
    # Skip first 7 rows (header information)
    df = pd.read_csv(csv_file, skiprows=7)
    
    # Extract data columns
    # Column indices: 0=Frame, 1=Time, 2-5=Quaternion (X,Y,Z,W), 6-8=Position (X,Y,Z)
    time = df.iloc[:, 1].values
    quat_x = df.iloc[:, 2].values
    quat_y = df.iloc[:, 3].values
    quat_z = df.iloc[:, 4].values
    quat_w = df.iloc[:, 5].values
    pos_x = df.iloc[:, 6].values / 1000.0  # Convert mm to m
    pos_y = df.iloc[:, 7].values / 1000.0
    pos_z = df.iloc[:, 8].values / 1000.0
    
    # Convert quaternions to Euler angles
    roll = []
    pitch = []
    yaw = []
    rotation_matrices = []
    
    for x, y, z, w in zip(quat_x, quat_y, quat_z, quat_w):
        r = Rotation.from_quat([x, y, z, w])
        euler = r.as_euler('xyz', degrees=False)
        roll.append(euler[0])
        pitch.append(euler[1])
        yaw.append(euler[2])
        rotation_matrices.append(r.as_matrix())
    
    return {
        'time': time,
        'position': np.column_stack([pos_x, pos_y, pos_z]),
        'quaternion': np.column_stack([quat_x, quat_y, quat_z, quat_w]),
        'euler': np.column_stack([roll, pitch, yaw]),
        'rotation_matrices': np.array(rotation_matrices)
    }

def draw_coordinate_frame(ax, position, rotation_matrix, scale=0.05):
    """Draw a coordinate frame at the given position."""
    # X axis (red)
    x_end = position + rotation_matrix[:, 0] * scale
    ax.plot([position[0], x_end[0]], [position[1], x_end[1]], [position[2], x_end[2]], 
            'r-', linewidth=2)
    
    # Y axis (green)
    y_end = position + rotation_matrix[:, 1] * scale
    ax.plot([position[0], y_end[0]], [position[1], y_end[1]], [position[2], y_end[2]], 
            'g-', linewidth=2)
    
    # Z axis (blue)
    z_end = position + rotation_matrix[:, 2] * scale
    ax.plot([position[0], z_end[0]], [position[1], z_end[1]], [position[2], z_end[2]], 
            'b-', linewidth=2)

def visualize_3d_trajectory(data, frame_interval=50):
    """Create 3D trajectory visualization without coordinate frames."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    positions = data['position']
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=1.5, alpha=0.6, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=100, marker='s', label='End', zorder=5)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('Optitrack 3D Trajectory', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig

def visualize_position_time(data):
    """Create position vs time plots."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Position over Time', fontsize=14, fontweight='bold')
    
    time = data['time']
    positions = data['position']
    
    # X position
    axes[0].plot(time, positions[:, 0], 'r-', linewidth=1.5)
    axes[0].set_ylabel('X (m)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('X Position')
    
    # Y position
    axes[1].plot(time, positions[:, 1], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Y (m)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Y Position')
    
    # Z position
    axes[2].plot(time, positions[:, 2], 'b-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Z (m)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Z Position')
    
    plt.tight_layout()
    return fig

def visualize_orientation_time(data):
    """Create orientation (Euler angles) vs time plots."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('Orientation (Euler Angles) over Time', fontsize=14, fontweight='bold')
    
    time = data['time']
    euler = data['euler']
    euler_deg = np.degrees(euler)
    
    # Roll
    axes[0].plot(time, euler_deg[:, 0], 'r-', linewidth=1.5)
    axes[0].set_ylabel('Roll (degrees)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Roll')
    
    # Pitch
    axes[1].plot(time, euler_deg[:, 1], 'g-', linewidth=1.5)
    axes[1].set_ylabel('Pitch (degrees)', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Pitch')
    
    # Yaw
    axes[2].plot(time, euler_deg[:, 2], 'b-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Yaw (degrees)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Yaw')
    
    plt.tight_layout()
    return fig

def visualize_2d_projections(data):
    """Create 2D projections of the trajectory."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('2D Projections of Trajectory', fontsize=14, fontweight='bold')
    
    positions = data['position']
    
    # XY projection
    axes[0].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5, alpha=0.7)
    axes[0].scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', 
                   label='Start', zorder=5)
    axes[0].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='s', 
                   label='End', zorder=5)
    axes[0].set_xlabel('X (m)', fontsize=11)
    axes[0].set_ylabel('Y (m)', fontsize=11)
    axes[0].set_title('XY Projection')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axis('equal')
    
    # XZ projection
    axes[1].plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1.5, alpha=0.7)
    axes[1].scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', 
                   label='Start', zorder=5)
    axes[1].scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='s', 
                   label='End', zorder=5)
    axes[1].set_xlabel('X (m)', fontsize=11)
    axes[1].set_ylabel('Z (m)', fontsize=11)
    axes[1].set_title('XZ Projection')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axis('equal')
    
    # YZ projection
    axes[2].plot(positions[:, 1], positions[:, 2], 'b-', linewidth=1.5, alpha=0.7)
    axes[2].scatter(positions[0, 1], positions[0, 2], c='green', s=100, marker='o', 
                   label='Start', zorder=5)
    axes[2].scatter(positions[-1, 1], positions[-1, 2], c='red', s=100, marker='s', 
                   label='End', zorder=5)
    axes[2].set_xlabel('Y (m)', fontsize=11)
    axes[2].set_ylabel('Z (m)', fontsize=11)
    axes[2].set_title('YZ Projection')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axis('equal')
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Visualize Optitrack poses from CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="data/high-bay/raw/optitrack_success1_cut.csv",
        help="Path to Optitrack CSV file.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=50,
        help="Interval between coordinate frames in 3D plot (default: 50).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save plots to files instead of displaying.",
    )
    args = parser.parse_args()
    
    print(f"Loading Optitrack data from {args.csv_file}...")
    data = load_optitrack_data(args.csv_file)
    
    print(f"Loaded {len(data['time'])} poses")
    print(f"Time range: {data['time'][0]:.3f} to {data['time'][-1]:.3f} seconds")
    print(f"Duration: {data['time'][-1] - data['time'][0]:.3f} seconds")
    print(f"Position range:")
    print(f"  X: [{data['position'][:, 0].min():.3f}, {data['position'][:, 0].max():.3f}] m")
    print(f"  Y: [{data['position'][:, 1].min():.3f}, {data['position'][:, 1].max():.3f}] m")
    print(f"  Z: [{data['position'][:, 2].min():.3f}, {data['position'][:, 2].max():.3f}] m")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    fig1 = visualize_3d_trajectory(data, frame_interval=args.frame_interval)
    fig2 = visualize_position_time(data)
    fig3 = visualize_orientation_time(data)
    fig4 = visualize_2d_projections(data)
    
    if args.save:
        base_name = args.csv_file.replace('.csv', '')
        fig1.savefig(f"{base_name}_3d_trajectory.png", dpi=150, bbox_inches='tight')
        fig2.savefig(f"{base_name}_position_time.png", dpi=150, bbox_inches='tight')
        fig3.savefig(f"{base_name}_orientation_time.png", dpi=150, bbox_inches='tight')
        fig4.savefig(f"{base_name}_2d_projections.png", dpi=150, bbox_inches='tight')
        print(f"\nPlots saved:")
        print(f"  - {base_name}_3d_trajectory.png")
        print(f"  - {base_name}_position_time.png")
        print(f"  - {base_name}_orientation_time.png")
        print(f"  - {base_name}_2d_projections.png")
    else:
        plt.show()

if __name__ == "__main__":
    main()
