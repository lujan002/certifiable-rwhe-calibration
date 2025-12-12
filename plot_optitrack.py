#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

def quaternion_to_euler(x, y, z, w):
    """
    Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians.
    Uses the 'xyz' extrinsic rotation convention (equivalent to 'ZYX' intrinsic).
    """
    r = Rotation.from_quat([x, y, z, w])
    euler = r.as_euler('xyz', degrees=False)  # Returns [roll, pitch, yaw]
    return euler

def parse_args():
    parser = argparse.ArgumentParser(description="Plot OptiTrack poses from CSV.")
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="data/high-bay/raw/optitrack_success3.csv",
        help="Path to OptiTrack CSV (default: hardcoded example path).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_file = args.csv_file
    df = pd.read_csv(csv_file, skiprows=7)
    print(f"Loaded data from {csv_file} with shape {df.shape}")
    
    # Extract quaternion columns (columns 2, 3, 4, 5 are indices 1, 2, 3, 4 in 0-indexed)
    # Assuming column 0 is time and columns 1-4 are quat_x, quat_y, quat_z, quat_w
    quat_x = df.iloc[:, 2].values
    quat_y = df.iloc[:, 3].values
    quat_z = df.iloc[:, 4].values
    quat_w = df.iloc[:, 5].values
    
    # Time column (assuming first column is time)
    time = df.iloc[:, 1].values 

    # Convert quaternions to Euler angles
    roll = []
    pitch = []
    yaw = []
    
    for x, y, z, w in zip(quat_x, quat_y, quat_z, quat_w):
        euler = quaternion_to_euler(x, y, z, w)
        roll.append(euler[0])
        pitch.append(euler[1])
        yaw.append(euler[2])
    
    # Convert to degrees for easier interpretation
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle('Optitrack Euler Angles over Time', fontsize=16)
    
    # Roll
    axes[0].plot(time, roll_deg, 'r-', linewidth=1.5)
    axes[0].set_ylabel('Roll (degrees)')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Roll')
    
    # Pitch
    axes[1].plot(time, pitch_deg, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Pitch (degrees)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Pitch')
    
    # Yaw
    axes[2].plot(time, yaw_deg, 'b-', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Yaw (degrees)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Yaw')
    
    plt.tight_layout()
    plt.show()
    
    # Save converted data to new CSV
    output_df = pd.DataFrame({
        'time': time,
        'roll_rad': roll,
        'pitch_rad': pitch,
        'yaw_rad': yaw,
        'roll_deg': roll_deg,
        'pitch_deg': pitch_deg,
        'yaw_deg': yaw_deg
    })
    
    output_file = csv_file.replace('.csv', '_euler.csv')
    output_df.to_csv(output_file, index=False)
    print(f"\nConverted data saved to: {output_file}")

if __name__ == "__main__":
    main()