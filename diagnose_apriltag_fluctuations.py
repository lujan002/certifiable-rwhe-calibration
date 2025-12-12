#!/usr/bin/env python3
"""
Diagnose AprilTag detection fluctuations and identify potential issues.

Analyzes CSV data from AprilTag detection to:
1. Compute distance statistics and identify outliers
2. Visualize distance over time
3. Check for common issues (tag size, calibration, etc.)
4. Suggest filtering thresholds
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter

def load_detection_data(csv_file):
    """Load AprilTag detection CSV."""
    df = pd.read_csv(csv_file)
    
    # Filter out NaN rows (no detections)
    df_valid = df.dropna(subset=['tag_id', 'x', 'y', 'z'])
    
    # Compute distance for each detection
    df_valid['distance'] = np.sqrt(df_valid['x']**2 + df_valid['y']**2 + df_valid['z']**2) / 1000.0  # Convert mm to m
    
    return df_valid

def analyze_fluctuations(df, tag_id=None):
    """Analyze distance fluctuations for a specific tag or all tags."""
    if tag_id is not None:
        df_tag = df[df['tag_id'] == tag_id].copy()
        tag_label = f"Tag {tag_id}"
    else:
        df_tag = df.copy()
        tag_label = "All Tags"
    
    if len(df_tag) == 0:
        print(f"No data found for {tag_label}")
        return None
    
    distances = df_tag['distance'].values
    timestamps = df_tag['timestamp'].values
    
    # Compute statistics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    # Compute frame-to-frame differences
    if len(distances) > 1:
        diffs = np.abs(np.diff(distances))
        max_diff = np.max(diffs)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
    else:
        diffs = []
        max_diff = mean_diff = std_diff = 0
    
    # Identify outliers using IQR method
    Q1 = np.percentile(distances, 25)
    Q3 = np.percentile(distances, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = distances[(distances < lower_bound) | (distances > upper_bound)]
    outlier_pct = len(outliers) / len(distances) * 100
    
    # Compute coefficient of variation (CV) - relative variability
    cv = (std_dist / mean_dist) * 100 if mean_dist > 0 else 0
    
    stats_dict = {
        'tag_label': tag_label,
        'n_detections': len(df_tag),
        'mean_distance': mean_dist,
        'std_distance': std_dist,
        'median_distance': median_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'range': max_dist - min_dist,
        'max_frame_diff': max_diff,
        'mean_frame_diff': mean_diff,
        'std_frame_diff': std_diff,
        'cv_percent': cv,
        'outlier_count': len(outliers),
        'outlier_percent': outlier_pct,
        'distances': distances,
        'timestamps': timestamps,
        'diffs': diffs
    }
    
    return stats_dict

def print_diagnosis(stats_dict):
    """Print diagnostic information."""
    s = stats_dict
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS: {s['tag_label']}")
    print(f"{'='*60}")
    print(f"Total detections: {s['n_detections']}")
    print(f"\nDistance Statistics:")
    print(f"  Mean:     {s['mean_distance']:.4f} m ({s['mean_distance']*1000:.1f} mm)")
    print(f"  Std Dev:  {s['std_distance']:.4f} m ({s['std_distance']*1000:.1f} mm)")
    print(f"  Median:   {s['median_distance']:.4f} m ({s['median_distance']*1000:.1f} mm)")
    print(f"  Range:    {s['min_distance']:.4f} to {s['max_distance']:.4f} m")
    print(f"            ({s['range']*1000:.1f} mm total range)")
    print(f"  CV:       {s['cv_percent']:.2f}% (relative variability)")
    
    print(f"\nFrame-to-Frame Changes:")
    print(f"  Max jump:     {s['max_frame_diff']*1000:.1f} mm")
    print(f"  Mean change:  {s['mean_frame_diff']*1000:.1f} mm")
    print(f"  Std change:   {s['std_frame_diff']*1000:.1f} mm")
    
    print(f"\nOutliers (IQR method):")
    print(f"  Count:        {s['outlier_count']} ({s['outlier_percent']:.1f}%)")
    
    # Diagnosis
    print(f"\n{'='*60}")
    print("ASSESSMENT:")
    print(f"{'='*60}")
    
    issues = []
    recommendations = []
    
    # Check standard deviation
    if s['std_distance'] > 0.05:  # > 5cm
        issues.append(f"High variability: std dev = {s['std_distance']*1000:.1f} mm (expected < 10 mm)")
        recommendations.append("Check camera calibration quality")
        recommendations.append("Verify tag size specification is correct")
    elif s['std_distance'] > 0.02:  # > 2cm
        issues.append(f"Moderate variability: std dev = {s['std_distance']*1000:.1f} mm")
        recommendations.append("Consider improving lighting conditions")
        recommendations.append("Check for motion blur in video")
    
    # Check frame-to-frame jumps
    if s['max_frame_diff'] > 0.1:  # > 10cm
        issues.append(f"Large frame jumps: max = {s['max_frame_diff']*1000:.1f} mm")
        recommendations.append("Filter outliers using median filter or IQR method")
        recommendations.append("Check for detection errors (wrong tag ID, false positives)")
    elif s['max_frame_diff'] > 0.05:  # > 5cm
        issues.append(f"Moderate frame jumps: max = {s['max_frame_diff']*1000:.1f} mm")
        recommendations.append("Apply smoothing filter (e.g., Savitzky-Golay)")
    
    # Check coefficient of variation
    if s['cv_percent'] > 10:
        issues.append(f"High relative variability: CV = {s['cv_percent']:.1f}%")
        recommendations.append("Tag may be too far from camera (small in image)")
        recommendations.append("Improve image quality (resolution, focus)")
    
    # Check outlier percentage
    if s['outlier_percent'] > 10:
        issues.append(f"Many outliers: {s['outlier_percent']:.1f}% of detections")
        recommendations.append("Apply outlier rejection filter")
    
    if len(issues) == 0:
        print("✓ Distance measurements appear stable")
        print("  (std dev < 20 mm, frame jumps < 50 mm)")
    else:
        print("⚠ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
    
    if len(recommendations) > 0:
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    
    return issues, recommendations

def plot_distance_analysis(stats_dict, save_path=None):
    """Create visualization plots."""
    s = stats_dict
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'AprilTag Distance Analysis: {s["tag_label"]}', fontsize=14, fontweight='bold')
    
    # Plot 1: Distance over time
    axes[0].plot(s['timestamps'], s['distances'], 'b-', linewidth=1, alpha=0.7, label='Raw distance')
    axes[0].axhline(s['mean_distance'], color='r', linestyle='--', linewidth=2, label=f'Mean: {s["mean_distance"]:.3f}m')
    axes[0].axhline(s['mean_distance'] + s['std_distance'], color='orange', linestyle=':', linewidth=1.5, label=f'±1σ')
    axes[0].axhline(s['mean_distance'] - s['std_distance'], color='orange', linestyle=':', linewidth=1.5)
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Distance (m)', fontsize=11)
    axes[0].set_title('Distance Over Time', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Frame-to-frame differences
    if len(s['diffs']) > 0:
        axes[1].plot(s['timestamps'][1:], s['diffs'] * 1000, 'g-', linewidth=1, alpha=0.7)
        axes[1].axhline(s['mean_frame_diff'] * 1000, color='r', linestyle='--', linewidth=2, 
                       label=f'Mean: {s["mean_frame_diff"]*1000:.1f}mm')
        axes[1].axhline(s['mean_frame_diff'] * 1000 + s['std_frame_diff'] * 1000, color='orange', 
                       linestyle=':', linewidth=1.5, label='±1σ')
        axes[1].axhline(s['mean_frame_diff'] * 1000 - s['std_frame_diff'] * 1000, color='orange', 
                       linestyle=':', linewidth=1.5)
        axes[1].set_xlabel('Time (s)', fontsize=11)
        axes[1].set_ylabel('Frame-to-Frame Change (mm)', fontsize=11)
        axes[1].set_title('Frame-to-Frame Distance Changes', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    # Plot 3: Histogram of distances
    axes[2].hist(s['distances'], bins=50, edgecolor='black', alpha=0.7)
    axes[2].axvline(s['mean_distance'], color='r', linestyle='--', linewidth=2, label=f'Mean: {s["mean_distance"]:.3f}m')
    axes[2].axvline(s['median_distance'], color='g', linestyle='--', linewidth=2, label=f'Median: {s["median_distance"]:.3f}m')
    axes[2].set_xlabel('Distance (m)', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Distance Distribution', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    return fig

def suggest_filtering(stats_dict):
    """Suggest filtering parameters."""
    s = stats_dict
    
    print(f"\n{'='*60}")
    print("FILTERING SUGGESTIONS:")
    print(f"{'='*60}")
    
    # IQR-based outlier removal
    Q1 = np.percentile(s['distances'], 25)
    Q3 = np.percentile(s['distances'], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"\n1. Outlier Removal (IQR method):")
    print(f"   Remove distances outside: [{lower_bound:.4f}, {upper_bound:.4f}] m")
    print(f"   This would remove {s['outlier_count']} outliers ({s['outlier_percent']:.1f}%)")
    
    # Median filter window size
    if s['std_frame_diff'] > 0.02:  # > 2cm frame changes
        window_size = min(11, max(5, int(len(s['distances']) / 20)))
        print(f"\n2. Median Filter:")
        print(f"   Window size: {window_size} frames")
        print(f"   This will smooth out short-term fluctuations")
    
    # Savitzky-Golay filter
    if len(s['distances']) > 15:
        window_length = min(15, len(s['distances']) // 10 * 2 + 1)
        if window_length % 2 == 0:
            window_length += 1
        print(f"\n3. Savitzky-Golay Smoothing:")
        print(f"   Window length: {window_length} frames")
        print(f"   Polynomial order: 3")
        print(f"   This preserves trends while smoothing noise")

def main():
    parser = argparse.ArgumentParser(description="Diagnose AprilTag detection fluctuations")
    parser.add_argument("csv_file", help="Path to AprilTag detection CSV file")
    parser.add_argument("--tag-id", type=int, default=None, help="Analyze specific tag ID (default: all tags)")
    parser.add_argument("--save-plot", type=str, default=None, help="Save plot to file (default: display)")
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_file}...")
    df = load_detection_data(args.csv_file)
    
    print(f"Loaded {len(df)} valid detections")
    if args.tag_id is None:
        print(f"Tag IDs found: {sorted(df['tag_id'].unique())}")
    
    # Analyze
    stats_dict = analyze_fluctuations(df, tag_id=args.tag_id)
    
    if stats_dict is None:
        return
    
    # Print diagnosis
    issues, recommendations = print_diagnosis(stats_dict)
    
    # Suggest filtering
    suggest_filtering(stats_dict)
    
    # Create plots
    plot_path = args.save_plot
    if plot_path is None and args.tag_id is not None:
        plot_path = args.csv_file.replace('.csv', f'_tag{args.tag_id}_diagnosis.png')
    elif plot_path is None:
        plot_path = args.csv_file.replace('.csv', '_diagnosis.png')
    
    plot_distance_analysis(stats_dict, save_path=plot_path)

if __name__ == "__main__":
    main()
