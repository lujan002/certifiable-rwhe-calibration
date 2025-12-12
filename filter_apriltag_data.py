#!/usr/bin/env python3
"""
Filter and smooth AprilTag detection data to reduce fluctuations.

Applies various filtering methods:
1. Outlier removal (IQR method)
2. Median filter
3. Savitzky-Golay smoothing
"""

import argparse
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy import stats

def remove_outliers_iqr(df, tag_id=None, multiplier=1.5):
    """Remove outliers using IQR method."""
    if tag_id is not None:
        df_tag = df[df['tag_id'] == tag_id].copy()
    else:
        df_tag = df.copy()
    
    # Compute distance
    df_tag['distance'] = np.sqrt(df_tag['x']**2 + df_tag['y']**2 + df_tag['z']**2) / 1000.0
    
    # Compute IQR bounds
    Q1 = df_tag['distance'].quantile(0.25)
    Q3 = df_tag['distance'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Filter
    mask = (df_tag['distance'] >= lower_bound) & (df_tag['distance'] <= upper_bound)
    df_filtered = df_tag[mask].copy()
    
    removed = len(df_tag) - len(df_filtered)
    print(f"  Removed {removed} outliers ({removed/len(df_tag)*100:.1f}%)")
    print(f"  Bounds: [{lower_bound:.4f}, {upper_bound:.4f}] m")
    
    return df_filtered.drop(columns=['distance'])

def apply_median_filter(df, tag_id=None, window_size=5):
    """Apply median filter to smooth distance measurements."""
    if tag_id is not None:
        df_tag = df[df['tag_id'] == tag_id].copy()
        other_df = df[df['tag_id'] != tag_id]
    else:
        df_tag = df.copy()
        other_df = pd.DataFrame()
    
    # Sort by timestamp
    df_tag = df_tag.sort_values('timestamp').reset_index(drop=True)
    
    # Compute distance
    distances = np.sqrt(df_tag['x']**2 + df_tag['y']**2 + df_tag['z']**2) / 1000.0
    
    # Apply median filter
    filtered_distances = medfilt(distances, kernel_size=window_size)
    
    # Scale translation vectors to match filtered distances
    scale_factors = filtered_distances / (distances + 1e-10)  # Avoid division by zero
    
    df_tag['x'] = df_tag['x'] * scale_factors
    df_tag['y'] = df_tag['y'] * scale_factors
    df_tag['z'] = df_tag['z'] * scale_factors
    
    # Recompute quaternion if needed (rotation should be unchanged)
    # For now, we'll keep the original quaternion
    
    print(f"  Applied median filter with window size {window_size}")
    
    # Combine back
    if len(other_df) > 0:
        return pd.concat([df_tag, other_df], ignore_index=True)
    return df_tag

def apply_savgol_filter(df, tag_id=None, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter to smooth distance measurements."""
    if tag_id is not None:
        df_tag = df[df['tag_id'] == tag_id].copy()
        other_df = df[df['tag_id'] != tag_id]
    else:
        df_tag = df.copy()
        other_df = pd.DataFrame()
    
    # Sort by timestamp
    df_tag = df_tag.sort_values('timestamp').reset_index(drop=True)
    
    if len(df_tag) < window_length:
        print(f"  Warning: Not enough data points ({len(df_tag)}) for window size {window_length}")
        return df
    
    # Compute distance
    distances = np.sqrt(df_tag['x']**2 + df_tag['y']**2 + df_tag['z']**2) / 1000.0
    
    # Apply Savitzky-Golay filter
    filtered_distances = savgol_filter(distances, window_length, polyorder)
    
    # Scale translation vectors to match filtered distances
    scale_factors = filtered_distances / (distances + 1e-10)
    
    df_tag['x'] = df_tag['x'] * scale_factors
    df_tag['y'] = df_tag['y'] * scale_factors
    df_tag['z'] = df_tag['z'] * scale_factors
    
    print(f"  Applied Savitzky-Golay filter (window={window_length}, order={polyorder})")
    
    # Combine back
    if len(other_df) > 0:
        return pd.concat([df_tag, other_df], ignore_index=True)
    return df_tag

def filter_by_max_change(df, tag_id=None, max_change_mm=50):
    """Remove frames with excessive frame-to-frame changes."""
    if tag_id is not None:
        df_tag = df[df['tag_id'] == tag_id].copy()
        other_df = df[df['tag_id'] != tag_id]
    else:
        df_tag = df.copy()
        other_df = pd.DataFrame()
    
    # Sort by timestamp
    df_tag = df_tag.sort_values('timestamp').reset_index(drop=True)
    
    if len(df_tag) < 2:
        return df
    
    # Compute distances
    distances = np.sqrt(df_tag['x']**2 + df_tag['y']**2 + df_tag['z']**2) / 1000.0
    
    # Compute frame-to-frame changes
    diffs = np.abs(np.diff(distances)) * 1000  # Convert to mm
    
    # Mark frames with excessive changes
    # Keep first frame, then check subsequent frames
    keep_mask = np.ones(len(df_tag), dtype=bool)
    for i in range(1, len(df_tag)):
        if diffs[i-1] > max_change_mm:
            keep_mask[i] = False
    
    df_filtered = df_tag[keep_mask].copy()
    
    removed = len(df_tag) - len(df_filtered)
    print(f"  Removed {removed} frames with changes > {max_change_mm} mm ({removed/len(df_tag)*100:.1f}%)")
    
    # Combine back
    if len(other_df) > 0:
        return pd.concat([df_filtered, other_df], ignore_index=True)
    return df_filtered

def main():
    parser = argparse.ArgumentParser(description="Filter AprilTag detection data")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument("--tag-id", type=int, default=None, help="Filter specific tag ID (default: all)")
    parser.add_argument("--remove-outliers", action="store_true", help="Remove outliers using IQR method")
    parser.add_argument("--median-filter", type=int, default=None, help="Apply median filter with given window size")
    parser.add_argument("--savgol-filter", type=int, default=None, nargs=2, metavar=('WINDOW', 'ORDER'),
                       help="Apply Savitzky-Golay filter (window_length polyorder)")
    parser.add_argument("--max-change", type=float, default=None, metavar='MM',
                       help="Remove frames with frame-to-frame changes > MM millimeters")
    args = parser.parse_args()
    
    print(f"Loading {args.input_csv}...")
    df = pd.read_csv(args.input_csv)
    
    # Filter out NaN rows for processing
    df_valid = df.dropna(subset=['tag_id', 'x', 'y', 'z'])
    df_nan = df[df['tag_id'].isna() | df['x'].isna()]
    
    print(f"Loaded {len(df_valid)} valid detections")
    if args.tag_id is None:
        print(f"Tag IDs: {sorted(df_valid['tag_id'].unique())}")
    
    # Apply filters
    df_filtered = df_valid.copy()
    
    if args.remove_outliers:
        print("\nRemoving outliers (IQR method)...")
        df_filtered = remove_outliers_iqr(df_filtered, tag_id=args.tag_id)
    
    if args.max_change is not None:
        print(f"\nFiltering by max change ({args.max_change} mm)...")
        df_filtered = filter_by_max_change(df_filtered, tag_id=args.tag_id, max_change_mm=args.max_change)
    
    if args.median_filter is not None:
        print(f"\nApplying median filter...")
        df_filtered = apply_median_filter(df_filtered, tag_id=args.tag_id, window_size=args.median_filter)
    
    if args.savgol_filter is not None:
        window_length, polyorder = args.savgol_filter
        print(f"\nApplying Savitzky-Golay filter...")
        df_filtered = apply_savgol_filter(df_filtered, tag_id=args.tag_id, 
                                         window_length=window_length, polyorder=polyorder)
    
    # Combine with NaN rows (preserve structure)
    # Re-sort by timestamp to maintain order
    if len(df_nan) > 0:
        df_output = pd.concat([df_filtered, df_nan], ignore_index=True)
    else:
        df_output = df_filtered
    
    df_output = df_output.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nWriting filtered data to {args.output_csv}...")
    print(f"  Original: {len(df)} rows")
    print(f"  Filtered: {len(df_output)} rows")
    print(f"  Removed: {len(df) - len(df_output)} rows")
    
    df_output.to_csv(args.output_csv, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
