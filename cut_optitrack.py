#!/usr/bin/env python3
"""
Script to cut an Optitrack CSV file at a given timestep.
Removes all rows up to the timestep and offsets remaining rows' time by that timestep.
"""

import csv
import sys
import argparse


def cut_optitrack_csv(input_file, timestep, output_file=None):
    """
    Cut an Optitrack CSV file at a given timestep.
    
    Args:
        input_file: Path to input CSV file
        timestep: Time in seconds to cut at
        output_file: Path to output CSV file (default: optitrack_cut.csv in same directory)
    """
    if output_file is None:
        # Create output filename by replacing input filename
        import os
        base_dir = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(base_dir, f"{name_without_ext}_cut.csv")
    
    # Read the input file
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if len(rows) < 8:
        raise ValueError("CSV file appears to be too short (expected at least 8 rows including headers)")
    
    # Find the header row (row with "Frame" and "Time (Seconds)")
    header_row_idx = None
    time_col_idx = None
    frame_col_idx = None
    
    for i, row in enumerate(rows):
        if len(row) >= 2:
            if 'Frame' in row[0] and 'Time' in row[1]:
                header_row_idx = i
                frame_col_idx = 0
                time_col_idx = 1
                break
    
    if header_row_idx is None:
        raise ValueError("Could not find header row with 'Frame' and 'Time (Seconds)' columns")
    
    # Separate header rows from data rows
    header_rows = rows[:header_row_idx + 1]  # Include the header row
    data_rows = rows[header_row_idx + 1:]
    
    if len(data_rows) == 0:
        raise ValueError("No data rows found in CSV file")
    
    # Find the first row where time >= timestep
    cutoff_idx = None
    cutoff_time = None
    
    for i, row in enumerate(data_rows):
        if len(row) > time_col_idx:
            try:
                time_val = float(row[time_col_idx])
                if time_val >= timestep:
                    cutoff_idx = i
                    cutoff_time = time_val
                    break
            except (ValueError, IndexError):
                continue
    
    # If timestep is beyond all data, raise an error
    if cutoff_idx is None:
        raise ValueError(f"Timestep {timestep} is beyond all data in the file (last time: {float(data_rows[-1][time_col_idx])})")
    
    # If timestep is between rows, use the next row's time (already handled above)
    # Now cut the data and offset times
    cut_data_rows = data_rows[cutoff_idx:]
    
    # Offset all times by the cutoff time
    for row in cut_data_rows:
        if len(row) > time_col_idx:
            try:
                original_time = float(row[time_col_idx])
                new_time = original_time - cutoff_time
                row[time_col_idx] = f"{new_time:.6f}"
            except (ValueError, IndexError):
                pass
    
    # Update frame numbers to start from 0
    for i, row in enumerate(cut_data_rows):
        if len(row) > frame_col_idx:
            row[frame_col_idx] = str(i)
    
    # Write the output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(header_rows)
        writer.writerows(cut_data_rows)
    
    print(f"Successfully cut CSV file:")
    print(f"  Input: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Cut at timestep: {timestep} seconds")
    print(f"  Actual cutoff time: {cutoff_time} seconds")
    print(f"  Rows removed: {cutoff_idx}")
    print(f"  Rows remaining: {len(cut_data_rows)}")


def main():
    parser = argparse.ArgumentParser(
        description='Cut an Optitrack CSV file at a given timestep',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cut_optitrack.py optitrack.csv 10.5
  python cut_optitrack.py optitrack.csv 10.5 -o output.csv
        """
    )
    parser.add_argument('input_file', help='Path to input Optitrack CSV file')
    parser.add_argument('timestep', type=float, help='Timestep in seconds to cut at')
    parser.add_argument('-o', '--output', dest='output_file', default=None,
                       help='Path to output CSV file (default: <input>_cut.csv)')
    
    args = parser.parse_args()
    
    try:
        cut_optitrack_csv(args.input_file, args.timestep, args.output_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
