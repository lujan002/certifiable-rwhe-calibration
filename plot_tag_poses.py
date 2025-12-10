#!/usr/bin/env python3
"""
Interactive 3D visualization of tag poses from CSV file.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

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

def plot_poses_interactive(tag_ids, translations, rotation_matrices, arrow_scale=0.2):
    """Create interactive 3D plot of tag poses."""
    fig = go.Figure()
    
    # Extract positions
    xs = translations[:, 0]
    ys = translations[:, 1]
    zs = translations[:, 2]
    
    # Extract z-axis directions from rotation matrices (third column)
    z_axes = rotation_matrices[:, :, 2]  # Shape: (n_tags, 3)
    
    # Calculate arrow endpoints
    arrow_lengths = arrow_scale * np.linalg.norm(z_axes, axis=1)
    arrow_xs = xs + arrow_scale * z_axes[:, 0]
    arrow_ys = ys + arrow_scale * z_axes[:, 1]
    arrow_zs = zs + arrow_scale * z_axes[:, 2]
    
    # Plot tag origins
    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers+text',
        marker=dict(
            size=8,
            color=tag_ids,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Tag ID")
        ),
        text=[f"tag {int(tid)}" for tid in tag_ids],
        textposition="middle right",
        name="Tag Origins"
    ))
    
    # Plot pose arrows (z-axis)
    for i in range(len(tag_ids)):
        fig.add_trace(go.Scatter3d(
            x=[xs[i], arrow_xs[i]],
            y=[ys[i], arrow_ys[i]],
            z=[zs[i], arrow_zs[i]],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=[6, 8]),
            showlegend=(i == 0),
            name="Z-axis" if i == 0 else "",
            hoverinfo='skip'
        ))
    
    # Set layout
    fig.update_layout(
        title="Tag Poses Visualization",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    return fig

def main():
    csv_filename = "tag_poses.csv"
    
    if len(sys.argv) > 1:
        csv_filename = sys.argv[1]
    
    print(f"Loading poses from '{csv_filename}'...")
    tag_ids, translations, rotation_matrices = load_poses_from_csv(csv_filename)
    
    print(f"Loaded {len(tag_ids)} tag poses")
    print(f"Tag IDs: {tag_ids}")
    
    print("Creating interactive plot...")
    fig = plot_poses_interactive(tag_ids, translations, rotation_matrices)
    
    print("Displaying plot (close browser window to exit)...")
    fig.show()
    
    # Optionally save as HTML
    html_filename = csv_filename.replace('.csv', '_plot.html')
    fig.write_html(html_filename)
    print(f"Plot also saved to '{html_filename}'")

if __name__ == "__main__":
    main()
