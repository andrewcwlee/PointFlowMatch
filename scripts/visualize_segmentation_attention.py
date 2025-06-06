#!/usr/bin/env python3
"""
Visualize segmentation-based attention masks for debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pfp import DATA_DIRS
from pfp.data.dataset_pcd_attention import RobotDatasetPcdAttention

def visualize_attention(
    data_path: str = None,
    sample_idx: int = 0,
    timestep: int = 0,
    save_path: str = "./attention_vis.png"
):
    """Visualize attention mask on point cloud."""
    
    if data_path is None:
        data_path = DATA_DIRS.PFP / "unplug_charger" / "train_segmented"
    
    # Create dataset
    dataset = RobotDatasetPcdAttention(
        data_path=str(data_path),
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
        use_pc_color=True,  # Use colors for better visualization
        n_points=4096,
        use_bounding_box=True,
        bbox_mode="segmentation",
        object_ids=[31, 34, 35, 92],
        distance_threshold=0.01,
    )
    
    # Load sample
    result = dataset[sample_idx]
    if len(result) != 4:
        print("No attention mask found!")
        return
        
    pcd, robot_state_obs, robot_state_pred, attention_mask = result
    
    # Extract data for visualization
    pcd_t = pcd[timestep]  # Shape: (N, 3) or (N, 6) if color
    att_t = attention_mask[timestep]  # Shape: (N,)
    gripper_pos = robot_state_obs[timestep, :3]
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Original point cloud
    ax1 = fig.add_subplot(121, projection='3d')
    if pcd_t.shape[1] > 3:  # Has color
        colors = pcd_t[:, 3:6]
        ax1.scatter(pcd_t[:, 0], pcd_t[:, 1], pcd_t[:, 2], 
                   c=colors, s=1, alpha=0.6)
    else:
        ax1.scatter(pcd_t[:, 0], pcd_t[:, 1], pcd_t[:, 2], 
                   c='blue', s=1, alpha=0.6)
    
    # Add gripper
    ax1.scatter(*gripper_pos, c='red', s=100, marker='x', linewidths=3)
    ax1.set_title(f"Original Point Cloud (t={timestep})")
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # Right: Attention visualization
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Color by attention (red = high attention, blue = low)
    colors_att = plt.cm.viridis(att_t)[:, :3]
    ax2.scatter(pcd_t[:, 0], pcd_t[:, 1], pcd_t[:, 2], 
               c=colors_att, s=2, alpha=0.8)
    
    # Add gripper
    ax2.scatter(*gripper_pos, c='red', s=100, marker='*', linewidths=3)
    ax2.set_title(f"Attention Weights (t={timestep})")
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # Add colorbar
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    mappable.set_array(att_t)
    cbar = plt.colorbar(mappable, ax=ax2, shrink=0.5, aspect=10)
    cbar.set_label('Attention Weight', rotation=270, labelpad=15)
    
    # Set consistent view angles
    for ax in [ax1, ax2]:
        ax.view_init(elev=20, azim=45)
        ax.set_xlim([-0.3, 0.4])
        ax.set_ylim([-0.2, 0.3])
        ax.set_zlim([0.6, 1.5])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    
    # Print statistics
    print(f"\nAttention Statistics:")
    print(f"  Total points: {len(att_t)}")
    print(f"  Points with attention > 0: {(att_t > 0).sum()}")
    print(f"  Attention coverage: {att_t.mean():.2%}")
    print(f"  Gripper position: {gripper_pos}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="./attention_vis.png")
    
    args = parser.parse_args()
    visualize_attention(
        data_path=args.data_path,
        sample_idx=args.sample,
        timestep=args.timestep,
        save_path=args.save_path
    )