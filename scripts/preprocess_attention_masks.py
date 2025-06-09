"""
Preprocess attention masks for all episodes in a dataset.
This script loads segmented data and pre-computes attention masks,
saving them back to the replay buffer for efficient training.
"""
import os
import sys
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pfp import DATA_DIRS
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.data.dataset_pcd_attention import (
    extract_exact_object_points_from_masks_local,
    create_attention_mask_from_object_points
)

# Import exact mask extraction if available
try:
    from scripts.exact_bbox_extraction_for_replay import extract_exact_object_points_from_masks
except ImportError:
    extract_exact_object_points_from_masks = None


def preprocess_attention_masks(
    data_path: Path,
    object_ids: List[int] = None,
    distance_threshold: float = 0.01,
    gripper_radius: float = 0.05,
    n_points: int = None,  # If None, don't sample
    overwrite: bool = False
) -> None:
    """
    Preprocess attention masks for a dataset.
    
    Args:
        data_path: Path to the dataset directory
        object_ids: List of object IDs to track (default: [31, 34, 35, 92])
        distance_threshold: Distance threshold for point-to-object matching
        gripper_radius: Radius around gripper for attention
        n_points: Number of points to sample (if None, saves full masks)
        overwrite: Whether to overwrite existing attention masks
    """
    if object_ids is None:
        object_ids = [31, 34, 35, 92]  # Default for unplug_charger task
    
    # Use appropriate extraction function
    extract_func = extract_exact_object_points_from_masks if extract_exact_object_points_from_masks is not None else extract_exact_object_points_from_masks_local
    
    # Open replay buffer
    replay_buffer = RobotReplayBuffer.create_from_path(str(data_path), mode="a")  # Open in append mode
    
    # Check if attention masks already exist
    if "attention_masks" in replay_buffer.keys() and not overwrite:
        print(f"Attention masks already exist in {data_path}. Use --overwrite to regenerate.")
        return
    
    # Check required keys
    required_keys = ["pcd_xyz", "robot_state", "segmentation_masks", "camera_point_clouds"]
    missing_keys = [key for key in required_keys if key not in replay_buffer.keys()]
    if missing_keys:
        print(f"Error: Missing required keys in replay buffer: {missing_keys}")
        return
    
    # Get dataset info
    n_episodes = replay_buffer.n_episodes
    print(f"Processing {n_episodes} episodes in {data_path}")
    
    # Collect all attention masks first
    all_attention_masks = []
    
    # Process each episode
    for episode_idx in tqdm(range(n_episodes), desc="Processing episodes"):
        episode_data = replay_buffer.get_episode(episode_idx)
        
        # Extract data
        pcd_xyz = episode_data["pcd_xyz"]
        robot_state = episode_data["robot_state"]
        segmentation_masks = episode_data["segmentation_masks"]
        camera_point_clouds = episode_data["camera_point_clouds"]
        
        episode_length = len(pcd_xyz)
        
        # Initialize attention masks for the episode
        # Use full point cloud size if n_points is None
        mask_n_points = n_points if n_points is not None else pcd_xyz.shape[1]
        attention_masks = np.zeros((episode_length, mask_n_points), dtype=np.float32)
        
        for t in range(episode_length):
            # Get point cloud
            pcd = pcd_xyz[t]
            
            # Extract gripper position
            gripper_pos = robot_state[t, :3]
            
            # Extract exact object points from masks
            object_points = extract_func(
                camera_point_clouds[t], 
                segmentation_masks[t], 
                object_ids, 
                debug=(episode_idx == 0 and t == 0)  # Debug first timestep of first episode
            )
            
            # Create attention mask from object points
            full_attention_mask = create_attention_mask_from_object_points(
                pcd, 
                object_points, 
                gripper_pos, 
                distance_threshold,
                gripper_radius
            )
            
            # Save full or sampled attention mask
            if n_points is not None and pcd.shape[0] > n_points:
                # Sample if requested
                rng = np.random.RandomState(seed=episode_idx * 1000 + t)
                random_indices = rng.choice(pcd.shape[0], n_points, replace=False)
                attention_masks[t] = full_attention_mask[random_indices]
            else:
                # Save full mask
                attention_masks[t, :len(full_attention_mask)] = full_attention_mask
        
        # Collect attention masks for this episode
        all_attention_masks.append(attention_masks)
    
    # Concatenate all attention masks
    all_attention_masks = np.concatenate(all_attention_masks, axis=0)
    
    # Add attention masks to replay buffer
    if "attention_masks" in replay_buffer.keys() and overwrite:
        # Delete existing attention masks
        del replay_buffer.root['data']['attention_masks']
    
    # Create new attention_masks dataset in the data group
    # Determine chunk size based on mask shape
    chunk_size = all_attention_masks.shape[1] if all_attention_masks.shape[1] < 10000 else min(4096, all_attention_masks.shape[1])
    replay_buffer.root['data'].create_dataset(
        "attention_masks",
        data=all_attention_masks,
        dtype=np.float32,
        chunks=(1, chunk_size),  # Chunk by timestep
        compressor=replay_buffer.resolve_compressor('default')
    )
    
    print(f"Successfully preprocessed attention masks for {n_episodes} episodes")
    print(f"Total shape: {all_attention_masks.shape}")
    print(f"Saved to: {data_path}")


def main():
    """Process attention masks for common datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess attention masks for datasets")
    parser.add_argument("--task", type=str, default="unplug_charger", help="Task name")
    parser.add_argument("--split", type=str, default="both", choices=["train", "valid", "both"], 
                        help="Which split to process")
    parser.add_argument("--object-ids", type=int, nargs="+", default=[31, 34, 35, 92],
                        help="Object IDs to track")
    parser.add_argument("--distance-threshold", type=float, default=0.01,
                        help="Distance threshold for point-to-object matching")
    parser.add_argument("--gripper-radius", type=float, default=0.05,
                        help="Radius around gripper for attention")
    parser.add_argument("--n-points", type=int, default=None,
                        help="Number of points to sample from point cloud (default: None = save full masks)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing attention masks")
    
    args = parser.parse_args()
    
    # Determine which splits to process
    splits = []
    if args.split in ["train", "both"]:
        splits.append("train_segmented")
    if args.split in ["valid", "both"]:
        splits.append("valid_segmented")
    
    # Process each split
    for split in splits:
        data_path = DATA_DIRS.PFP / args.task / split
        if not data_path.exists():
            print(f"Warning: Path {data_path} does not exist. Skipping.")
            continue
        
        print(f"\nProcessing {split} split...")
        preprocess_attention_masks(
            data_path,
            object_ids=args.object_ids,
            distance_threshold=args.distance_threshold,
            gripper_radius=args.gripper_radius,
            n_points=args.n_points,
            overwrite=args.overwrite
        )


if __name__ == "__main__":
    main()