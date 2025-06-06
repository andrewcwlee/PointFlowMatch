"""Test script to verify preprocessed attention masks."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from pfp import DATA_DIRS
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.data.dataset_pcd_attention import RobotDatasetPcdAttention


def test_preprocessed_masks(task_name="unplug_charger", split="train_segmented"):
    """Test loading preprocessed attention masks."""
    data_path = DATA_DIRS.PFP / task_name / split
    
    # Open replay buffer to check for masks
    replay_buffer = RobotReplayBuffer.create_from_path(str(data_path), mode="r")
    
    print(f"\nChecking {data_path}")
    print(f"Available keys: {list(replay_buffer.keys())}")
    
    if "attention_masks" in replay_buffer.keys():
        print("\n✓ Found preprocessed attention masks!")
        
        # Check a sample episode
        episode_data = replay_buffer.get_episode(0)
        masks = episode_data["attention_masks"]
        
        print(f"Attention masks shape: {masks.shape}")
        print(f"Attention masks dtype: {masks.dtype}")
        print(f"Attention coverage: {masks.mean():.2%}")
        print(f"Min value: {masks.min()}, Max value: {masks.max()}")
        
        # Test dataset loading
        print("\nTesting dataset loading...")
        dataset = RobotDatasetPcdAttention(
            str(data_path),
            n_obs_steps=2,
            n_pred_steps=8,
            subs_factor=5,
            use_pc_color=False,
            n_points=4096,
            use_bounding_box=True,
            bbox_mode="segmentation"
        )
        
        # Test multiple samples to ensure consistency
        print("\nTesting multiple samples...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            if len(sample) == 4:
                pcd, robot_state_obs, robot_state_pred, attention_mask = sample
                print(f"\nSample {i}:")
                print(f"  PCD shape: {pcd.shape}")
                print(f"  Attention mask shape: {attention_mask.shape}")
                print(f"  Attention coverage: {attention_mask.mean():.2%}")
                
                # Verify shapes match
                assert pcd.shape[1] == 4096, f"PCD should have 4096 points, got {pcd.shape[1]}"
                assert attention_mask.shape[-1] == 4096, f"Mask should have 4096 points, got {attention_mask.shape[-1]}"
            else:
                print(f"\n✗ Sample {i} did not return attention mask")
                break
        else:
            print(f"\n✓ All samples successfully loaded with correct shapes!")
            
    else:
        print("\n✗ No preprocessed attention masks found")
        print("Run: bash bash/preprocess_masks.sh")


if __name__ == "__main__":
    # Test both splits
    for split in ["train_segmented", "valid_segmented"]:
        test_preprocessed_masks("unplug_charger", split)
    print("\nDone!")