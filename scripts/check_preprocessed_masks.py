"""Check if attention masks have been preprocessed."""
import os
import numpy as np
from pfp import DATA_DIRS


def check_masks(data_path):
    """Check for preprocessed attention masks in the dataset."""
    print(f"\nChecking: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"  Path does not exist!")
        return False
    
    # Check if this is a zarr replay buffer format
    if os.path.exists(os.path.join(data_path, 'replay_buffer.zarr')):
        # Check for attention masks in zarr format
        from pfp.data.replay_buffer import RobotReplayBuffer
        try:
            replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
            
            if "attention_masks" in replay_buffer.keys():
                print(f"  Found attention_masks in replay buffer!")
                # Check shape and values
                attention_masks = replay_buffer["attention_masks"]
                print(f"    Shape: {attention_masks.shape}")
                print(f"    Data type: {attention_masks.dtype}")
                
                # Sample first mask
                if len(attention_masks) > 0:
                    first_mask = attention_masks[0]
                    print(f"    First mask shape: {first_mask.shape}")
                    print(f"    Mask range: [{first_mask.min():.3f}, {first_mask.max():.3f}]")
                    print(f"    Points with attention > 0.5: {(first_mask > 0.5).sum()}")
                
                return True
            else:
                print(f"  No attention_masks found in replay buffer")
                print(f"  Available keys: {list(replay_buffer.keys())}")
                return False
                
        except Exception as e:
            print(f"  Error loading replay buffer: {e}")
            return False
    
    # Old format - check for episode directories
    has_masks = False
    episodes = sorted([d for d in os.listdir(data_path) if d.startswith('episode_')])
    
    if not episodes:
        print(f"  No episode directories found (might be zarr format)")
        return False
    
    # Check first few episodes
    for ep in episodes[:3]:
        ep_path = os.path.join(data_path, ep, 'point_cloud')
        if os.path.exists(ep_path):
            files = os.listdir(ep_path)
            mask_files = [f for f in files if '_attention_mask.npy' in f]
            
            if mask_files:
                has_masks = True
                # Load and check a mask
                mask_path = os.path.join(ep_path, mask_files[0])
                mask = np.load(mask_path)
                print(f"  {ep}: Found {len(mask_files)} masks, shape: {mask.shape}")
                print(f"    Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                print(f"    Points with attention > 0.5: {(mask > 0.5).sum()}")
            else:
                print(f"  {ep}: No attention masks found")
    
    return has_masks


def main():
    task_name = "unplug_charger"
    
    # Check both train and valid segmented data
    train_path = DATA_DIRS.PFP / task_name / "train_segmented"
    valid_path = DATA_DIRS.PFP / task_name / "valid_segmented"
    
    print("Checking for preprocessed attention masks...")
    
    train_has_masks = check_masks(train_path)
    valid_has_masks = check_masks(valid_path)
    
    print("\nSummary:")
    print(f"Train segmented has masks: {train_has_masks}")
    print(f"Valid segmented has masks: {valid_has_masks}")
    
    if not (train_has_masks and valid_has_masks):
        print("\nAttention masks not found! You need to run:")
        print("python scripts/preprocess_attention_masks.py \\")
        print("    --task unplug_charger \\")
        print("    --split both \\")
        print("    --object-ids 31 34 35 92 \\")
        print("    --distance-threshold 0.01 \\")
        print("    --gripper-radius 0.05")
    else:
        print("\nAttention masks found! Ready to train with masked dataset.")


if __name__ == "__main__":
    main()