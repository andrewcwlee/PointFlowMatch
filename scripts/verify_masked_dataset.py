"""Verify masked dataset can load zarr data."""
import sys
sys.path.append('/home/andrewlee/_research/PointFlowMatch')

from pfp.data.dataset_pcd_masked import RobotDatasetPcdMasked
from pfp import DATA_DIRS
import numpy as np


def main():
    task_name = "unplug_charger"
    data_path = DATA_DIRS.PFP / task_name / "train_segmented"
    
    print(f"Testing masked dataset with data from: {data_path}")
    
    try:
        # Create dataset
        dataset = RobotDatasetPcdMasked(
            data_path=str(data_path),
            n_obs_steps=2,
            n_pred_steps=32,
            use_pc_color=False,
            n_points=4096,
            object_ids=[31, 34, 35, 92],
            include_gripper=True,
            gripper_radius=0.05,
            min_points=100
        )
        
        print(f"\nDataset created successfully!")
        print(f"Length: {len(dataset)}")
        
        # Test loading first sample
        if len(dataset) > 0:
            print("\nLoading first sample...")
            sample = dataset[0]
            pcd, robot_state_obs, robot_state_pred = sample
            
            print(f"\nSample shapes:")
            print(f"  Point cloud: {pcd.shape}")
            print(f"  Robot state obs: {robot_state_obs.shape}")
            print(f"  Robot state pred: {robot_state_pred.shape}")
            
            # Check statistics
            pcd_np = pcd.numpy()
            valid_points = np.any(pcd_np != 0, axis=-1)
            n_valid_total = valid_points.sum()
            
            print(f"\nPoint cloud statistics:")
            print(f"  Total points: {pcd.shape[1] * pcd.shape[2]}")
            print(f"  Valid (non-zero) points: {n_valid_total}")
            print(f"  Percentage of valid points: {100 * n_valid_total / (pcd.shape[1] * pcd.shape[2]):.1f}%")
            
            # Check per timestep
            for t in range(pcd.shape[0]):
                valid_t = np.any(pcd_np[t] != 0, axis=-1).sum()
                print(f"  Timestep {t}: {valid_t} valid points")
            
            # Get attention mask statistics
            print(f"\nTesting attention mask filtering...")
            try:
                # Access the dataset's internal components to get raw attention mask
                raw_data = dataset.sampler.sample_sequence(0)
                if 'attention_masks' in raw_data:
                    attention_masks = raw_data['attention_masks'][::dataset.subs_factor][:dataset.n_obs_steps]
                    
                    print(f"\nAttention mask statistics:")
                    for t in range(len(attention_masks)):
                        mask = attention_masks[t]
                        total_points = len(mask)
                        points_with_attention = (mask > 0.5).sum()
                        points_high_attention = (mask > 0.8).sum()
                        
                        print(f"  Timestep {t}:")
                        print(f"    Total points in original PC: {total_points}")
                        print(f"    Points with attention > 0.5: {points_with_attention} ({100*points_with_attention/total_points:.1f}%)")
                        print(f"    Points with attention > 0.8: {points_high_attention} ({100*points_high_attention/total_points:.1f}%)")
                        print(f"    Attention range: [{mask.min():.3f}, {mask.max():.3f}]")
                        print(f"    Mean attention: {mask.mean():.3f}")
                        
                        # Show distribution
                        hist, bin_edges = np.histogram(mask, bins=[0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
                        print(f"    Attention distribution:")
                        for i in range(len(hist)):
                            print(f"      [{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}]: {hist[i]} points ({100*hist[i]/total_points:.1f}%)")
                else:
                    print("  No attention masks found in raw data")
                    
            except Exception as e:
                print(f"  Error analyzing attention masks: {e}")
                
            print(f"\nObject filtering summary:")
            print(f"  Target object IDs: {dataset.object_ids}")
            print(f"  Original point clouds are filtered to keep only points")
            print(f"  belonging to these objects (attention > 0.5)")
            print(f"  Final point clouds maintain {dataset.n_points} points through")
            print(f"  sampling (with replacement if needed)")
                
            print("\nDataset is working correctly!")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()