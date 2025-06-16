"""Analyze collected demonstration data for BC training"""

import zarr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pfp import DATA_DIRS


def analyze_data(data_path: str, task: str = "unplug_charger"):
    """Analyze demonstration data for BC training"""
    
    data_path = Path(data_path)
    print(f"Analyzing data at: {data_path}")
    
    if not data_path.exists():
        print(f"Data path does not exist: {data_path}")
        return
    
    # Check data structure
    data_dir = data_path / "data"
    meta_dir = data_path / "meta"
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    print("\nData Structure:")
    print("Data directory contents:")
    for item in data_dir.iterdir():
        if item.is_dir():
            try:
                data_array = zarr.open(str(item), mode='r')
                print(f"  {item.name}: {data_array.shape} ({data_array.dtype})")
            except Exception as e:
                print(f"  {item.name}: Error loading - {e}")
    
    if meta_dir.exists():
        print("Meta directory contents:")
        for item in meta_dir.iterdir():
            if item.is_dir():
                try:
                    data_array = zarr.open(str(item), mode='r')
                    print(f"  {item.name}: {data_array.shape} ({data_array.dtype})")
                except Exception as e:
                    print(f"  {item.name}: Error loading - {e}")
    
    # Load robot states
    robot_state_path = data_dir / "robot_state"
    if not robot_state_path.exists():
        print(f"Robot state data not found: {robot_state_path}")
        return
    
    robot_states = zarr.open(str(robot_state_path), mode='r')[:]
    print(f"\nRobot State Analysis:")
    print(f"  Total timesteps: {len(robot_states)}")
    print(f"  State dimension: {robot_states.shape[1]}")
    
    # Position analysis
    positions = robot_states[:, :3]
    print(f"  Position ranges:")
    print(f"    X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}] (mean: {positions[:, 0].mean():.3f})")
    print(f"    Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] (mean: {positions[:, 1].mean():.3f})")
    print(f"    Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}] (mean: {positions[:, 2].mean():.3f})")
    
    # Rotation analysis (6D representation)
    rotations = robot_states[:, 3:9]
    print(f"  6D Rotation ranges:")
    for i in range(6):
        print(f"    R{i}: [{rotations[:, i].min():.3f}, {rotations[:, i].max():.3f}]")
    
    # Check 6D rotation validity (should have norm around sqrt(2))
    rot_norms = np.linalg.norm(rotations, axis=1)
    print(f"  6D Rotation norms: mean={rot_norms.mean():.3f}, std={rot_norms.std():.3f}")
    print(f"  Expected norm: ~{np.sqrt(2):.3f}")
    
    # Gripper analysis
    grippers = robot_states[:, 9]
    print(f"  Gripper states:")
    print(f"    Range: [{grippers.min():.3f}, {grippers.max():.3f}]")
    print(f"    Unique values: {len(np.unique(grippers))}")
    
    # Analyze transitions (action deltas)
    print(f"\nTransition Analysis:")
    transitions = robot_states[1:] - robot_states[:-1]
    
    pos_deltas = transitions[:, :3]
    pos_delta_norms = np.linalg.norm(pos_deltas, axis=1)
    print(f"  Position deltas:")
    print(f"    Mean magnitude: {pos_delta_norms.mean():.4f}")
    print(f"    Max magnitude: {pos_delta_norms.max():.4f}")
    print(f"    95th percentile: {np.percentile(pos_delta_norms, 95):.4f}")
    
    rot_deltas = transitions[:, 3:9]
    rot_delta_norms = np.linalg.norm(rot_deltas, axis=1)
    print(f"  Rotation deltas:")
    print(f"    Mean magnitude: {rot_delta_norms.mean():.4f}")
    print(f"    Max magnitude: {rot_delta_norms.max():.4f}")
    
    grip_deltas = transitions[:, 9]
    print(f"  Gripper deltas:")
    print(f"    Mean: {grip_deltas.mean():.4f}")
    print(f"    Max abs: {np.abs(grip_deltas).max():.4f}")
    
    # Episode analysis
    episode_ends_path = meta_dir / "episode_ends"
    if episode_ends_path.exists():
        episode_ends = zarr.open(str(episode_ends_path), mode='r')[:]
    else:
        print("Warning: No episode_ends found, assuming single episode")
        episode_ends = np.array([len(robot_states)])
    num_episodes = len(episode_ends)
    episode_lengths = []
    
    prev_end = 0
    for end in episode_ends:
        episode_lengths.append(end - prev_end)
        prev_end = end
    
    print(f"\nEpisode Analysis:")
    print(f"  Number of episodes: {num_episodes}")
    print(f"  Episode lengths: mean={np.mean(episode_lengths):.1f}, "
          f"min={np.min(episode_lengths)}, max={np.max(episode_lengths)}")
    
    # Point cloud analysis
    pcd_path = data_dir / "pcd_xyz"
    if pcd_path.exists():
        pcds = zarr.open(str(pcd_path), mode='r')[:]
    else:
        print("Warning: No point cloud data found")
        pcds = None
    print(f"\nPoint Cloud Analysis:")
    if pcds is not None:
        print(f"  Shape: {pcds.shape}")
        print(f"  Points per timestep: {pcds.shape[1]}")
        
        # Check for valid point clouds
        valid_points = ~np.isnan(pcds).any(axis=2)
        avg_valid_points = valid_points.sum(axis=1).mean()
        print(f"  Average valid points per timestep: {avg_valid_points:.1f}")
        
        # Point cloud bounds
        valid_pcds = pcds[~np.isnan(pcds).any(axis=2)]
        if len(valid_pcds) > 0:
            print(f"  Point cloud bounds:")
            print(f"    X: [{valid_pcds[:, 0].min():.3f}, {valid_pcds[:, 0].max():.3f}]")
            print(f"    Y: [{valid_pcds[:, 1].min():.3f}, {valid_pcds[:, 1].max():.3f}]")
            print(f"    Z: [{valid_pcds[:, 2].min():.3f}, {valid_pcds[:, 2].max():.3f}]")
    else:
        print("  No point cloud data available")
    
    # Check attention masks
    mask_dir = data_path / "data" / "attention_masks"
    if mask_dir.exists():
        mask_files = list(mask_dir.iterdir())
        print(f"\nAttention Masks:")
        print(f"  Found {len(mask_files)} mask files")
        
        if mask_files:
            # Sample a few masks
            try:
                # Try to load as zarr array
                sample_mask = zarr.open(str(mask_files[0]), mode='r')[:]
                print(f"  Sample mask shape: {sample_mask.shape}")
                print(f"  Sample mask stats: mean={sample_mask.mean():.3f}, "
                      f"max={sample_mask.max():.3f}, "
                      f"coverage={np.mean(sample_mask > 0):.1%}")
            except Exception as e:
                # If not zarr, try numpy
                try:
                    sample_mask = np.load(str(mask_files[0]))
                    print(f"  Sample mask shape: {sample_mask.shape}")
                    print(f"  Sample mask stats: mean={sample_mask.mean():.3f}, "
                          f"max={sample_mask.max():.3f}, "
                          f"coverage={np.mean(sample_mask > 0):.1%}")
                except:
                    print(f"  Could not load sample mask file: {mask_files[0].name}")
    else:
        print(f"\nAttention Masks: Not found")
    
    # Generate some plots
    print(f"\nGenerating analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Position trajectories
    axes[0, 0].plot(positions[:, 0], label='X', alpha=0.7)
    axes[0, 0].plot(positions[:, 1], label='Y', alpha=0.7)
    axes[0, 0].plot(positions[:, 2], label='Z', alpha=0.7)
    axes[0, 0].set_title('Position Trajectories')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Position (m)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Position deltas histogram
    axes[0, 1].hist(pos_delta_norms, bins=50, alpha=0.7)
    axes[0, 1].set_title('Position Delta Magnitudes')
    axes[0, 1].set_xlabel('Delta magnitude (m)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True)
    
    # 6D rotation norms
    axes[0, 2].hist(rot_norms, bins=50, alpha=0.7)
    axes[0, 2].axvline(np.sqrt(2), color='red', linestyle='--', label=f'Expected ({np.sqrt(2):.3f})')
    axes[0, 2].set_title('6D Rotation Norms')
    axes[0, 2].set_xlabel('Norm')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Gripper states
    axes[1, 0].plot(grippers)
    axes[1, 0].set_title('Gripper States')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Gripper State')
    axes[1, 0].grid(True)
    
    # Episode lengths
    axes[1, 1].hist(episode_lengths, bins=20, alpha=0.7)
    axes[1, 1].set_title('Episode Lengths')
    axes[1, 1].set_xlabel('Length (timesteps)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True)
    
    # Valid points per timestep
    valid_points_per_step = valid_points.sum(axis=1)
    axes[1, 2].plot(valid_points_per_step)
    axes[1, 2].set_title('Valid Points per Timestep')
    axes[1, 2].set_xlabel('Timestep')
    axes[1, 2].set_ylabel('Number of valid points')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = data_path / f"data_analysis_{task}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to: {plot_path}")
    plt.show()
    
    return {
        'num_episodes': num_episodes,
        'total_timesteps': len(robot_states),
        'avg_episode_length': np.mean(episode_lengths),
        'position_delta_mean': pos_delta_norms.mean(),
        'position_delta_max': pos_delta_norms.max(),
        'has_attention_masks': mask_dir.exists(),
        'avg_valid_points': avg_valid_points,
    }


if __name__ == "__main__":
    # Analyze training data
    task = "unplug_charger"
    train_path = DATA_DIRS.PFP / task / "train_segmented"
    
    print("Analyzing training data...")
    train_stats = analyze_data(train_path, task)
    
    # Also analyze validation data if available
    val_path = DATA_DIRS.PFP / task / "valid_segmented"
    if val_path.exists():
        print("\n" + "="*80)
        print("Analyzing validation data...")
        val_stats = analyze_data(val_path, task)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"Training data: {train_stats['num_episodes']} episodes, "
          f"{train_stats['total_timesteps']} timesteps")
    if 'val_stats' in locals():
        print(f"Validation data: {val_stats['num_episodes']} episodes, "
              f"{val_stats['total_timesteps']} timesteps")
    print(f"Data appears valid for BC training: {'Yes' if train_stats['position_delta_mean'] < 0.1 else 'Check large deltas'}")
    print("="*80)