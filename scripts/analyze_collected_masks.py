#!/usr/bin/env python3
"""
Analyze collected segmentation masks to identify charger and gripper objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from pfp.data.replay_buffer import RobotReplayBuffer
from tqdm import tqdm


def analyze_object_properties(replay_buffer, n_frames=10):
    """
    Analyze properties of each object across multiple frames.
    """
    # Sample frames evenly across the episode
    total_frames = len(replay_buffer["robot_state"])
    frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
    
    # Camera index mapping
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    # Collect statistics for each object
    object_stats = {}
    
    for frame_idx in tqdm(frame_indices, desc="Analyzing frames"):
        masks = replay_buffer["segmentation_masks"][frame_idx]  # (5, H, W)
        robot_state = replay_buffer["robot_state"][frame_idx]
        gripper_pos = robot_state[:3]
        gripper_state = robot_state[9]  # 0=closed, 1=open
        
        # Analyze each camera
        for cam_idx, cam_name in enumerate(camera_names):
            mask = masks[cam_idx]
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]
            
            for obj_id in unique_ids:
                if obj_id not in object_stats:
                    object_stats[obj_id] = {
                        'frame_count': 0,
                        'total_pixels': 0,
                        'camera_counts': {cam: 0 for cam in camera_names},
                        'positions': [],
                        'when_gripper_open': 0,
                        'when_gripper_closed': 0,
                    }
                
                # Count pixels
                n_pixels = (mask == obj_id).sum()
                object_stats[obj_id]['frame_count'] += 1
                object_stats[obj_id]['total_pixels'] += n_pixels
                object_stats[obj_id]['camera_counts'][cam_name] += 1
                
                # Track gripper state
                if gripper_state > 0.5:
                    object_stats[obj_id]['when_gripper_open'] += 1
                else:
                    object_stats[obj_id]['when_gripper_closed'] += 1
    
    return object_stats


def identify_charger_and_gripper(object_stats):
    """
    Use heuristics to identify charger and gripper objects.
    
    Heuristics:
    - Charger: Visible in most frames, medium pixel count, visible when gripper is both open and closed
    - Gripper: High pixel count, visible in all frames, especially in wrist camera
    """
    candidates = {
        'charger': [],
        'gripper': []
    }
    
    for obj_id, stats in object_stats.items():
        avg_pixels = stats['total_pixels'] / max(stats['frame_count'], 1)
        visibility = stats['frame_count']
        
        # Gripper candidates: very high visibility, large pixel count
        if visibility > 8 and avg_pixels > 500:
            candidates['gripper'].append({
                'id': obj_id,
                'score': visibility * avg_pixels,
                'avg_pixels': avg_pixels,
                'visibility': visibility,
                'wrist_camera': stats['camera_counts']['wrist']
            })
        
        # Charger candidates: medium visibility, medium pixel count
        # Should be visible when gripper is both open (before grasp) and closed (during grasp)
        if 4 < visibility < 9 and 100 < avg_pixels < 2000:
            if stats['when_gripper_open'] > 0 and stats['when_gripper_closed'] > 0:
                candidates['charger'].append({
                    'id': obj_id,
                    'score': visibility * avg_pixels,
                    'avg_pixels': avg_pixels,
                    'visibility': visibility,
                    'open_closed_ratio': stats['when_gripper_open'] / max(stats['when_gripper_closed'], 1)
                })
    
    # Sort by score
    candidates['gripper'].sort(key=lambda x: x['score'], reverse=True)
    candidates['charger'].sort(key=lambda x: x['score'], reverse=True)
    
    return candidates


def visualize_candidate_objects(replay_buffer, candidates, save_dir):
    """
    Visualize the top candidate objects.
    """
    # Use middle frame
    frame_idx = len(replay_buffer["robot_state"]) // 2
    masks = replay_buffer["segmentation_masks"][frame_idx]
    images = replay_buffer["images"][frame_idx]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show RGB images
    for i, (ax, img) in enumerate(zip(axes[0], images[:3])):
        ax.imshow(img)
        ax.set_title(f"Camera {i}")
        ax.axis('off')
    
    # Show top charger candidate
    if candidates['charger']:
        charger_id = candidates['charger'][0]['id']
        combined_mask = np.zeros_like(masks[3])  # Use front camera
        combined_mask[masks[3] == charger_id] = 1
        
        axes[1, 0].imshow(combined_mask, cmap='hot')
        axes[1, 0].set_title(f"Top Charger Candidate (ID: {charger_id})")
        axes[1, 0].axis('off')
    
    # Show top gripper candidates
    if len(candidates['gripper']) >= 2:
        for i, gripper_data in enumerate(candidates['gripper'][:2]):
            gripper_id = gripper_data['id']
            combined_mask = np.zeros_like(masks[4])  # Use wrist camera
            combined_mask[masks[4] == gripper_id] = 1
            
            axes[1, i+1].imshow(combined_mask, cmap='hot')
            axes[1, i+1].set_title(f"Gripper Candidate {i+1} (ID: {gripper_id})")
            axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / "candidate_objects.png", dpi=150, bbox_inches='tight')
    plt.close()


def compute_tight_bounding_boxes(replay_buffer, charger_id, gripper_ids, n_frames=5):
    """
    Compute tight bounding boxes for charger and gripper across frames.
    """
    frame_indices = np.linspace(0, len(replay_buffer["robot_state"])-1, n_frames, dtype=int)
    
    bboxes = {
        'charger': [],
        'gripper': []
    }
    
    for frame_idx in frame_indices:
        pcd = replay_buffer["pcd_xyz"][frame_idx]
        
        # Create combined mask from all cameras
        # This is approximate - better would be to project masks to 3D
        # For now, we'll use height-based filtering
        
        # Charger bbox (approximate based on typical height)
        charger_height_range = [0.8, 1.0]
        charger_mask = (pcd[:, 2] >= charger_height_range[0]) & (pcd[:, 2] <= charger_height_range[1])
        if charger_mask.any():
            charger_points = pcd[charger_mask]
            bbox = {
                'min': charger_points.min(axis=0),
                'max': charger_points.max(axis=0),
                'center': charger_points.mean(axis=0),
                'size': charger_points.max(axis=0) - charger_points.min(axis=0)
            }
            bboxes['charger'].append(bbox)
        
        # Gripper bbox (around end-effector position)
        gripper_pos = replay_buffer["robot_state"][frame_idx][:3]
        gripper_mask = np.linalg.norm(pcd - gripper_pos, axis=1) < 0.15
        if gripper_mask.any():
            gripper_points = pcd[gripper_mask]
            bbox = {
                'min': gripper_points.min(axis=0),
                'max': gripper_points.max(axis=0),
                'center': gripper_points.mean(axis=0),
                'size': gripper_points.max(axis=0) - gripper_points.min(axis=0)
            }
            bboxes['gripper'].append(bbox)
    
    return bboxes


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze collected segmentation masks")
    parser.add_argument("--data_path", type=str, 
                       default="demos/sim/unplug_charger/train_segmented",
                       help="Path to segmented demo data")
    parser.add_argument("--output_dir", type=str, 
                       default="./mask_analysis",
                       help="Output directory")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    
    # Load replay buffer
    replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
    
    # Load object info
    object_info_path = data_path / "object_info.json"
    if object_info_path.exists():
        with open(object_info_path, 'r') as f:
            object_info = json.load(f)
        print(f"Found {len(object_info['all_ids'])} objects")
    else:
        print("No object_info.json found")
        object_info = {'all_ids': []}
    
    # Analyze objects
    print("\nAnalyzing object properties...")
    object_stats = analyze_object_properties(replay_buffer)
    
    # Identify charger and gripper
    print("\nIdentifying charger and gripper...")
    candidates = identify_charger_and_gripper(object_stats)
    
    print("\nTop Charger Candidates:")
    for i, cand in enumerate(candidates['charger'][:3]):
        print(f"  {i+1}. ID {cand['id']}: avg_pixels={cand['avg_pixels']:.0f}, visibility={cand['visibility']}")
    
    print("\nTop Gripper Candidates:")
    for i, cand in enumerate(candidates['gripper'][:5]):
        print(f"  {i+1}. ID {cand['id']}: avg_pixels={cand['avg_pixels']:.0f}, visibility={cand['visibility']}, wrist_camera={cand['wrist_camera']}")
    
    # Visualize candidates
    print("\nCreating visualizations...")
    visualize_candidate_objects(replay_buffer, candidates, output_dir)
    
    # Compute bounding boxes
    if candidates['charger'] and candidates['gripper']:
        charger_id = candidates['charger'][0]['id']
        gripper_ids = [g['id'] for g in candidates['gripper'][:3]]
        
        print("\nComputing tight bounding boxes...")
        bboxes = compute_tight_bounding_boxes(replay_buffer, charger_id, gripper_ids)
        
        # Save results
        results = {
            'charger_id': int(charger_id),
            'gripper_ids': [int(gid) for gid in gripper_ids],
            'charger_bbox_samples': [
                {
                    'center': bbox['center'].tolist(),
                    'size': bbox['size'].tolist()
                }
                for bbox in bboxes['charger']
            ],
            'gripper_bbox_samples': [
                {
                    'center': bbox['center'].tolist(),
                    'size': bbox['size'].tolist()
                }
                for bbox in bboxes['gripper']
            ]
        }
        
        results_path = output_dir / "identified_objects.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_path}")
        print("\nIdentified objects:")
        print(f"  Charger: ID {charger_id}")
        print(f"  Gripper: IDs {gripper_ids}")
        
        # Update original object_info.json
        object_info['charger_id'] = int(charger_id)
        object_info['gripper_ids'] = [int(gid) for gid in gripper_ids]
        
        with open(object_info_path, 'w') as f:
            json.dump(object_info, f, indent=2)
        print(f"\nUpdated {object_info_path}")


if __name__ == "__main__":
    main()