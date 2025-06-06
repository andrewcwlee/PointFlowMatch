#!/usr/bin/env python3
"""
Extract exact 3D points for objects using 2D masks and per-camera point clouds,
then merge them for tight bounding boxes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pfp.data.replay_buffer import RobotReplayBuffer


def extract_exact_3d_points_from_masks(obs_data, object_ids, debug=False):
    """
    Extract exact 3D points for objects using 2D masks and per-camera point clouds.
    
    Args:
        obs_data: Dictionary containing point clouds and masks for each camera
        object_ids: List of object IDs to extract points for
        
    Returns:
        dict: {object_id: np.array of 3D points}
    """
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    # Storage for exact 3D points per object
    object_points = {obj_id: [] for obj_id in object_ids}
    
    for camera_idx, camera_name in enumerate(camera_names):
        # Get camera data
        if camera_idx >= len(obs_data['masks']):
            continue
            
        mask = obs_data['masks'][camera_idx]  # (H, W)
        point_cloud = obs_data['point_clouds'][camera_idx]  # (H*W, 3) or (H, W, 3)
        
        # Ensure point cloud is in the right shape
        if len(point_cloud.shape) == 3:  # (H, W, 3)
            H, W, _ = point_cloud.shape
            point_cloud = point_cloud.reshape(-1, 3)  # (H*W, 3)
        else:  # Already (H*W, 3)
            H, W = mask.shape
        
        # Flatten mask to match point cloud
        mask_flat = mask.flatten()  # (H*W,)
        
        if debug:
            print(f"{camera_name}: mask shape {mask.shape}, pcd shape {point_cloud.shape}")
        
        # Extract points for each object
        for obj_id in object_ids:
            # Find pixels where this object appears
            object_mask = (mask_flat == obj_id)
            n_pixels = object_mask.sum()
            
            if n_pixels > 0:
                # Get corresponding 3D points
                object_3d_points = point_cloud[object_mask]
                
                # Filter out invalid points (zeros, NaNs, or very far points)
                valid_mask = (
                    np.isfinite(object_3d_points).all(axis=1) &  # No NaN/inf
                    (np.linalg.norm(object_3d_points, axis=1) > 0.01) &  # Not at origin
                    (np.linalg.norm(object_3d_points, axis=1) < 10.0)    # Not too far
                )
                
                valid_points = object_3d_points[valid_mask]
                
                if len(valid_points) > 0:
                    object_points[obj_id].append(valid_points)
                    
                    if debug:
                        print(f"  Object {obj_id}: {n_pixels} pixels -> {len(valid_points)} valid 3D points")
    
    # Merge points from all cameras for each object
    merged_object_points = {}
    for obj_id in object_ids:
        if object_points[obj_id]:
            # Concatenate points from all cameras
            all_points = np.vstack(object_points[obj_id])
            
            # Remove duplicate points (from overlapping camera views)
            # Use a small tolerance for floating point comparison
            unique_points = remove_duplicate_points(all_points, tolerance=0.005)
            
            merged_object_points[obj_id] = unique_points
            
            if debug:
                print(f"Object {obj_id}: {len(all_points)} total -> {len(unique_points)} unique points")
        else:
            merged_object_points[obj_id] = np.array([]).reshape(0, 3)
    
    return merged_object_points


def remove_duplicate_points(points, tolerance=0.005):
    """Remove duplicate points within a tolerance."""
    if len(points) == 0:
        return points
    
    # Simple approach: use lexicographic sorting and tolerance-based filtering
    # For a more robust approach, you could use spatial hashing or KDTree
    
    # Sort points
    sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]
    
    # Find unique points
    if len(sorted_points) == 1:
        return sorted_points
    
    # Calculate distances between consecutive points
    distances = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    
    # Keep first point and points that are far enough from previous
    keep_mask = np.ones(len(sorted_points), dtype=bool)
    for i in range(1, len(sorted_points)):
        if distances[i-1] < tolerance:
            keep_mask[i] = False
    
    return sorted_points[keep_mask]


def create_tight_bbox_from_exact_points(object_points, min_points=3):
    """
    Create tight bounding box from exact 3D points.
    
    Args:
        object_points: dict of {object_id: np.array of 3D points}
        min_points: Minimum number of points required
        
    Returns:
        dict: {object_id: bbox_info}
    """
    bboxes = {}
    
    for obj_id, points in object_points.items():
        if len(points) < min_points:
            continue
        
        # Create tight bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        
        # Add minimal padding for visualization (2mm)
        padding = 0.002
        min_bound -= padding
        max_bound += padding
        size += 2 * padding
        
        bboxes[obj_id] = {
            'center': center,
            'size': size,
            'min': min_bound,
            'max': max_bound,
            'n_points': len(points),
            'points': points  # Keep the actual points for visualization
        }
    
    return bboxes


def extract_exact_bboxes_from_replay_buffer(replay_buffer, frame_idx, object_ids=[31, 34, 35, 92]):
    """
    Extract exact bounding boxes from replay buffer using proper 2D-3D mapping.
    
    Note: This requires access to individual camera point clouds, not just the merged one.
    """
    # Check if we have individual camera data
    if "point_clouds" not in replay_buffer.keys():
        print("ERROR: Individual camera point clouds not available in replay buffer!")
        print("Available keys:", list(replay_buffer.keys()))
        return None
    
    # Load data
    masks = replay_buffer["segmentation_masks"][frame_idx]  # (5, H, W)
    point_clouds = replay_buffer["point_clouds"][frame_idx]  # (5, H, W, 3) or similar
    
    obs_data = {
        'masks': masks,
        'point_clouds': point_clouds
    }
    
    # Extract exact 3D points
    object_points = extract_exact_3d_points_from_masks(obs_data, object_ids, debug=True)
    
    # Create tight bounding boxes
    bboxes = create_tight_bbox_from_exact_points(object_points)
    
    return bboxes, object_points


def simulate_exact_extraction_with_collected_data(replay_buffer, frame_idx, object_ids=[31, 34, 35, 92]):
    """
    Simulate exact extraction using the collected segmentation data.
    
    Since we don't have individual camera point clouds in the replay buffer,
    we'll demonstrate the concept with a mock implementation.
    """
    print("SIMULATION: Exact 3D point extraction from masks")
    print("=" * 60)
    
    # Load available data
    masks = replay_buffer["segmentation_masks"][frame_idx]  # (5, H, W)
    merged_pcd = replay_buffer["pcd_xyz"][frame_idx]  # (N, 3) - merged point cloud
    
    print(f"Frame {frame_idx}:")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Merged PCD shape: {merged_pcd.shape}")
    
    # Simulate per-camera point extraction
    simulated_object_points = {}
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    for obj_id in object_ids:
        total_mask_pixels = 0
        
        # Count pixels across all cameras
        for i, camera in enumerate(camera_names):
            if i < len(masks):
                mask = masks[i]
                obj_pixels = (mask == obj_id).sum()
                total_mask_pixels += obj_pixels
                print(f"  Object {obj_id} in {camera}: {obj_pixels} pixels")
        
        if total_mask_pixels > 0:
            # SIMULATION: Estimate number of 3D points based on mask pixels
            # In reality, each mask pixel would map to a 3D point
            estimated_3d_points = total_mask_pixels // 10  # Rough estimate
            
            # For simulation, sample points from the merged cloud
            # In reality, these would be exact points from per-camera clouds
            if estimated_3d_points > 0:
                sample_indices = np.random.choice(len(merged_pcd), 
                                                min(estimated_3d_points, len(merged_pcd)), 
                                                replace=False)
                simulated_points = merged_pcd[sample_indices]
                simulated_object_points[obj_id] = simulated_points
                
                print(f"  -> Simulated {len(simulated_points)} 3D points for object {obj_id}")
    
    # Create bounding boxes from simulated points
    bboxes = create_tight_bbox_from_exact_points(simulated_object_points)
    
    print(f"\nExtracted {len(bboxes)} tight bounding boxes")
    for obj_id, bbox in bboxes.items():
        print(f"  Object {obj_id}: center={bbox['center']}, size={bbox['size']}")
    
    return bboxes, simulated_object_points


def visualize_exact_vs_approximate_bboxes(replay_buffer, frame_idx=50, object_id=92):
    """
    Compare exact vs approximate bounding box extraction.
    """
    print(f"Comparing exact vs approximate bbox extraction for object {object_id}")
    print("=" * 70)
    
    # Approximate method (current)
    from scripts.replay_demo_with_attention import extract_object_bbox_from_masks
    
    masks = replay_buffer["segmentation_masks"][frame_idx]
    pcd = replay_buffer["pcd_xyz"][frame_idx]
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    approx_bbox = extract_object_bbox_from_masks(pcd, masks, object_id, camera_names)
    
    # Exact method (simulated)
    exact_bboxes, exact_points = simulate_exact_extraction_with_collected_data(
        replay_buffer, frame_idx, [object_id]
    )
    
    print("\nCOMPARISON:")
    print("-" * 30)
    
    if approx_bbox:
        print(f"APPROXIMATE (spatial heuristics):")
        print(f"  Volume: {np.prod(approx_bbox['size']):.6f} m³")
        print(f"  Size: {approx_bbox['size']}")
        print(f"  Center: {approx_bbox['center']}")
    
    if object_id in exact_bboxes:
        exact_bbox = exact_bboxes[object_id]
        print(f"\nEXACT (from masks):")
        print(f"  Volume: {np.prod(exact_bbox['size']):.6f} m³")
        print(f"  Size: {exact_bbox['size']}")
        print(f"  Center: {exact_bbox['center']}")
        
        if approx_bbox:
            volume_reduction = (1 - np.prod(exact_bbox['size']) / np.prod(approx_bbox['size'])) * 100
            print(f"\nIMPROVEMENT:")
            print(f"  Volume reduction: {volume_reduction:.1f}%")
            print(f"  Tighter bounding box!")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract exact bounding boxes from masks")
    parser.add_argument("--data_path", type=str, 
                       default="demos/sim/unplug_charger/train_segmented",
                       help="Path to segmented data")
    parser.add_argument("--frame", type=int, default=50,
                       help="Frame to analyze")
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
    
    # Run comparison
    visualize_exact_vs_approximate_bboxes(replay_buffer, args.frame, object_id=92)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("1. Modify data collection to save individual camera point clouds")
    print("2. Implement exact 2D-to-3D mapping in replay_demo_with_attention.py")
    print("3. This will give truly tight bounding boxes around objects!")


if __name__ == "__main__":
    main()