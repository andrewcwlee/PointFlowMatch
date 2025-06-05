#!/usr/bin/env python3
"""
Exact bounding box extraction functions for use in replay_demo_with_attention.py
"""

import numpy as np


def extract_exact_object_points_from_masks(camera_point_clouds, masks, object_ids, debug=False):
    """
    Extract exact 3D points for objects using 2D masks and per-camera point clouds.
    
    Args:
        camera_point_clouds: np.array shape (5, H*W, 3) - individual camera point clouds
        masks: np.array shape (5, H, W) - segmentation masks
        object_ids: list of object IDs to extract
        
    Returns:
        dict: {object_id: np.array of merged 3D points from all cameras}
    """
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    # Storage for exact 3D points per object
    object_points = {obj_id: [] for obj_id in object_ids}
    
    for camera_idx in range(len(camera_names)):
        if camera_idx >= len(masks) or camera_idx >= len(camera_point_clouds):
            continue
            
        camera_name = camera_names[camera_idx]
        mask = masks[camera_idx]  # (H, W)
        point_cloud = camera_point_clouds[camera_idx]  # Could be (H, W, 3) or (H*W, 3)
        
        H, W = mask.shape
        
        # Reshape point cloud to match flattened mask
        if len(point_cloud.shape) == 3:  # (H, W, 3)
            if point_cloud.shape[:2] != (H, W):
                if debug:
                    print(f"  WARNING: Point cloud shape {point_cloud.shape} doesn't match mask shape {(H, W)}")
                continue
            point_cloud = point_cloud.reshape(-1, 3)  # (H*W, 3)
        else:  # Should be (H*W, 3)
            expected_size = H * W
            if point_cloud.shape[0] != expected_size:
                if debug:
                    print(f"  WARNING: Point cloud size {point_cloud.shape[0]} doesn't match expected {expected_size}")
                continue
        
        # Flatten mask to match point cloud
        mask_flat = mask.flatten()  # (H*W,)
        
        if debug and camera_idx == 0:  # Print debug info for first camera only
            print(f"Camera {camera_name}: mask {mask.shape}, pcd {point_cloud.shape}")
        
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
                        print(f"  {camera_name} - Object {obj_id}: {n_pixels} pixels -> {len(valid_points)} valid 3D points")
    
    # Merge points from all cameras for each object
    merged_object_points = {}
    for obj_id in object_ids:
        if object_points[obj_id]:
            # Concatenate points from all cameras
            all_points = np.vstack(object_points[obj_id])
            
            # Remove duplicate points (from overlapping camera views)
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
    
    # Sort points for consistent processing
    sorted_indices = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    sorted_points = points[sorted_indices]
    
    if len(sorted_points) <= 1:
        return sorted_points
    
    # Calculate distances between consecutive points
    distances = np.linalg.norm(np.diff(sorted_points, axis=0), axis=1)
    
    # Keep first point and points that are far enough from previous
    keep_mask = np.ones(len(sorted_points), dtype=bool)
    for i in range(1, len(sorted_points)):
        if distances[i-1] < tolerance:
            keep_mask[i] = False
    
    return sorted_points[keep_mask]


def create_exact_bboxes_from_object_points(object_points, gripper_pos=None, include_gripper=True):
    """
    Create exact bounding boxes from object points, optionally including gripper.
    
    Args:
        object_points: dict {object_id: np.array of 3D points}
        gripper_pos: np.array (3,) gripper position
        include_gripper: bool, whether to include gripper in combined bbox
        
    Returns:
        dict: individual bboxes and combined bbox
    """
    individual_bboxes = {}
    all_key_points = []
    
    # Create bboxes for individual objects
    for obj_id, points in object_points.items():
        if len(points) < 3:
            continue
        
        # Create tight bounding box
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        center = (min_bound + max_bound) / 2
        size = max_bound - min_bound
        
        # Add minimal padding (2mm) for visualization
        padding = 0.002
        min_bound_padded = min_bound - padding
        max_bound_padded = max_bound + padding
        size_padded = size + 2 * padding
        
        individual_bboxes[obj_id] = {
            'center': center,
            'size': size_padded,
            'min': min_bound_padded,
            'max': max_bound_padded,
            'n_points': len(points),
            'exact_points': points
        }
        
        # Collect points for combined bbox
        all_key_points.extend([min_bound, max_bound])
    
    # Add gripper to combined bbox if requested
    if include_gripper and gripper_pos is not None:
        all_key_points.append(gripper_pos)
    
    # Create combined bounding box
    combined_bbox = None
    if len(all_key_points) > 0:
        all_points = np.array(all_key_points)
        combined_min = all_points.min(axis=0)
        combined_max = all_points.max(axis=0)
        combined_center = (combined_min + combined_max) / 2
        combined_size = combined_max - combined_min
        
        # Add small padding to combined bbox
        padding = 0.005
        combined_min -= padding
        combined_max += padding
        combined_size += 2 * padding
        
        combined_bbox = {
            'center': combined_center,
            'size': combined_size,
            'min': combined_min,
            'max': combined_max,
            'includes_gripper': include_gripper
        }
    
    return individual_bboxes, combined_bbox


def extract_exact_bboxes_from_replay_data(camera_point_clouds, masks, gripper_pos, 
                                         object_ids=[31, 34, 35, 92], debug=False):
    """
    Main function to extract exact bounding boxes from replay data.
    
    Args:
        camera_point_clouds: np.array (5, H*W, 3)
        masks: np.array (5, H, W)
        gripper_pos: np.array (3,)
        object_ids: list of object IDs
        
    Returns:
        tuple: (individual_bboxes, combined_bbox, object_points)
    """
    if debug:
        print(f"Extracting exact bboxes for objects {object_ids}")
    
    # Extract exact 3D points for each object
    object_points = extract_exact_object_points_from_masks(
        camera_point_clouds, masks, object_ids, debug=debug
    )
    
    # Create bounding boxes
    individual_bboxes, combined_bbox = create_exact_bboxes_from_object_points(
        object_points, gripper_pos, include_gripper=True
    )
    
    if debug:
        print(f"Extracted {len(individual_bboxes)} individual bboxes")
        if combined_bbox:
            print(f"Combined bbox: center={combined_bbox['center']}, size={combined_bbox['size']}")
    
    return individual_bboxes, combined_bbox, object_points


# Example usage:
if __name__ == "__main__":
    print("This module provides exact bbox extraction functions.")
    print("Import these functions into replay_demo_with_attention.py:")
    print("- extract_exact_bboxes_from_replay_data()")
    print("- extract_exact_object_points_from_masks()")
    print("- create_exact_bboxes_from_object_points()")