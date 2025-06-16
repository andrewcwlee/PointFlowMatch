#!/usr/bin/env python3
"""
Analyze segmentation masks in RLBench to identify objects in the scene.
Based on: https://github.com/stepjam/RLBench/issues/72
"""

import numpy as np
from pathlib import Path
from collections import defaultdict
from pfp.envs.rlbench_env import RLBenchEnv
from rlbench.utils import name_to_task_class


def get_object_names_from_segmentation(env: RLBenchEnv, task_name: str):
    """
    Extract object names from segmentation masks.
    
    Returns:
        dict: Mapping of object IDs to object names
    """
    # Reset environment to get a fresh scene
    env.task.reset()
    
    # Get observation with masks
    obs = env.task.get_observation()
    
    # Get the task object to access scene info
    task_class = name_to_task_class(task_name)
    
    # Dictionary to store object ID to name mapping
    object_id_to_name = {}
    
    # Get all objects in the scene through PyRep
    scene_objects = env.task._scene._active_task.get_base().get_objects_in_tree()
    
    for obj in scene_objects:
        try:
            # Get object handle (ID)
            obj_handle = obj.get_handle()
            obj_name = obj.get_name()
            object_id_to_name[obj_handle] = obj_name
        except:
            continue
    
    return object_id_to_name


def analyze_segmentation_masks(env: RLBenchEnv):
    """
    Analyze segmentation masks from all cameras and identify unique objects.
    """
    # Get observation
    obs = env.task.get_observation()
    
    # Camera names and their masks
    camera_masks = {
        'right_shoulder': obs.right_shoulder_mask,
        'left_shoulder': obs.left_shoulder_mask,
        'overhead': obs.overhead_mask,
        'front': obs.front_mask,
        'wrist': obs.wrist_mask
    }
    
    # Collect unique object IDs from all cameras
    all_object_ids = set()
    camera_object_counts = {}
    
    for camera_name, mask in camera_masks.items():
        if mask is not None:
            unique_ids = np.unique(mask)
            # Filter out background (usually 0 or -1)
            unique_ids = unique_ids[unique_ids > 0]
            
            all_object_ids.update(unique_ids)
            camera_object_counts[camera_name] = len(unique_ids)
            
            print(f"\n{camera_name} camera:")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Unique object IDs: {unique_ids}")
            print(f"  Number of objects: {len(unique_ids)}")
    
    print(f"\nTotal unique objects across all cameras: {len(all_object_ids)}")
    print(f"All object IDs: {sorted(list(all_object_ids))}")
    
    return all_object_ids, camera_masks


def mask_point_cloud_by_object(obs, object_id: int, camera: str = 'front'):
    """
    Extract point cloud for a specific object using segmentation mask.
    
    Args:
        obs: RLBench observation
        object_id: ID of the object to extract
        camera: Which camera to use
        
    Returns:
        numpy array of points belonging to the object
    """
    # Get mask and point cloud for the specified camera
    if camera == 'front':
        mask = obs.front_mask
        point_cloud = obs.front_point_cloud
    elif camera == 'left_shoulder':
        mask = obs.left_shoulder_mask
        point_cloud = obs.left_shoulder_point_cloud
    elif camera == 'right_shoulder':
        mask = obs.right_shoulder_mask
        point_cloud = obs.right_shoulder_point_cloud
    elif camera == 'overhead':
        mask = obs.overhead_mask
        point_cloud = obs.overhead_point_cloud
    elif camera == 'wrist':
        mask = obs.wrist_mask
        point_cloud = obs.wrist_point_cloud
    else:
        raise ValueError(f"Unknown camera: {camera}")
    
    # Create boolean mask for the specific object
    object_mask = (mask == object_id)
    
    # Extract points belonging to this object
    # Point cloud shape is (H, W, 3), mask is (H, W)
    object_points = point_cloud[object_mask]
    
    return object_points, object_mask


def visualize_segmented_objects(env: RLBenchEnv, task_name: str):
    """
    Visualize each segmented object separately.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Get observation
    obs = env.task.get_observation()
    
    # Analyze segmentation
    all_object_ids, camera_masks = analyze_segmentation_masks(env)
    
    # Try to get object names
    try:
        object_id_to_name = get_object_names_from_segmentation(env, task_name)
        print("\nObject ID to Name mapping:")
        for obj_id, name in object_id_to_name.items():
            if obj_id in all_object_ids:
                print(f"  ID {obj_id}: {name}")
    except Exception as e:
        print(f"\nCould not get object names: {e}")
        object_id_to_name = {}
    
    # Visualize each object separately
    fig = plt.figure(figsize=(20, 10))
    
    # Use front camera for visualization
    n_objects = min(6, len(all_object_ids))  # Limit to 6 objects for visualization
    
    for idx, obj_id in enumerate(sorted(list(all_object_ids))[:n_objects]):
        ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
        
        # Get points for this object from front camera
        object_points, _ = mask_point_cloud_by_object(obs, obj_id, 'front')
        
        if len(object_points) > 0:
            # Downsample if too many points
            if len(object_points) > 1000:
                indices = np.random.choice(len(object_points), 1000, replace=False)
                object_points = object_points[indices]
            
            # Plot points
            ax.scatter(object_points[:, 0], object_points[:, 1], object_points[:, 2], 
                      s=1, alpha=0.6)
            
            # Set title with object name if available
            obj_name = object_id_to_name.get(obj_id, f"Unknown (ID: {obj_id})")
            ax.set_title(f"Object: {obj_name}\nPoints: {len(object_points)}")
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set equal aspect ratio
            ax.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig("segmented_objects.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return all_object_ids, object_id_to_name


def print_task_objects(task_name: str):
    """
    Print all objects specific to a task.
    """
    print(f"\nAnalyzing task: {task_name}")
    print("="*50)
    
    # Create environment
    env = RLBenchEnv(
        task_name=task_name,
        voxel_size=0.005,
        n_points=4096,
        use_pc_color=True,
        headless=True,
        vis=False,
    )
    
    # Get task-specific objects
    task = env.task
    
    # Common task objects based on RLBench structure
    print("\nTask-specific objects:")
    
    # Try to access common task attributes
    task_attrs = ['success_sensor', 'target', 'robot', 'gripper', 'tip']
    
    for attr in dir(task):
        if not attr.startswith('_') and hasattr(task, attr):
            obj = getattr(task, attr)
            if hasattr(obj, 'get_name'):
                try:
                    print(f"  {attr}: {obj.get_name()}")
                except:
                    pass
    
    # Analyze segmentation
    all_object_ids, object_id_to_name = visualize_segmented_objects(env, task_name)
    
    # For unplug_charger specifically
    if task_name == "unplug_charger":
        print("\nUnplug Charger specific objects:")
        print("  - Charger (the plug to be unplugged)")
        print("  - Socket/Outlet (where the charger is plugged)")
        print("  - Robot gripper")
        print("  - Table/Surface")
    
    return env, all_object_ids, object_id_to_name


def main():
    """
    Main function to analyze segmentation for different tasks.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze RLBench segmentation masks")
    parser.add_argument("--task", type=str, default="unplug_charger",
                       help="Task name to analyze")
    parser.add_argument("--save_dir", type=str, default="./segmentation_analysis",
                       help="Directory to save analysis results")
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Analyze the task
    env, object_ids, object_names = print_task_objects(args.task)
    
    # Save results
    results_file = save_dir / f"{args.task}_objects.txt"
    with open(results_file, 'w') as f:
        f.write(f"Task: {args.task}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Object IDs found in segmentation:\n")
        for obj_id in sorted(object_ids):
            obj_name = object_names.get(obj_id, "Unknown")
            f.write(f"  ID {obj_id}: {obj_name}\n")
        
        f.write(f"\nTotal objects: {len(object_ids)}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Close environment
    env.env.shutdown()


if __name__ == "__main__":
    main()