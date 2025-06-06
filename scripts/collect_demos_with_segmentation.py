import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from rlbench.backend.observation import Observation
from pfp import DATA_DIRS, set_seeds
from pfp.envs.rlbench_env import RLBenchEnv
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.visualization import RerunViewer as RV


def identify_charger_and_gripper(masks, task_name="unplug_charger"):
    """
    Identify which object IDs correspond to charger and gripper.
    
    Returns:
        dict: {'charger_id': int, 'gripper_ids': list[int]}
    """
    # Collect all unique object IDs from all cameras
    all_object_ids = set()
    for mask in masks.values():
        if mask is not None:
            unique_ids = np.unique(mask)
            unique_ids = unique_ids[unique_ids > 0]  # Remove background
            all_object_ids.update(unique_ids)
    
    
    print(f"Found {len(all_object_ids)} unique objects")
    print(f"Object IDs: {sorted(list(all_object_ids))}")
    
    # Return all IDs for now - user can identify later
    return {
        'all_ids': [int(obj_id) for obj_id in sorted(list(all_object_ids))],
        'charger_id': None,  # To be identified
        'gripper_ids': None  # To be identified
    }


def extract_bounding_box_from_mask(point_cloud, mask, object_id):
    """
    Extract tight bounding box for a specific object from its mask.
    
    Returns:
        dict: {'center': [x,y,z], 'extents': [dx,dy,dz], 'min': [x,y,z], 'max': [x,y,z]}
    """
    # Get mask for specific object
    obj_mask = (mask == object_id)
    
    if not np.any(obj_mask):
        return None
    
    # Extract points belonging to this object
    obj_points = point_cloud[obj_mask]
    
    if len(obj_points) == 0:
        return None
    
    # Compute tight bounding box
    min_bound = obj_points.min(axis=0)
    max_bound = obj_points.max(axis=0)
    center = (min_bound + max_bound) / 2
    extents = (max_bound - min_bound) / 2
    
    return {
        'center': center.astype(np.float32),
        'extents': extents.astype(np.float32),
        'min': min_bound.astype(np.float32),
        'max': max_bound.astype(np.float32),
        'n_points': len(obj_points)
    }


@hydra.main(version_base=None, config_path="../conf", config_name="collect_demos_train")
def main(cfg: OmegaConf):
    """
    Collect demos with full segmentation masks saved separately.
    """
    set_seeds(cfg.seed)
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    print(OmegaConf.to_yaml(cfg))

    assert cfg.mode in ["train", "valid"]
    if cfg.env_config.vis:
        RV("pfp_collect_demos")
    
    # Create environment
    env = RLBenchEnv(use_pc_color=True, **cfg.env_config)
    
    # Identify objects once at the beginning
    env.task.reset()
    obs = env.task.get_observation()
    
    masks = {
        'right_shoulder': obs.right_shoulder_mask,
        'left_shoulder': obs.left_shoulder_mask,
        'overhead': obs.overhead_mask,
        'front': obs.front_mask,
        'wrist': obs.wrist_mask
    }
    
    object_info = identify_charger_and_gripper(masks, cfg.env_config.task_name)
    
    # Setup data saving
    if cfg.save_data:
        # Create a special directory for segmentation data
        data_path = DATA_DIRS.PFP / cfg.env_config.task_name / f"{cfg.mode}_segmented"
        if data_path.is_dir():
            print(f"ERROR: Data path {data_path} already exists! Exiting...")
            return
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="a")
        
        # Save object info as metadata
        import json
        metadata_path = data_path / "object_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(object_info, f, indent=2)
        print(f"Saved object info to {metadata_path}")

    # Collect episodes
    for episode_idx in tqdm(range(cfg.num_episodes)):
        data_history = list()
        demo = env.task.get_demos(1, live_demos=True)[0]
        observations: list[Observation] = demo._observations
        
        for step_idx, obs in enumerate(observations):
            robot_state = env.get_robot_state(obs)
            images = env.get_images(obs)
            pcd = env.get_pcd(obs)
            pcd_xyz = np.asarray(pcd.points)
            pcd_color = np.asarray(pcd.colors)
            
            # Get all masks
            masks_dict = {
                'right_shoulder': obs.right_shoulder_mask,
                'left_shoulder': obs.left_shoulder_mask,
                'overhead': obs.overhead_mask,
                'front': obs.front_mask,
                'wrist': obs.wrist_mask
            }
            
            # Get point clouds for bounding box extraction
            point_clouds_dict = {
                'right_shoulder': obs.right_shoulder_point_cloud,
                'left_shoulder': obs.left_shoulder_point_cloud,
                'overhead': obs.overhead_point_cloud,
                'front': obs.front_point_cloud,
                'wrist': obs.wrist_point_cloud
            }
            
            # For first few steps of first episode, analyze objects in detail
            if episode_idx == 0 and step_idx < 5:
                print(f"\nEpisode {episode_idx}, Step {step_idx}:")
                
                # Use front camera for analysis (usually has best view)
                front_mask = masks_dict['front']
                front_pc = point_clouds_dict['front']
                
                for obj_id in object_info['all_ids'][:10]:  # First 10 objects
                    bbox = extract_bounding_box_from_mask(front_pc, front_mask, obj_id)
                    if bbox:
                        print(f"  Object {obj_id}: center={bbox['center']}, size={2*bbox['extents']}, points={bbox['n_points']}")
            
            # Prepare data for saving
            # Convert masks to uint16 to handle object IDs > 255
            masks_array = []
            for camera in ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']:
                mask = masks_dict[camera]
                if mask is not None:
                    masks_array.append(mask.astype(np.uint16))
                else:
                    # Create dummy mask
                    masks_array.append(np.zeros((128, 128), dtype=np.uint16))
            
            masks_stacked = np.stack(masks_array, axis=0)  # (5, H, W)
            
            # Prepare individual camera point clouds for exact bbox extraction
            point_clouds_array = []
            for camera in ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']:
                pc = point_clouds_dict[camera]
                if pc is not None:
                    point_clouds_array.append(pc.astype(np.float32))
                else:
                    # Create dummy point cloud
                    point_clouds_array.append(np.zeros((128*128, 3), dtype=np.float32))
            
            point_clouds_stacked = np.stack(point_clouds_array, axis=0)  # (5, H*W, 3)
            
            data_point = {
                "pcd_xyz": pcd_xyz.astype(np.float32),
                "pcd_color": (pcd_color * 255).astype(np.uint8),
                "robot_state": robot_state.astype(np.float32),
                "images": images,
                "segmentation_masks": masks_stacked,  # (5, H, W) with camera order as above
                "camera_point_clouds": point_clouds_stacked,  # (5, H*W, 3) individual camera PCDs
            }
            
            data_history.append(data_point)
            env.vis_step(robot_state, np.concatenate((pcd_xyz, pcd_color), axis=-1))

        if cfg.save_data:
            replay_buffer.add_episode_from_list(data_history, compressors="disk")
            print(f"Saved episode {episode_idx} with {len(data_history)} steps")

    print("\n" + "="*60)
    print("Data collection complete!")
    print(f"Saved to: {data_path}")
    print("\nNext steps:")
    print("1. Run analyze_collected_masks.py to identify charger and gripper IDs")
    print("2. Update object_info.json with the correct IDs")
    print("3. Use the segmentation masks to create tight bounding boxes")
    
    return


if __name__ == "__main__":
    main()