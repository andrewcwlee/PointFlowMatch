from __future__ import annotations
import torch
import numpy as np
import pypose as pp
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.se3_utils import transform_th
from pfp import DATA_DIRS
# Import exact mask extraction functions
try:
    from scripts.exact_bbox_extraction_for_replay import extract_exact_object_points_from_masks
except ImportError:
    # Fallback - implement locally
    extract_exact_object_points_from_masks = None


def rand_range(low: float, high: float, size: tuple[int], device) -> torch.Tensor:
    return torch.rand(size, device=device) * (high - low) + low


def augment_pcd_data(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Augment point cloud data with random SE3 transformations."""
    if len(batch) == 3:
        pcd, robot_state_obs, robot_state_pred = batch
        attention_mask = None
    else:
        pcd, robot_state_obs, robot_state_pred, attention_mask = batch
        
    BT_robot_obs = robot_state_obs.shape[:-1]
    BT_robot_pred = robot_state_pred.shape[:-1]

    # sigma=(sigma_transl, sigma_rot_rad)
    transform = pp.randn_SE3(sigma=(0.1, 0.2), device=pcd.device).matrix()

    pcd[..., :3] = transform_th(transform, pcd[..., :3])
    robot_obs_pseudoposes = robot_state_obs[..., :9].reshape(*BT_robot_obs, 3, 3)
    robot_pred_pseudoposes = robot_state_pred[..., :9].reshape(*BT_robot_pred, 3, 3)
    robot_obs_pseudoposes = transform_th(transform, robot_obs_pseudoposes)
    robot_pred_pseudoposes = transform_th(transform, robot_pred_pseudoposes)
    robot_state_obs[..., :9] = robot_obs_pseudoposes.reshape(*BT_robot_obs, 9)
    robot_state_pred[..., :9] = robot_pred_pseudoposes.reshape(*BT_robot_pred, 9)

    # We shuffle the points, i.e. shuffle pcd along dim=2 (B, T, P, 3)
    idx = torch.randperm(pcd.shape[2])
    pcd = pcd[:, :, idx, :]
    
    # Also shuffle attention mask if present
    if attention_mask is not None:
        attention_mask = attention_mask[:, :, idx]
        return pcd, robot_state_obs, robot_state_pred, attention_mask
    
    return pcd, robot_state_obs, robot_state_pred


def extract_exact_object_points_from_masks_local(camera_point_clouds, masks, object_ids, debug=False):
    """
    Local implementation of exact object point extraction.
    Extract exact 3D points for objects using 2D masks and per-camera point clouds.
    """
    camera_names = ['right_shoulder', 'left_shoulder', 'overhead', 'front', 'wrist']
    
    # Storage for exact 3D points per object
    object_points = {obj_id: [] for obj_id in object_ids}
    
    for camera_idx in range(len(camera_names)):
        if camera_idx >= len(masks) or camera_idx >= len(camera_point_clouds):
            continue
            
        camera_name = camera_names[camera_idx]
        mask = masks[camera_idx]  # (H, W)
        point_cloud = camera_point_clouds[camera_idx]  # (H*W, 3) or (H, W, 3)
        
        H, W = mask.shape
        
        # Reshape point cloud to match flattened mask
        if len(point_cloud.shape) == 3:  # (H, W, 3)
            if point_cloud.shape[:2] != (H, W):
                if debug:
                    print(f"  WARNING: Point cloud shape {point_cloud.shape} doesn't match mask shape {(H, W)}")
                continue
            point_cloud = point_cloud.reshape(-1, 3)  # (H*W, 3)
        
        # Flatten mask to match point cloud
        mask_flat = mask.flatten()  # (H*W,)
        
        # Extract points for each object
        for obj_id in object_ids:
            # Find pixels where this object appears
            object_mask = (mask_flat == obj_id)
            n_pixels = object_mask.sum()
            
            if n_pixels > 0:
                # Get corresponding 3D points
                object_3d_points = point_cloud[object_mask]
                
                # Filter out invalid points
                valid_mask = (
                    np.isfinite(object_3d_points).all(axis=1) &  # No NaN/inf
                    (np.linalg.norm(object_3d_points, axis=1) > 0.01) &  # Not at origin
                    (np.linalg.norm(object_3d_points, axis=1) < 10.0)    # Not too far
                )
                
                valid_points = object_3d_points[valid_mask]
                
                if len(valid_points) > 0:
                    object_points[obj_id].append(valid_points)
    
    # Merge points from all cameras for each object
    merged_object_points = {}
    for obj_id in object_ids:
        if object_points[obj_id]:
            # Concatenate points from all cameras
            all_points = np.vstack(object_points[obj_id])
            merged_object_points[obj_id] = all_points
        else:
            merged_object_points[obj_id] = np.array([]).reshape(0, 3)
    
    return merged_object_points


def create_attention_mask_from_object_points(pcd: np.ndarray, object_points: dict, 
                                            gripper_pos: np.ndarray = None, 
                                            distance_threshold: float = 0.01,
                                            gripper_radius: float = 0.05):
    """
    Create attention mask where points belonging to objects (or near them) have weight 1.
    
    Args:
        pcd: Point cloud (N, 3)
        object_points: Dict of {object_id: exact 3D points array}
        gripper_pos: Optional gripper position to include
        distance_threshold: Distance threshold for considering a point as belonging to an object
        gripper_radius: Radius around gripper to include
        
    Returns:
        attention_mask: (N,) array with 1 for object points, 0 otherwise
    """
    attention_mask = np.zeros(len(pcd), dtype=np.float32)
    
    # Combine all object points into one array for efficient distance computation
    all_object_points = []
    for obj_id, obj_pts in object_points.items():
        if len(obj_pts) > 0:
            all_object_points.append(obj_pts)
    
    if len(all_object_points) > 0:
        all_object_points = np.vstack(all_object_points)
        
        # Vectorized distance computation using broadcasting
        # For each point in pcd, find minimum distance to any object point
        # Process in batches to manage memory
        for i in range(0, len(pcd), 100):  # Process 100 points at a time
            batch_end = min(i + 100, len(pcd))
            batch_pcd = pcd[i:batch_end]
            
            # Compute distances from batch points to all object points
            # Shape: (batch_size, num_object_points)
            distances = np.linalg.norm(
                batch_pcd[:, None, :] - all_object_points[None, :, :], 
                axis=2
            )
            
            # Find minimum distance for each point in batch
            min_distances = distances.min(axis=1)
            
            # Set attention to 1 for points close to objects
            mask = min_distances < distance_threshold
            attention_mask[i:batch_end][mask] = 1.0
    
    # Include gripper region if provided
    if gripper_pos is not None:
        # Add region around gripper
        gripper_distances = np.linalg.norm(pcd - gripper_pos[None, :], axis=1)
        gripper_mask = gripper_distances < gripper_radius
        attention_mask[gripper_mask] = 1.0
    
    return attention_mask


def points_in_bbox(points: np.ndarray, bbox_center: np.ndarray, bbox_extents: np.ndarray, 
                   bbox_rotation: np.ndarray = None) -> np.ndarray:
    """
    Check if points are inside a 3D bounding box.
    
    Args:
        points: Point cloud of shape (N, 3)
        bbox_center: Center of bounding box (3,)
        bbox_extents: Half-extents of bounding box in each dimension (3,)
        bbox_rotation: Optional rotation matrix (3, 3)
    
    Returns:
        Boolean mask of shape (N,) indicating which points are inside the bbox
    """
    # Translate points to bbox center
    points_centered = points - bbox_center
    
    # Apply inverse rotation if provided
    if bbox_rotation is not None:
        points_centered = points_centered @ bbox_rotation.T
    
    # Check if points are within bbox extents
    inside = np.all(np.abs(points_centered) <= bbox_extents, axis=1)
    return inside


class RobotDatasetPcdAttention(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        use_pc_color: bool,
        n_points: int,
        subs_factor: int = 1,  # 1 means no subsampling
        use_bounding_box: bool = True,
        bbox_mode: str = "gripper",  # "gripper", "custom", "segmentation"
        bbox_padding: float = 0.1,  # Extra padding around bounding boxes
        object_ids: list = None,  # Object IDs for segmentation mode
        distance_threshold: float = 0.01,  # Distance threshold for segmentation mode
        gripper_radius: float = 0.05,  # Radius around gripper for attention
    ) -> None:
        """
        Dataset with attention supervision support.
        
        Args:
            bbox_mode: How to generate attention masks
                - "gripper": Box around gripper only
                - "custom": Load custom bounding boxes from data
                - "segmentation": Use exact segmentation masks (requires segmented data)
            bbox_padding: Extra padding for bounding boxes (gripper mode)
            object_ids: List of object IDs to track (segmentation mode)
            distance_threshold: Distance threshold for point-to-object matching (segmentation mode)
            gripper_radius: Radius around gripper for attention (segmentation mode)
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "pcd_xyz"]
        data_key_first_k = {"pcd_xyz": n_obs_steps * subs_factor}
        
        if use_pc_color:
            data_keys.append("pcd_color")
            data_key_first_k["pcd_color"] = n_obs_steps * subs_factor
            
        # Check if bounding box data exists in the replay buffer
        self.has_bbox_data = False
        self.has_segmentation_data = False
        
        if use_bounding_box:
            if bbox_mode == "segmentation":
                # Check for segmentation data
                if "segmentation_masks" in replay_buffer.keys() and "camera_point_clouds" in replay_buffer.keys():
                    data_keys.extend(["segmentation_masks", "camera_point_clouds"])
                    data_key_first_k["segmentation_masks"] = n_obs_steps * subs_factor
                    data_key_first_k["camera_point_clouds"] = n_obs_steps * subs_factor
                    self.has_segmentation_data = True
                else:
                    print(f"Warning: Segmentation mode requested but no segmentation data found in {data_path}")
                    print("Falling back to gripper mode")
                    bbox_mode = "gripper"
            elif bbox_mode == "custom" and "bbox_center" in replay_buffer.keys():
                data_keys.extend(["bbox_center", "bbox_extents"])
                data_key_first_k["bbox_center"] = n_obs_steps * subs_factor
                data_key_first_k["bbox_extents"] = n_obs_steps * subs_factor
                self.has_bbox_data = True
                
                # Optional: rotation data
                if "bbox_rotation" in replay_buffer.keys():
                    data_keys.append("bbox_rotation")
                    data_key_first_k["bbox_rotation"] = n_obs_steps * subs_factor
                
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=(n_obs_steps + n_pred_steps) * subs_factor - (subs_factor - 1),
            pad_before=(n_obs_steps - 1) * subs_factor,
            pad_after=(n_pred_steps - 1) * subs_factor + (subs_factor - 1),
            keys=data_keys,
            key_first_k=data_key_first_k,
        )
        self.n_obs_steps = n_obs_steps
        self.n_prediction_steps = n_pred_steps
        self.subs_factor = subs_factor
        self.use_pc_color = use_pc_color
        self.n_points = n_points
        self.use_bounding_box = use_bounding_box
        self.bbox_mode = bbox_mode
        self.bbox_padding = bbox_padding
        self.object_ids = object_ids if object_ids is not None else [31, 34, 35, 92]  # Default IDs
        self.distance_threshold = distance_threshold
        self.gripper_radius = gripper_radius
        self.rng = np.random.default_rng()
        
        # Use appropriate extraction function
        if extract_exact_object_points_from_masks is not None:
            self.extract_func = extract_exact_object_points_from_masks
        else:
            self.extract_func = extract_exact_object_points_from_masks_local
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def generate_attention_mask_from_segmentation(self, pcd: np.ndarray, robot_state: np.ndarray,
                                                 segmentation_masks: np.ndarray, 
                                                 camera_point_clouds: np.ndarray) -> np.ndarray:
        """
        Generate attention mask from exact segmentation masks.
        
        Args:
            pcd: Point cloud (T, N, 3)
            robot_state: Robot state including gripper position (T, D)
            segmentation_masks: Segmentation masks (T, 5, H, W)
            camera_point_clouds: Individual camera point clouds (T, 5, H*W, 3)
        
        Returns:
            Attention mask (T, N)
        """
        T, N, _ = pcd.shape
        attention_mask = np.zeros((T, N), dtype=np.float32)
        
        for t in range(T):
            # Extract gripper position
            gripper_pos = robot_state[t, :3]
            
            # Extract exact object points from masks
            object_points = self.extract_func(
                camera_point_clouds[t], 
                segmentation_masks[t], 
                self.object_ids, 
                debug=(t == 0)  # Debug first timestep
            )
            
            # Create attention mask from object points
            attention_mask[t] = create_attention_mask_from_object_points(
                pcd[t], 
                object_points, 
                gripper_pos, 
                self.distance_threshold,
                self.gripper_radius
            )
            
        return attention_mask
    
    def generate_attention_mask_from_gripper(self, pcd: np.ndarray, robot_state: np.ndarray) -> np.ndarray:
        """
        Generate attention mask based on gripper position.
        
        Args:
            pcd: Point cloud (T, N, 3)
            robot_state: Robot state including gripper position (T, D)
        
        Returns:
            Attention mask (T, N)
        """
        T, N, _ = pcd.shape
        attention_mask = np.zeros((T, N), dtype=np.float32)
        
        for t in range(T):
            # Extract gripper position from robot state (first 3 values)
            gripper_pos = robot_state[t, :3]
            
            # Simple sphere around gripper
            bbox_center = gripper_pos
            bbox_extents = np.array([self.bbox_padding] * 3)
            
            # Check which points are inside
            inside = points_in_bbox(pcd[t], bbox_center, bbox_extents)
            attention_mask[t] = inside.astype(np.float32)
            
        return attention_mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        pcd = sample["pcd_xyz"][: cur_step_i : self.subs_factor]
        
        if self.use_pc_color:
            pcd_color = sample["pcd_color"][: cur_step_i : self.subs_factor]
            pcd_color = pcd_color.astype(np.float32) / 255.0
            pcd = np.concatenate([pcd, pcd_color], axis=-1)
            
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor].astype(np.float32)
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor].astype(np.float32)
        
        # Generate or load attention mask
        if self.use_bounding_box:
            if self.bbox_mode == "segmentation" and self.has_segmentation_data:
                # Use exact segmentation masks
                segmentation_masks = sample["segmentation_masks"][: cur_step_i : self.subs_factor]
                camera_point_clouds = sample["camera_point_clouds"][: cur_step_i : self.subs_factor]
                attention_mask = self.generate_attention_mask_from_segmentation(
                    pcd[..., :3], robot_state_obs, segmentation_masks, camera_point_clouds
                )
            elif self.bbox_mode == "custom" and self.has_bbox_data:
                # Use provided bounding box data
                bbox_centers = sample["bbox_center"][: cur_step_i : self.subs_factor]
                bbox_extents = sample["bbox_extents"][: cur_step_i : self.subs_factor] + self.bbox_padding
                bbox_rotations = sample.get("bbox_rotation", None)
                if bbox_rotations is not None:
                    bbox_rotations = bbox_rotations[: cur_step_i : self.subs_factor]
                
                # Generate attention mask from bounding boxes
                T, N, _ = pcd.shape[:3]
                attention_mask = np.zeros((T, N), dtype=np.float32)
                
                for t in range(T):
                    rotation = bbox_rotations[t] if bbox_rotations is not None else None
                    inside = points_in_bbox(pcd[t, :, :3], bbox_centers[t], bbox_extents[t], rotation)
                    attention_mask[t] = inside.astype(np.float32)
                    
            elif self.bbox_mode == "gripper":
                # Generate attention mask from gripper position
                attention_mask = self.generate_attention_mask_from_gripper(pcd[..., :3], robot_state_obs)
            else:
                # Fallback: no specific attention, create uniform mask
                T, N = pcd.shape[:2]
                attention_mask = np.ones((T, N), dtype=np.float32)
        else:
            # No attention mask needed
            attention_mask = None
        
        # Random sample pcd points
        if pcd.shape[1] > self.n_points:
            random_indices = np.random.choice(pcd.shape[1], self.n_points, replace=False)
            pcd = pcd[:, random_indices]
            if attention_mask is not None:
                attention_mask = attention_mask[:, random_indices]
                
        if attention_mask is not None:
            return pcd, robot_state_obs, robot_state_pred, attention_mask
        else:
            return pcd, robot_state_obs, robot_state_pred


if __name__ == "__main__":
    # Test segmentation mode
    dataset = RobotDatasetPcdAttention(
        data_path=DATA_DIRS.PFP / "unplug_charger" / "train_segmented",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
        use_pc_color=False,
        n_points=4096,
        use_bounding_box=True,
        bbox_mode="segmentation",
        object_ids=[31, 34, 35, 92],
        distance_threshold=0.01,
    )
    i = 20
    result = dataset[i]
    if len(result) == 4:
        obs, robot_state_obs, robot_state_pred, attention_mask = result
        print("attention_mask shape: ", attention_mask.shape)
        print("attention_mask mean: ", attention_mask.mean())
    else:
        obs, robot_state_obs, robot_state_pred = result
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")