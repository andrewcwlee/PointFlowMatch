from __future__ import annotations
import torch
import numpy as np
import pypose as pp
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.se3_utils import transform_th
from pfp import DATA_DIRS
import os


def rand_range(low: float, high: float, size: tuple[int], device) -> torch.Tensor:
    return torch.rand(size, device=device) * (high - low) + low


def augment_pcd_data(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    """Augment point cloud data with random SE3 transformations."""
    pcd, robot_state_obs, robot_state_pred = batch
        
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

    # We shuffle the points - handle both 3D (T, P, C) and 4D (B, T, P, C) tensors
    if pcd.dim() == 4:
        # Batch format: shuffle along dim=2 (B, T, P, C)
        idx = torch.randperm(pcd.shape[2])
        pcd = pcd[:, :, idx, :]
    else:
        # Single sample format: shuffle along dim=1 (T, P, C)
        idx = torch.randperm(pcd.shape[1])
        pcd = pcd[:, idx, :]
    
    return pcd, robot_state_obs, robot_state_pred


class RobotDatasetPcdMasked(torch.utils.data.Dataset):
    """
    Dataset that returns only masked point clouds based on segmentation.
    Only points belonging to specified objects are kept.
    """
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        use_pc_color: bool,
        n_points: int,
        subs_factor: int = 1,
        object_ids: list = None,  # Object IDs to keep
        include_gripper: bool = True,  # Whether to include gripper points
        gripper_radius: float = 0.05,  # Radius around gripper
        min_points: int = 100,  # Minimum points required
    ) -> None:
        """
        Dataset that filters point clouds to only include specified objects.
        
        Args:
            object_ids: List of object IDs to keep in the point cloud
            include_gripper: Whether to include points near the gripper
            gripper_radius: Radius around gripper to include points
            min_points: Minimum number of points required (pad with zeros if less)
        """
        # Check if this is segmented data (multiple formats possible)
        has_segmentation = (
            os.path.exists(os.path.join(data_path, "segmentation")) or  # Old episode format
            os.path.exists(os.path.join(data_path, "replay_buffer.zarr")) or  # Zarr in subdirectory
            os.path.exists(os.path.join(data_path, "data", "attention_masks")) or  # Direct zarr format
            os.path.exists(os.path.join(data_path, ".zgroup"))  # Zarr root marker
        )
        if not has_segmentation:
            raise ValueError(f"Segmentation data not found in {data_path}. Please use segmented data.")
            
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        data_keys = ["robot_state", "pcd_xyz"]
        data_key_first_k = {"pcd_xyz": n_obs_steps * subs_factor}
        
        if use_pc_color:
            data_keys.append("pcd_color")
            data_key_first_k["pcd_color"] = n_obs_steps * subs_factor
            
        # Load preprocessed attention masks
        if "attention_masks" in replay_buffer.keys():
            print(f"Found pre-computed attention masks in {data_path}")
            data_keys.append("attention_masks")
            data_key_first_k["attention_masks"] = n_obs_steps * subs_factor
            self.has_precomputed_masks = True
        else:
            print(f"Warning: No pre-computed masks found. Point filtering may be slower.")
            self.has_precomputed_masks = False
                
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
        self.object_ids = object_ids if object_ids is not None else [31, 34, 35, 92]  # Default IDs
        self.include_gripper = include_gripper
        self.gripper_radius = gripper_radius
        self.min_points = min_points
        self.rng = np.random.default_rng()
        
        print(f"Initialized masked dataset with object IDs: {self.object_ids}")
        
    def __len__(self) -> int:
        return len(self.sampler)
    
    def filter_points_by_mask(self, pcd, attention_mask):
        """Filter point cloud to only include points with high attention."""
        # Attention mask is binary or continuous [0, 1]
        # Keep points with attention > 0.5
        mask = attention_mask > 0.5
        filtered_pcd = pcd[mask]
        
        # If too few points, pad with zeros
        n_filtered = filtered_pcd.shape[0]
        if n_filtered < self.min_points:
            # Pad with zeros
            padding = np.zeros((self.min_points - n_filtered, pcd.shape[1]))
            filtered_pcd = np.concatenate([filtered_pcd, padding], axis=0)
        
        return filtered_pcd, n_filtered
    
    def sample_points(self, pcd, n_points):
        """Sample n_points from the point cloud."""
        n_total = pcd.shape[0]
        
        if n_total >= n_points:
            # Randomly sample
            indices = self.rng.choice(n_total, n_points, replace=False)
        else:
            # Sample with replacement if not enough points
            indices = self.rng.choice(n_total, n_points, replace=True)
            
        return pcd[indices]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        data = self.sampler.sample_sequence(idx)
        
        # Extract robot states
        robot_state_full = data["robot_state"].astype(np.float32)[::self.subs_factor]
        robot_state_obs = robot_state_full[:self.n_obs_steps]
        robot_state_pred = robot_state_full[self.n_obs_steps:]
        
        # Process point clouds
        pcd_xyz = data["pcd_xyz"][::self.subs_factor][:self.n_obs_steps]
        
        # Get colors if requested
        if self.use_pc_color:
            pcd_color = data["pcd_color"][::self.subs_factor][:self.n_obs_steps]
            pcd = np.concatenate([pcd_xyz, pcd_color], axis=-1).astype(np.float32)
        else:
            pcd = pcd_xyz.astype(np.float32)
            
        # Filter points based on attention masks
        if self.has_precomputed_masks:
            attention_masks = data["attention_masks"][::self.subs_factor][:self.n_obs_steps]
            
            # Process each timestep
            filtered_pcds = []
            for t in range(self.n_obs_steps):
                # Filter points for this timestep
                filtered_pcd, n_valid = self.filter_points_by_mask(pcd[t], attention_masks[t])
                
                # Sample to target number of points
                if filtered_pcd.shape[0] > 0:
                    sampled_pcd = self.sample_points(filtered_pcd, self.n_points)
                else:
                    # If no valid points, create zero padding
                    sampled_pcd = np.zeros((self.n_points, pcd.shape[-1]))
                    
                filtered_pcds.append(sampled_pcd)
                
            pcd = np.stack(filtered_pcds, axis=0)
        else:
            print("Warning: No attention masks available. Using full point cloud.")
            # Sample from full point cloud
            sampled_pcds = []
            for t in range(self.n_obs_steps):
                sampled_pcd = self.sample_points(pcd[t], self.n_points)
                sampled_pcds.append(sampled_pcd)
            pcd = np.stack(sampled_pcds, axis=0)
            
        # Normalize point cloud
        # Center the point cloud around a reference point
        center = np.array([0.4, 0.0, 1.4], dtype=np.float32)
        pcd[..., :3] -= center
        
        # Convert to tensors
        pcd = torch.from_numpy(pcd)
        robot_state_obs = torch.from_numpy(robot_state_obs)
        robot_state_pred = torch.from_numpy(robot_state_pred)
        
        # Apply augmentation (can be controlled by a flag if needed)
        # For now, always apply augmentation during training
        return augment_pcd_data((pcd, robot_state_obs, robot_state_pred))