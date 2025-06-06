from __future__ import annotations
import torch
import numpy as np
import pypose as pp
from diffusion_policy.common.sampler import SequenceSampler
from pfp.data.replay_buffer import RobotReplayBuffer
from pfp.common.se3_utils import transform_th
from pfp import DATA_DIRS


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


class RobotDatasetPcdUncropped(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: str,
        n_obs_steps: int,
        n_pred_steps: int,
        use_pc_color: bool,
        n_points: int,
        subs_factor: int = 1,  # 1 means no subsampling
        voxel_size: float = 0.01,  # Voxel size for downsampling
        workspace_bounds: list = None,  # Optional workspace bounds [x_min, x_max, y_min, y_max, z_min, z_max]
    ) -> None:
        """
        Dataset that loads RAW point clouds without workspace cropping.
        
        This requires the demo data to have been collected with individual camera
        point clouds saved separately (not merged/cropped).
        """
        replay_buffer = RobotReplayBuffer.create_from_path(data_path, mode="r")
        
        # Check if raw camera data is available
        camera_keys = ["right_shoulder_pcd", "left_shoulder_pcd", "overhead_pcd", "front_pcd", "wrist_pcd"]
        has_raw_cameras = all(key in replay_buffer.keys() for key in camera_keys[:3])  # At least 3 cameras
        
        if has_raw_cameras:
            print("Found raw camera point clouds - will merge without workspace cropping")
            data_keys = ["robot_state"] + camera_keys
            data_key_first_k = {key: n_obs_steps * subs_factor for key in camera_keys}
        else:
            print("Raw camera data not found - using pre-merged point clouds (already cropped)")
            data_keys = ["robot_state", "pcd_xyz"]
            data_key_first_k = {"pcd_xyz": n_obs_steps * subs_factor}
            
        if use_pc_color:
            if has_raw_cameras:
                color_keys = [key.replace("pcd", "rgb") for key in camera_keys]
                data_keys.extend(color_keys)
                data_key_first_k.update({key: n_obs_steps * subs_factor for key in color_keys})
            else:
                data_keys.append("pcd_color")
                data_key_first_k["pcd_color"] = n_obs_steps * subs_factor
                
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
        self.voxel_size = voxel_size
        self.workspace_bounds = workspace_bounds
        self.has_raw_cameras = has_raw_cameras
        self.camera_keys = camera_keys if has_raw_cameras else []
        self.rng = np.random.default_rng()
        return

    def __len__(self) -> int:
        return len(self.sampler)

    def merge_raw_cameras(self, sample: dict, cur_step_i: int) -> np.ndarray:
        """Merge raw camera point clouds without workspace cropping."""
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Open3D required for raw camera merging")
            
        pcds = []
        for key in self.camera_keys:
            if key in sample:
                camera_pcd_data = sample[key][: cur_step_i : self.subs_factor]  # (T, N, 3)
                
                # Process each timestep
                for t in range(camera_pcd_data.shape[0]):
                    # Create Open3D point cloud
                    pcd_o3d = o3d.geometry.PointCloud()
                    pcd_o3d.points = o3d.utility.Vector3dVector(camera_pcd_data[t])
                    
                    # Add colors if available
                    if self.use_pc_color:
                        color_key = key.replace("pcd", "rgb")
                        if color_key in sample:
                            colors = sample[color_key][t].astype(np.float32) / 255.0
                            pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
                    
                    pcds.append(pcd_o3d)
        
        # Merge all point clouds
        if len(pcds) == 0:
            raise ValueError("No valid camera point clouds found")
            
        merged_pcd = pcds[0]
        for pcd in pcds[1:]:
            merged_pcd += pcd
            
        # Apply workspace bounds if specified (optional cropping)
        if self.workspace_bounds is not None:
            x_min, x_max, y_min, y_max, z_min, z_max = self.workspace_bounds
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                min_bound=[x_min, y_min, z_min],
                max_bound=[x_max, y_max, z_max]
            )
            merged_pcd = merged_pcd.crop(bbox)
            
        # Voxel downsample
        if len(merged_pcd.points) > 0:
            merged_pcd = merged_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
        # Convert back to numpy
        points = np.asarray(merged_pcd.points)
        colors = np.asarray(merged_pcd.colors) if merged_pcd.has_colors() else None
        
        # Sample or pad to desired number of points
        if len(points) > self.n_points:
            indices = self.rng.choice(len(points), self.n_points, replace=False)
            points = points[indices]
            if colors is not None:
                colors = colors[indices]
        elif len(points) < self.n_points:
            # Pad with zeros
            pad_points = np.zeros((self.n_points - len(points), 3))
            points = np.vstack([points, pad_points])
            if colors is not None:
                pad_colors = np.zeros((self.n_points - len(colors), 3))
                colors = np.vstack([colors, pad_colors])
                
        # Combine points and colors
        if colors is not None and self.use_pc_color:
            pcd_final = np.concatenate([points, colors], axis=-1)
        else:
            pcd_final = points
            
        return pcd_final

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        sample: dict[str, np.ndarray] = self.sampler.sample_sequence(idx)
        cur_step_i = self.n_obs_steps * self.subs_factor
        
        if self.has_raw_cameras:
            # Merge raw camera data without workspace cropping
            pcd = self.merge_raw_cameras(sample, cur_step_i)
            # Add time dimension
            pcd = np.expand_dims(pcd, 0).repeat(self.n_obs_steps, axis=0)
        else:
            # Use pre-merged data (already cropped)
            pcd = sample["pcd_xyz"][: cur_step_i : self.subs_factor]
            if self.use_pc_color:
                pcd_color = sample["pcd_color"][: cur_step_i : self.subs_factor]
                pcd_color = pcd_color.astype(np.float32) / 255.0
                pcd = np.concatenate([pcd, pcd_color], axis=-1)
                
        robot_state_obs = sample["robot_state"][: cur_step_i : self.subs_factor].astype(np.float32)
        robot_state_pred = sample["robot_state"][cur_step_i :: self.subs_factor].astype(np.float32)
        
        # Random sample pcd points if needed
        if pcd.shape[1] > self.n_points:
            random_indices = np.random.choice(pcd.shape[1], self.n_points, replace=False)
            pcd = pcd[:, random_indices]
                
        return pcd, robot_state_obs, robot_state_pred


if __name__ == "__main__":
    dataset = RobotDatasetPcdUncropped(
        data_path=DATA_DIRS.PFP / "open_fridge" / "train",
        n_obs_steps=2,
        n_pred_steps=8,
        subs_factor=5,
        use_pc_color=False,
        n_points=4096,
        workspace_bounds=None,  # No cropping
    )
    i = 20
    obs, robot_state_obs, robot_state_pred = dataset[i]
    print("robot_state_obs: ", robot_state_obs)
    print("robot_state_pred: ", robot_state_pred)
    print("done")