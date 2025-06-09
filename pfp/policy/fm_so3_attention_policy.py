from __future__ import annotations
import hydra
import torch
import torch.nn as nn
import pypose as pp
from omegaconf import OmegaConf
from composer.models import ComposerModel
from pfp.policy.base_policy import BasePolicy
from pfp import DEVICE, REPO_DIRS
from pfp.common.se3_utils import pfp_to_pose_th
from pfp.common.fm_utils import get_timesteps


class FMSO3AttentionPolicy(ComposerModel, BasePolicy):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_obs_steps: int,
        n_pred_steps: int,
        num_k_infer: int,
        obs_encoder: nn.Module,
        diffusion_net: nn.Module,
        augment_data: bool,
        loss_weights: dict[int],
        norm_pcd_center: list,
        loss_type: str,
        lambda_attention: float = 1.0,  # Weight for attention loss
        pos_emb_scale: int = 20,
        flow_schedule: str = "linear",
        exp_scale: float = None,
        snr_sampler: str = "uniform",
        noise_type: str = "uniform",  # uniform | biased
    ) -> None:
        ComposerModel.__init__(self)
        BasePolicy.__init__(self, n_obs_steps)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_obs_steps = n_obs_steps
        self.n_pred_steps = n_pred_steps
        self.pos_emb_scale = pos_emb_scale
        self.num_k_infer = num_k_infer
        self.obs_encoder = obs_encoder
        self.diffusion_net = diffusion_net
        self.norm_pcd_center = norm_pcd_center
        self.augment_data = augment_data
        self.ny_shape = (n_pred_steps, y_dim)
        self.l_w = loss_weights
        self.lambda_attention = lambda_attention  # Attention loss weight
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        self.snr_sampler = snr_sampler
        self.noise_type = noise_type
        if loss_type == "l2":
            self.loss_fun = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fun = nn.L1Loss()
        else:
            raise NotImplementedError
        
        # Cosine similarity loss for attention supervision (more natural than BCE)
        # No need for a separate loss function - implemented as method
        return
    
    def cosine_similarity_loss(self, pred_logits: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity loss for attention supervision.
        
        Args:
            pred_logits: Predicted attention logits [B, T*NumPoints]
            gt_mask: Ground truth attention mask [B, T*NumPoints]
            
        Returns:
            Negative cosine similarity (for minimization)
        """
        # Convert logits to softmax probabilities (normalized)
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_norm = torch.nn.functional.normalize(pred_probs, dim=-1)
        
        # Normalize ground truth mask to unit vector
        gt_norm = torch.nn.functional.normalize(gt_mask, dim=-1)
        
        # Compute cosine similarity and return negative (to minimize)
        cosine_sim = torch.nn.functional.cosine_similarity(pred_norm, gt_norm, dim=-1)
        return -cosine_sim.mean()  # Negative for minimization

    def set_num_k_infer(self, num_k_infer: int):
        self.num_k_infer = num_k_infer
        return

    def set_flow_schedule(self, flow_schedule: str, exp_scale: float):
        self.flow_schedule = flow_schedule
        self.exp_scale = exp_scale
        return

    def _norm_obs(self, pcd: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        pcd[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        return pcd

    def _norm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        # I only do centering here, no scaling, to keep the relative distances and interpretability
        robot_state[..., :3] -= torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] -= torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _denorm_robot_state(self, robot_state: torch.Tensor) -> torch.Tensor:
        robot_state[..., :3] += torch.tensor(self.norm_pcd_center, device=DEVICE)
        robot_state[..., 9] += torch.tensor(0.5, device=DEVICE)
        return robot_state

    def _norm_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(batch) == 3:
            pcd, robot_state_obs, robot_state_pred = batch
            pcd = self._norm_obs(pcd)
            robot_state_obs = self._norm_robot_state(robot_state_obs)
            robot_state_pred = self._norm_robot_state(robot_state_pred)
            return pcd, robot_state_obs, robot_state_pred
        else:
            pcd, robot_state_obs, robot_state_pred, attention_mask = batch
            pcd = self._norm_obs(pcd)
            robot_state_obs = self._norm_robot_state(robot_state_obs)
            robot_state_pred = self._norm_robot_state(robot_state_pred)
            return pcd, robot_state_obs, robot_state_pred, attention_mask

    def _rand_range(self, low: float, high: float, size: tuple[int]) -> torch.Tensor:
        return torch.rand(size, device=DEVICE) * (high - low) + low

    def _augment_data(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if len(batch) == 3:
            pcd, robot_state_obs, robot_state_pred = batch
        else:
            pcd, robot_state_obs, robot_state_pred, attention_mask = batch

        # xyz1 = self._rand_range(low=0.8, high=1.2, size=(3,))
        xyz2 = self._rand_range(low=-0.2, high=0.2, size=(3,))
        pcd[..., :3] = pcd[..., :3] + xyz2  # * xyz1 + xyz2
        robot_state_obs[..., :3] = robot_state_obs[..., :3] + xyz2  # * xyz1 + xyz2
        robot_state_pred[..., :3] = robot_state_pred[..., :3] + xyz2  # * xyz1 + xyz2

        # We shuffle the points, i.e. shuffle pcd along dim=2 (B, T, P, 3)
        idx = torch.randperm(pcd.shape[2])
        pcd = pcd[:, :, idx, :]
        
        if len(batch) == 4:
            # Also shuffle attention mask
            attention_mask = attention_mask[:, :, idx]
            return pcd, robot_state_obs, robot_state_pred, attention_mask
        else:
            return pcd, robot_state_obs, robot_state_pred

    def _init_noise(
        self, batch_size: int, robot_state_obs: torch.Tensor
    ) -> tuple[torch.Tensor, pp.SO3, torch.Tensor]:
        B = batch_size
        T = self.n_pred_steps
        noise_xyz = torch.randn((B, T, 3), device=DEVICE)
        noise_gripper = torch.randn((B, T, 1), device=DEVICE)
        if self.noise_type == "uniform":
            noise_SO3 = pp.randn_SO3((B, T), device=DEVICE)
        elif self.noise_type == "biased":
            random_euler = torch.FloatTensor(B, T, 3).uniform_(-torch.pi / 2, torch.pi / 2)
            random_so3 = pp.Log(pp.euler2SO3(random_euler.to(DEVICE)))
            _, cur_SO3, _ = self._pfp_to_pp(robot_state_obs)
            start_SO3 = cur_SO3[:, -1:, :].expand(B, T, 4)  # Just take the current pose
            noise_SO3 = start_SO3 @ pp.Exp(random_so3)
        else:
            raise NotImplementedError
        return noise_xyz, noise_SO3, noise_gripper

    def _pfp_to_pp(self, pfp_state: torch.Tensor) -> tuple[pp.SE3, torch.Tensor]:
        """
        pfp_state: (B, T, 10) -> xyz, rot6d, grip
        """
        poses_th, gripper_th = pfp_to_pose_th(pfp_state)  # (B, T, 4, 4)
        xyz = poses_th[..., :3, 3]
        rot_SO3 = pp.mat2SO3(poses_th[..., :3, :3], check=False)  # (B, T, 4)
        gripper = gripper_th
        return xyz, rot_SO3, gripper

    def _sample_snr(self, batch_size: int) -> torch.Tensor:
        if self.snr_sampler == "uniform":
            return torch.rand((batch_size, 1, 1), device=DEVICE)
        elif self.snr_sampler == "logit_normal":
            return torch.sigmoid(torch.randn((batch_size, 1, 1), device=DEVICE))
        else:
            raise NotImplementedError

    def _pp_to_pfp(
        self, z_xyz: torch.Tensor, z_SO3: pp.SO3, z_gripper: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_xyz: (B, T, 3) xyz
            z_SO3: (B, T, 4) pp.SO3 rotation
            z_gripper: (B, T, 1) gripper
        Returns:
            z: (B, T, 10) pfp state
        """
        B, T, _ = z_xyz.shape
        z = torch.zeros((B, T, 10), device=DEVICE)
        rot = pp.matrix(z_SO3)
        z[..., :3] = z_xyz
        z[..., 3:9] = rot[..., :3, :2].mT.flatten(start_dim=-2)
        z[..., 9:] = z_gripper
        return z

    # ############### Training ################

    def forward(self, batch):
        """batch is the output of the dataloader"""
        return 0

    def loss(self, outputs, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """
        outputs: the output of the forward pass
        batch: the output of the dataloader
        """
        with torch.no_grad():
            batch = self._norm_data(batch)
            if self.augment_data:
                batch = self._augment_data(batch)
        
        if len(batch) == 3:
            # No attention mask provided
            pcd, robot_state_obs, robot_state_pred = batch
            attention_mask = None
        else:
            # Attention mask provided
            pcd, robot_state_obs, robot_state_pred, attention_mask = batch
            
        loss_xyz, loss_so3, loss_grip, loss_attention, attention_iou, cosine_sim_value, attention_entropy, gt_coverage, pred_sparsity = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred, attention_mask
        )
        
        # Calculate pure task loss (without attention)
        task_loss = (
            self.l_w["xyz"] * loss_xyz + 
            self.l_w["so3"] * loss_so3 + 
            self.l_w["grip"] * loss_grip
        )
        
        # Start with task loss
        total_loss = task_loss
        
        # Add attention loss if available
        if loss_attention is not None:
            total_loss = total_loss + self.lambda_attention * loss_attention
            
            # Log attention loss and metrics
            log_dict = {
                "loss/train/attention_cosine": loss_attention.item(),
                "metrics/train/attention_iou": attention_iou.item() if attention_iou is not None else 0.0,
            }
            
            # Add cosine similarity and other metrics if available
            if cosine_sim_value is not None:
                log_dict["metrics/train/cosine_similarity"] = cosine_sim_value.item()
            if attention_entropy is not None:
                log_dict["metrics/train/attention_entropy"] = attention_entropy.item()
            if gt_coverage is not None:
                log_dict["metrics/train/gt_coverage"] = gt_coverage.item()
            if pred_sparsity is not None:
                log_dict["metrics/train/pred_sparsity"] = pred_sparsity.item()
                
            self.logger.log_metrics(log_dict)
            
        # Log all losses: individual components, total (task), and total_attention (task + attention)
        self.logger.log_metrics(
            {
                "loss/train/xyz": loss_xyz.item(),
                "loss/train/so3": loss_so3.item(),
                "loss/train/grip": loss_grip.item(),
                "loss/train/total": task_loss.item(),           # Pure task performance (xyz + so3 + grip)
                "loss/train/total_attention": total_loss.item(),  # Complete training loss (task + attention)
            }
        )
        
        return total_loss

    def calculate_loss(
        self, 
        pcd: torch.Tensor, 
        robot_state_obs: torch.Tensor, 
        robot_state_pred: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        # Get encoder outputs - now returns both features and attention weights
        encoder_outputs = self.obs_encoder(pcd, robot_state_obs)
        
        if isinstance(encoder_outputs, tuple):
            nx, predicted_attention_map = encoder_outputs
        else:
            # Fallback for non-attention encoder
            nx = encoder_outputs
            predicted_attention_map = None
            
        ny: torch.Tensor = robot_state_pred

        B = ny.shape[0]
        T = ny.shape[1]

        # Sample random time step
        t = self._sample_snr(B)

        # Initialize start and end poses + gripper state
        z0_xyz, z0_SO3, z0_gripper = self._init_noise(B, robot_state_obs)
        z1_xyz, z1_SO3, z1_gripper = self._pfp_to_pp(ny)

        # Calculate relative change between them
        target_vel_xyz = z1_xyz - z0_xyz
        target_vel_so3 = pp.Log(pp.Inv(z0_SO3) @ z1_SO3)
        target_vel_gripper = z1_gripper - z0_gripper

        # Move to intermediate step
        zt_xyz = z0_xyz + target_vel_xyz * t
        zt_SO3: pp.SO3 = z0_SO3 @ pp.Exp(target_vel_so3 * t)
        zt_gripper: torch.Tensor = z0_gripper + target_vel_gripper * t

        # Convert to pfp network input representation
        zt_pfp = self._pp_to_pfp(zt_xyz, zt_SO3, zt_gripper)
        timesteps = t.squeeze() * self.pos_emb_scale

        # Do prediction
        pred_vel_pfp = self.diffusion_net(zt_pfp, timesteps, global_cond=nx)
        assert pred_vel_pfp.shape == (B, T, 7)
        pred_vel_xyz = pred_vel_pfp[..., :3]
        pred_vel_so3 = pred_vel_pfp[..., 3:6]
        pred_vel_gripper = pred_vel_pfp[..., 6:]

        # Calculate trajectory losses
        loss_xyz = self.loss_fun(pred_vel_xyz, target_vel_xyz)
        loss_so3 = self.loss_fun(pred_vel_so3, target_vel_so3)
        loss_grip = self.loss_fun(pred_vel_gripper, target_vel_gripper)
        
        # Calculate attention loss if both prediction and ground truth are available
        loss_attention = None
        attention_iou = None
        cosine_sim_value = None
        attention_entropy = None
        gt_coverage = None
        pred_sparsity = None
        
        if predicted_attention_map is not None and attention_mask is not None:
            # Flatten attention masks
            # predicted_attention_map: [B, T*NumPoints]
            # attention_mask: [B, T, NumPoints]
            attention_mask_flat = attention_mask.reshape(B, -1).float()
            
            # Cosine similarity loss
            loss_attention = self.cosine_similarity_loss(predicted_attention_map, attention_mask_flat)
            
            # Additional metrics for monitoring
            with torch.no_grad():
                # Convert logits to probabilities
                pred_probs = torch.softmax(predicted_attention_map, dim=-1)
                
                # Cosine similarity value (for logging)
                pred_norm = torch.nn.functional.normalize(pred_probs, dim=-1)
                gt_norm = torch.nn.functional.normalize(attention_mask_flat, dim=-1)
                cosine_sim_value = torch.nn.functional.cosine_similarity(pred_norm, gt_norm, dim=-1).mean()
                
                # IoU metric (keep for comparison with BCE)
                pred_binary = (pred_probs > 0.5).float()
                gt_binary = attention_mask_flat
                intersection = (pred_binary * gt_binary).sum(dim=1)
                union = ((pred_binary + gt_binary) > 0).float().sum(dim=1)
                attention_iou = (intersection / (union + 1e-6)).mean()
                
                # Attention entropy (measure of focus vs diffusion)
                attention_entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(dim=-1).mean()
                
                # Ground truth coverage (how much of GT is attended)
                gt_coverage = attention_mask_flat.sum(dim=-1).mean() / attention_mask_flat.shape[-1]
                
                # Prediction sparsity (how focused is the attention)
                pred_sparsity = (pred_probs > 0.1).float().sum(dim=-1).mean() / pred_probs.shape[-1]
        
        return loss_xyz, loss_so3, loss_grip, loss_attention, attention_iou, cosine_sim_value, attention_entropy, gt_coverage, pred_sparsity

    # ############### Inference ################

    def eval_forward(self, batch: tuple[torch.Tensor, ...], outputs=None) -> torch.Tensor:
        """
        batch: the output of the eval dataloader
        outputs: the output of the forward pass
        """
        batch = self._norm_data(batch)
        if len(batch) == 3:
            pcd, robot_state_obs, robot_state_pred = batch
            attention_mask = None
        else:
            pcd, robot_state_obs, robot_state_pred, attention_mask = batch

        # Eval loss
        loss_xyz, loss_so3, loss_grip, loss_attention, attention_iou, cosine_sim_value, attention_entropy, gt_coverage, pred_sparsity = self.calculate_loss(
            pcd, robot_state_obs, robot_state_pred, attention_mask
        )
        
        # Calculate pure task loss (without attention)
        task_loss = (
            self.l_w["xyz"] * loss_xyz + 
            self.l_w["so3"] * loss_so3 + 
            self.l_w["grip"] * loss_grip
        )
        
        # Start with task loss
        loss_total = task_loss
        
        if loss_attention is not None:
            loss_total = loss_total + self.lambda_attention * loss_attention
            
            # Log evaluation attention loss and metrics
            log_dict = {
                "loss/eval/attention_cosine": loss_attention.item(),
                "metrics/eval/attention_iou": attention_iou.item() if attention_iou is not None else 0.0,
            }
            
            # Add cosine similarity and other metrics if available
            if cosine_sim_value is not None:
                log_dict["metrics/eval/cosine_similarity"] = cosine_sim_value.item()
            if attention_entropy is not None:
                log_dict["metrics/eval/attention_entropy"] = attention_entropy.item()
            if gt_coverage is not None:
                log_dict["metrics/eval/gt_coverage"] = gt_coverage.item()
            if pred_sparsity is not None:
                log_dict["metrics/eval/pred_sparsity"] = pred_sparsity.item()
                
            self.logger.log_metrics(log_dict)
            
        # Log all losses: individual components, total (task), and total_attention (task + attention)
        self.logger.log_metrics(
            {
                "loss/eval/xyz": loss_xyz.item(),
                "loss/eval/so3": loss_so3.item(),
                "loss/eval/grip": loss_grip.item(),
                "loss/eval/total": task_loss.item(),         # Pure task performance (xyz + so3 + grip)
                "loss/eval/total_attention": loss_total.item(),  # Complete evaluation loss (task + attention)
            }
        )

        # Eval metrics
        pred_y = self.infer_y(pcd, robot_state_obs)
        mse_xyz = nn.functional.mse_loss(pred_y[..., :3], robot_state_pred[..., :3])
        mse_rot6d = nn.functional.mse_loss(pred_y[..., 3:9], robot_state_pred[..., 3:9])
        mse_grip = nn.functional.mse_loss(pred_y[..., 9], robot_state_pred[..., 9])
        self.logger.log_metrics(
            {
                "metrics/eval/mse_xyz": mse_xyz.item(),
                "metrics/eval/mse_rot6d": mse_rot6d.item(),
                "metrics/eval/mse_grip": mse_grip.item(),
            }
        )
        return pred_y

    def infer_y(
        self,
        pcd: torch.Tensor,
        robot_state_obs: torch.Tensor,
        noise=None,
        return_traj=False,
    ) -> torch.Tensor:
        encoder_outputs = self.obs_encoder(pcd, robot_state_obs)
        if isinstance(encoder_outputs, tuple):
            nx, _ = encoder_outputs  # Ignore attention weights during inference
        else:
            nx = encoder_outputs
            
        B = nx.shape[0]
        z_xyz, z_SO3, z_gripper = self._init_noise(B, robot_state_obs) if noise is None else noise
        z = self._pp_to_pfp(z_xyz, z_SO3, z_gripper)
        traj = [z]
        t0, dt = get_timesteps(self.flow_schedule, self.num_k_infer, exp_scale=self.exp_scale)
        for i in range(self.num_k_infer):
            t = torch.ones((B), device=DEVICE) * t0[i]
            timesteps = t * self.pos_emb_scale
            pred_vel_pfp = self.diffusion_net(z, timesteps, global_cond=nx)
            pred_vel_xyz = pred_vel_pfp[..., :3]
            pred_vel_so3 = pp.so3(pred_vel_pfp[..., 3:6])
            pred_vel_gripper = pred_vel_pfp[..., 6:]

            z_xyz = z_xyz + pred_vel_xyz * dt[i]
            z_SO3 = z_SO3 @ pp.Exp(pred_vel_so3 * dt[i])
            z_gripper = z_gripper + pred_vel_gripper * dt[i]

            z = self._pp_to_pfp(z_xyz, z_SO3, z_gripper)
            traj.append(z)
        return torch.stack(traj) if return_traj else traj[-1]

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_name: str,
        ckpt_episode: str,
        num_k_infer: int,
        flow_schedule: str = None,
        exp_scale: float = None,
    ):
        ckpt_dir = REPO_DIRS.CKPT / ckpt_name
        ckpt_path_list = list(ckpt_dir.glob(f"{ckpt_episode}*"))
        assert len(ckpt_path_list) > 0, f"No checkpoint found in {ckpt_dir} with {ckpt_episode}"
        assert len(ckpt_path_list) < 2, f"Multiple ckpts found in {ckpt_dir} with {ckpt_episode}"
        ckpt_fpath = ckpt_path_list[0]

        state_dict = torch.load(ckpt_fpath, map_location=DEVICE)
        cfg = OmegaConf.load(ckpt_dir / "config.yaml")
        # cfg.model.obs_encoder.encoder.random_crop = False
        assert cfg.model._target_.split(".")[-1] == cls.__name__
        model: FMSO3AttentionPolicy = hydra.utils.instantiate(cfg.model)
        model.load_state_dict(state_dict["state"]["model"])
        model.to(DEVICE)
        model.eval()
        if flow_schedule is not None:
            model.set_flow_schedule(flow_schedule, exp_scale)
        if num_k_infer is not None:
            model.set_num_k_infer(num_k_infer)
        return model


class FMSO3AttentionPolicyImage(FMSO3AttentionPolicy):

    def _norm_obs(self, image: torch.Tensor) -> torch.Tensor:
        """
        Image normalization is already done in the backbone, so here we just make it float
        """
        image = image.float() / 255.0
        return image