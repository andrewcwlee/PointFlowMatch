""" Transformer backbone with CLS token for point cloud processing """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from diffusion_policy.common.pytorch_util import replace_submodules
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D coordinates"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create a matrix to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for the sinusoidal pattern
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cos to odd indices
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for 3D points"""
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        
    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: [B, N, 3] or [B, N, 6] - 3D coordinates (and optionally colors)
            features: [B, N, d_model] - point features
        Returns:
            [B, N, d_model] - features with positional encoding
        """
        pos_encoding = self.proj(points[..., :3])  # Only use xyz coordinates
        return features + pos_encoding


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        Returns:
            output: [B, N, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feedforward with residual connection
        x = self.norm2(x + self.ff(x))
        
        return x


class PointTransformerBackbone(nn.Module):
    """Transformer backbone with CLS token for point cloud processing"""
    def __init__(
        self,
        embed_dim: int,
        input_channels: int,
        hidden_dim: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        positional_encoding: str = "learned",  # "learned" or "sinusoidal"
        use_group_norm: bool = False,
        use_gradient_checkpointing: bool = False,  # Enable for memory efficiency
    ):
        super().__init__()
        
        # Store dimensions
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Positional encoding
        if positional_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(3, hidden_dim)
        elif positional_encoding == "sinusoidal":
            self.pos_encoder = PositionalEncoding(hidden_dim)
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        self.positional_encoding = positional_encoding
        
        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection for CLS token
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Mish()
        )
        
        # Apply group norm if requested
        if use_group_norm:
            self._replace_batchnorm_with_groupnorm()
    
    def _replace_batchnorm_with_groupnorm(self):
        """Replace any BatchNorm layers with GroupNorm"""
        self.transformer_blocks = replace_submodules(
            root_module=self.transformer_blocks,
            predicate=lambda x: isinstance(x, nn.LayerNorm),
            func=lambda x: nn.GroupNorm(
                num_groups=x.normalized_shape[0] // 16, 
                num_channels=x.normalized_shape[0]
            ),
        )
    
    def forward(self, pcd: torch.Tensor, robot_state_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pcd: [B, T, N, C] - point cloud observations
            robot_state_obs: [B, T, D] - robot state observations
        Returns:
            nx: [B, embed_dim] - global features for UNet conditioning
        """
        B = pcd.shape[0]
        T = pcd.shape[1] if len(pcd.shape) == 4 else 1
        
        # Flatten batch and time dimensions
        if len(pcd.shape) == 4:
            pcd = pcd.reshape(-1, *pcd.shape[2:])  # [B*T, N, C]
            robot_state_obs = robot_state_obs.reshape(-1, *robot_state_obs.shape[2:])  # [B*T, D]
        
        BT, N, C = pcd.shape
        
        # Project input points to hidden dimension
        point_features = self.input_proj(pcd)  # [B*T, N, hidden_dim]
        
        # Add positional encoding
        if self.positional_encoding == "learned":
            point_features = self.pos_encoder(pcd, point_features)
        else:
            point_features = self.pos_encoder(point_features)
        
        # Expand CLS token and concatenate
        cls_tokens = self.cls_token.expand(BT, -1, -1)  # [B*T, 1, hidden_dim]
        x = torch.cat([cls_tokens, point_features], dim=1)  # [B*T, N+1, hidden_dim]
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training
                x = checkpoint(block, x)
            else:
                x = block(x)
        
        # Extract CLS token
        cls_output = x[:, 0, :]  # [B*T, hidden_dim]
        
        # Project CLS token to output dimension
        encoded_pcd = self.output_proj(cls_output)  # [B*T, embed_dim]
        
        # Concatenate with robot state
        nx = torch.cat([encoded_pcd, robot_state_obs], dim=1)  # [B*T, embed_dim + D]
        
        # Reshape back to batch dimension
        nx = nx.reshape(B, -1)  # [B, T*(embed_dim + D)]
        
        return nx