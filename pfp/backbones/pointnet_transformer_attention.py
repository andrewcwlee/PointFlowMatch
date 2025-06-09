""" PointNet with Transformer-based Attention - True self-attention mechanism """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from diffusion_policy.common.pytorch_util import replace_submodules
import math


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)))
            .view(1, 9)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.LayerNorm(512)
        self.bn5 = nn.LayerNorm(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = (
            Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)))
            .view(1, self.k * self.k)
            .repeat(batchsize, 1)
        )
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class TransformerAttentionPooling(nn.Module):
    """
    Transformer-based attention pooling for point clouds.
    Uses self-attention to learn relationships between points.
    """
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, 
            nhead, 
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Learnable global token for pooling
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] point features
        Returns:
            pooled: [B, D] global features
            attention_weights: [B, N] attention weights for each point
        """
        B, N, D = x.shape
        
        # Add global token to the sequence
        global_tokens = self.global_token.expand(B, -1, -1)
        x_with_global = torch.cat([global_tokens, x], dim=1)  # [B, N+1, D]
        
        # Self-attention
        x2, attn_weights = self.self_attn(
            x_with_global, x_with_global, x_with_global,
            need_weights=True, average_attn_weights=True
        )
        x_with_global = x_with_global + self.dropout1(x2)
        x_with_global = self.norm1(x_with_global)
        
        # Feedforward
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x_with_global))))
        x_with_global = x_with_global + self.dropout2(x2)
        x_with_global = self.norm2(x_with_global)
        
        # Extract global token as pooled representation
        pooled = x_with_global[:, 0]  # [B, D]
        
        # Get attention weights from global token to all points
        # These represent how much each point contributes to the global representation
        # Shape of attn_weights: [B, N+1, N+1]
        # We want the attention from global token (index 0) to all other tokens
        point_attention_weights = attn_weights[:, 0, 1:]  # [B, N]
        
        return pooled, point_attention_weights


class CrossAttentionPooling(nn.Module):
    """
    Cross-attention pooling using multiple learnable query tokens.
    This learns to extract different aspects of the point cloud.
    """
    def __init__(self, d_model, num_queries=1, nhead=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, d_model))
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # If using multiple queries, need to aggregate them
        if num_queries > 1:
            self.aggregation = nn.Linear(num_queries * d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] point features
        Returns:
            pooled: [B, D] global features
            attention_weights: [B, N] attention weights for each point
        """
        B, N, D = x.shape
        
        # Expand query tokens for batch
        queries = self.query_tokens.expand(B, -1, -1)  # [B, num_queries, D]
        
        # Cross-attention: queries attend to point features
        attended, attn_weights = self.cross_attn(
            queries, x, x,
            need_weights=True,
            average_attn_weights=True
        )
        
        # Add residual and normalize
        queries = self.norm(queries + attended)
        
        # Aggregate multiple queries if needed
        if self.num_queries > 1:
            queries_flat = queries.reshape(B, -1)  # [B, num_queries * D]
            pooled = self.aggregation(queries_flat)  # [B, D]
            # Average attention weights across queries
            point_attention_weights = attn_weights.mean(dim=1)  # [B, N]
        else:
            pooled = queries.squeeze(1)  # [B, D]
            point_attention_weights = attn_weights.squeeze(1)  # [B, N]
        
        return pooled, point_attention_weights


class PointNetfeatTransformerAttention(nn.Module):
    def __init__(self, input_channels: int, input_transform: bool, feature_transform=False,
                 attention_type='self', nhead=8, dropout=0.1):
        super(PointNetfeatTransformerAttention, self).__init__()
        self.input_transform = input_transform
        if self.input_transform:
            self.stn = STNkd(k=input_channels)
        self.conv1 = torch.nn.Conv1d(input_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)
        
        # Choose transformer attention type
        assert attention_type in ['self', 'cross']
        if attention_type == 'self':
            self.attention_pooling = TransformerAttentionPooling(
                d_model=1024,
                nhead=nhead,
                dim_feedforward=2048,
                dropout=dropout
            )
        else:  # cross
            self.attention_pooling = CrossAttentionPooling(
                d_model=1024,
                num_queries=1,
                nhead=nhead,
                dropout=dropout
            )
        
    def forward(self, x):
        b = x.size(0)
        if len(x.shape) == 4:
            x = x.view(b, -1, 3).permute(0, 2, 1).contiguous()

        if self.input_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None

        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # [B, 1024, NumPoints]
        
        # Transpose for transformer (expects [B, N, D])
        x = x.transpose(1, 2)  # [B, NumPoints, 1024]
        
        # Apply transformer attention pooling
        pooled_features, attention_weights = self.attention_pooling(x)
        
        # attention_weights are already normalized (sum to 1)
        # Convert to logits for consistency with other implementations
        # Using log to convert probabilities back to logits
        attention_logits = torch.log(attention_weights + 1e-8)
        
        return pooled_features, attention_logits


class PointNetTransformerAttentionBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_channels: int,
        input_transform: bool,
        use_group_norm: bool = False,
        attention_type: str = 'self',
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert input_channels in [3, 6], "Input channels must be 3 or 6"
        
        # Create PointNetfeat with transformer attention
        self.pointnet_feat = PointNetfeatTransformerAttention(
            input_channels, 
            input_transform,
            attention_type=attention_type,
            nhead=nhead,
            dropout=dropout
        )
        
        # Create the rest of the backbone
        self.feature_transform = nn.Sequential(
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.Mish(),
            nn.Linear(512, embed_dim),
        )
        
        if use_group_norm:
            self.pointnet_feat = replace_submodules(
                root_module=self.pointnet_feat,
                predicate=lambda x: isinstance(x, nn.BatchNorm1d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
            self.feature_transform = replace_submodules(
                root_module=self.feature_transform,
                predicate=lambda x: isinstance(x, nn.BatchNorm1d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features
                ),
            )
        return

    def forward(self, pcd: torch.Tensor, robot_state_obs: torch.Tensor = None):
        B = pcd.shape[0]
        # Flatten the batch and time dimensions
        pcd = pcd.float().reshape(-1, *pcd.shape[2:])
        robot_state_obs = robot_state_obs.float().reshape(-1, *robot_state_obs.shape[2:])
        # Permute [B, P, C] -> [B, C, P]
        pcd = pcd.permute(0, 2, 1)
        
        # Get both features and attention weights
        pointnet_features, attention_weights = self.pointnet_feat(pcd)
        encoded_pcd = self.feature_transform(pointnet_features)
        
        nx = torch.cat([encoded_pcd, robot_state_obs], dim=1)
        # Reshape back to the batch dimension
        nx = nx.reshape(B, -1)
        
        # Reshape attention logits back to batch dimension
        attention_logits = attention_weights.reshape(B, -1)
        
        return nx, attention_logits