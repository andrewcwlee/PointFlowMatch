""" PointNet with Attention - Enhanced version with supervised 3D spatial attention mechanism """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from diffusion_policy.common.pytorch_util import replace_submodules


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
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

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
        # self.bn4 = nn.BatchNorm1d(512)
        # self.bn5 = nn.BatchNorm1d(256)

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


class PointNetfeatAttention(nn.Module):
    def __init__(self, input_channels: int, input_transform: bool, feature_transform=False, 
                 attention_hidden_dim=256):
        super(PointNetfeatAttention, self).__init__()
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
        
        # Attention module - applied after conv2 (128 channels)
        self.attention_head = nn.Sequential(
            nn.Conv1d(128, attention_hidden_dim, 1),
            nn.BatchNorm1d(attention_hidden_dim),
            nn.ReLU(),
            nn.Conv1d(attention_hidden_dim, 1, 1),
            nn.Sigmoid()
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
        
        # Generate attention weights after conv2 (128 channels)
        attention_scores = self.attention_head(x)  # [B, 1, NumPoints]
        attention_weights = attention_scores  # Already sigmoid output
        
        # Modulate features with attention (f'_i = α_i * f_i)
        x = x * attention_weights  # Broadcasting: [B, 128, NumPoints] * [B, 1, NumPoints]
        
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Return both features and attention map
        return x, attention_weights.squeeze(1)  # attention_weights: [B, NumPoints]


class PointNetAttentionBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        input_channels: int,
        input_transform: bool,
        use_group_norm: bool = False,
        attention_hidden_dim: int = 256,
    ):
        super().__init__()
        assert input_channels in [3, 6], "Input channels must be 3 or 6"
        
        # Create PointNetfeat with attention
        self.pointnet_feat = PointNetfeatAttention(
            input_channels, 
            input_transform, 
            attention_hidden_dim=attention_hidden_dim
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
        # Reshape back to the batch dimension. Now the features of each time step are concatenated
        nx = nx.reshape(B, -1)
        
        # Reshape attention weights back to batch dimension
        # attention_weights shape: [B*T, NumPoints]
        attention_weights = attention_weights.reshape(B, -1)  # [B, T*NumPoints]
        
        return nx, attention_weights