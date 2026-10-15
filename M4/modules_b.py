import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, '/workspace/python_pro/test1/RopStitch/wCoefNet/Codes')
import grid_res

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W


class CrossAttentionMatcher(nn.Module):
    """
    Bilateral Multi-Head Cross-Attention Module for Inter-Image Feature Interaction.
    Allows features in Image 1 to query matching content in Image 2, and vice versa.
    """
    def __init__(self, in_channels=128, num_heads=4, hidden_dim=64):
        super(CrossAttentionMatcher, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Downsample spatial dimension for efficient long-range cross-attention
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)  # 64x64 -> 32x32

        # Linear projections for Q, K, V
        self.q1_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.k1_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.v1_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        self.q2_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.k2_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.v2_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)

        # Output projection back to in_channels
        self.out1_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        )
        self.out2_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        )

        # Feed Forward Network (FFN)
        self.ffn1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        )
        self.ffn2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, in_channels), num_channels=in_channels)
        )

    def _cross_attend(self, q, k, v, B, H, W):
        # Shape: [B, num_heads, H*W, head_dim]
        N = H * W
        q = q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B, num_heads, N, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(B, self.hidden_dim, H, W)
        return out

    def forward(self, f1, f2):
        """
        f1: [B, 128, H, W]
        f2: [B, 128, H, W]
        """
        B, C, H, W = f1.shape
        f1_down = self.down(f1)
        f2_down = self.down(f2)
        _, _, H_d, W_d = f1_down.shape

        q1 = self.q1_proj(f1_down)
        k1 = self.k1_proj(f1_down)
        v1 = self.v1_proj(f1_down)

        q2 = self.q2_proj(f2_down)
        k2 = self.k2_proj(f2_down)
        v2 = self.v2_proj(f2_down)

        # Cross attention: f1 queries f2, f2 queries f1
        out1_d = self._cross_attend(q1, k2, v2, B, H_d, W_d)
        out2_d = self._cross_attend(q2, k1, v1, B, H_d, W_d)

        # Upsample back to H, W
        out1_up = F.interpolate(out1_d, size=(H, W), mode='bilinear', align_corners=True)
        out2_up = F.interpolate(out2_d, size=(H, W), mode='bilinear', align_corners=True)

        # Residual add + FFN
        f1_enhanced = f1 + self.out1_proj(out1_up)
        f1_enhanced = f1_enhanced + self.ffn1(f1_enhanced)

        f2_enhanced = f2 + self.out2_proj(out2_up)
        f2_enhanced = f2_enhanced + self.ffn2(f2_enhanced)

        return f1_enhanced, f2_enhanced


class ConvMeshHead(nn.Module):
    """
    Topology-Preserving Fully Convolutional Mesh Regression Head.
    Completely eliminates the Flatten + Linear dense layer bottleneck.
    Directly predicts (grid_h+1, grid_w+1) vertex motion from 2D spatial features.
    """
    def __init__(self, in_channels=128, hidden_dim=64):
        super(ConvMeshHead, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # 2D spatial convolution blocks with group normalization & residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.GELU()
        )

        # Resample features to target mesh resolution (grid_h+1, grid_w+1) = (13, 13)
        self.pool_to_grid = nn.AdaptiveAvgPool2d((self.grid_h + 1, self.grid_w + 1))

        # Final 2D prediction head: outputs (dx, dy) for each vertex
        self.pred_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, 2, kernel_size=1)
        )

    def forward(self, feature_map):
        """
        feature_map: [B, in_channels, H, W]
        Returns:
            motion: [B, grid_h+1, grid_w+1, 2]
        """
        h = self.conv1(feature_map)
        h = h + self.conv2(h)
        grid_feat = self.pool_to_grid(h)  # [B, hidden_dim, 13, 13]
        motion = self.pred_head(grid_feat)  # [B, 2, 13, 13]

        # Permute to [B, 13, 13, 2] to match RopStitch mesh convention
        motion = motion.permute(0, 2, 3, 1).contiguous()
        return motion
