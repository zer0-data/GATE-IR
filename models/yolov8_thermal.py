"""
YOLOv8 Thermal: Custom Object Detector for Infrared Imagery

Component 3 (Stage C) of the GATE-IR pipeline.
Modified YOLOv8-Small architecture with:
- Single-channel thermal input
- P2 high-resolution detection head (160x160)
- Vision Transformer (ViT) neck for global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math


# ==============================================================================
# Building Blocks
# ==============================================================================

class ConvBNSiLU(nn.Module):
    """Standard Convolution + BatchNorm + SiLU activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block with residual connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        
        hidden = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, hidden, 3)
        self.cv2 = ConvBNSiLU(hidden, out_channels, 3)
        self.shortcut = shortcut and in_channels == out_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.cv2(self.cv1(x))
        if self.shortcut:
            out = out + x
        return out


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions (YOLOv8 style).
    
    Faster version of CSPBottleneck using split operations.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5
    ):
        super().__init__()
        
        self.c = int(out_channels * expansion)
        self.cv1 = ConvBNSiLU(in_channels, 2 * self.c, 1)
        self.cv2 = ConvBNSiLU((2 + n) * self.c, out_channels, 1)
        self.bottlenecks = nn.ModuleList([
            Bottleneck(self.c, self.c, shortcut, expansion=1.0)
            for _ in range(n)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer."""
    
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        
        hidden = in_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, hidden, 1)
        self.cv2 = ConvBNSiLU(hidden * 4, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ==============================================================================
# Vision Transformer Components
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """MLP (Feed-Forward) block for Transformer."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0
    ):
        super().__init__()
        
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """
    Standard Vision Transformer block.
    
    Consists of:
    - Layer Normalization
    - Multi-Head Self-Attention
    - Residual Connection
    - Layer Normalization
    - MLP (Feed-Forward Network)
    - Residual Connection
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerNeck(nn.Module):
    """
    Vision Transformer Neck for global context modeling.
    
    Placed between backbone and detection heads to capture
    long-range dependencies and recover information lost
    to weather degradation (e.g., rain streaks).
    
    Features are flattened, processed through transformer blocks,
    then reshaped back to spatial format.
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: Optional[int] = None,
        num_heads: int = 8,
        num_blocks: int = 1,
        mlp_ratio: float = 4.0,
        drop: float = 0.0
    ):
        """
        Initialize TransformerNeck.
        
        Args:
            in_channels: Number of input feature channels
            embed_dim: Transformer embedding dimension (defaults to in_channels)
            num_heads: Number of attention heads
            num_blocks: Number of transformer blocks
            mlp_ratio: MLP hidden dimension ratio
            drop: Dropout rate
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim or in_channels
        
        # Project to embedding dimension if different
        if self.embed_dim != in_channels:
            self.proj_in = nn.Conv2d(in_channels, self.embed_dim, 1)
            self.proj_out = nn.Conv2d(self.embed_dim, in_channels, 1)
        else:
            self.proj_in = nn.Identity()
            self.proj_out = nn.Identity()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, num_heads, mlp_ratio,
                qkv_bias=True, drop=drop
            )
            for _ in range(num_blocks)
        ])
        
        # Learnable position encoding (will be resized as needed)
        self.register_buffer('pos_embed', None)
    
    def get_pos_embed(self, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate or retrieve positional embeddings."""
        N = H * W
        
        # Generate 2D sinusoidal position encoding
        if self.pos_embed is None or self.pos_embed.shape[1] != N:
            pos_embed = self._create_2d_sincos_pos_embed(
                self.embed_dim, H, W, device
            )
            self.pos_embed = pos_embed
        
        return self.pos_embed
    
    def _create_2d_sincos_pos_embed(
        self,
        embed_dim: int,
        h: int,
        w: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create 2D sinusoidal positional embeddings."""
        grid_h = torch.arange(h, device=device, dtype=torch.float32)
        grid_w = torch.arange(w, device=device, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0).reshape(2, -1)
        
        # Compute sinusoidal embeddings
        dim_half = embed_dim // 4
        omega = torch.arange(dim_half, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / dim_half))
        
        out_h = grid[0:1].T @ omega.unsqueeze(0)
        out_w = grid[1:2].T @ omega.unsqueeze(0)
        
        pos_embed = torch.cat([
            torch.sin(out_h), torch.cos(out_h),
            torch.sin(out_w), torch.cos(out_w)
        ], dim=1)
        
        return pos_embed.unsqueeze(0)  # (1, H*W, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process features through transformer.
        
        Args:
            x: Feature map (B, C, H, W)
        
        Returns:
            Transformed feature map (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Project if needed
        x = self.proj_in(x)
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embeddings
        pos_embed = self.get_pos_embed(H, W, x.device)
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        
        # Project back if needed
        x = self.proj_out(x)
        
        return x


# ==============================================================================
# YOLOv8 Components
# ==============================================================================

class YOLOv8ThermalBackbone(nn.Module):
    """
    YOLOv8-Small backbone modified for single-channel thermal input.
    
    Outputs multi-scale features at P2, P3, P4, P5 resolutions.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50
    ):
        """
        Initialize backbone.
        
        Args:
            in_channels: Number of input channels (1 for thermal)
            base_channels: Base channel count
            depth_multiple: Depth multiplier for number of blocks
            width_multiple: Width multiplier for channel counts
        """
        super().__init__()
        
        self.in_channels = in_channels
        
        # Calculate actual channel counts
        def ch(c): return max(int(c * width_multiple), 8)
        def depth(d): return max(int(d * depth_multiple), 1)
        
        # P1/2: 320x320 (for 640 input)
        self.stem = ConvBNSiLU(in_channels, ch(64), 3, 2)
        
        # P2/4: 160x160
        self.stage1 = nn.Sequential(
            ConvBNSiLU(ch(64), ch(128), 3, 2),
            C2f(ch(128), ch(128), depth(3), True)
        )
        
        # P3/8: 80x80
        self.stage2 = nn.Sequential(
            ConvBNSiLU(ch(128), ch(256), 3, 2),
            C2f(ch(256), ch(256), depth(6), True)
        )
        
        # P4/16: 40x40
        self.stage3 = nn.Sequential(
            ConvBNSiLU(ch(256), ch(512), 3, 2),
            C2f(ch(512), ch(512), depth(6), True)
        )
        
        # P5/32: 20x20
        self.stage4 = nn.Sequential(
            ConvBNSiLU(ch(512), ch(1024), 3, 2),
            C2f(ch(1024), ch(1024), depth(3), True),
            SPPF(ch(1024), ch(1024), 5)
        )
        
        # Store output channel counts
        self.out_channels = {
            'P2': ch(128),
            'P3': ch(256),
            'P4': ch(512),
            'P5': ch(1024)
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor (B, 1, H, W)
        
        Returns:
            Dictionary of feature maps at P2, P3, P4, P5 resolutions
        """
        features = {}
        
        x = self.stem(x)  # P1
        
        x = self.stage1(x)
        features['P2'] = x  # P2/4
        
        x = self.stage2(x)
        features['P3'] = x  # P3/8
        
        x = self.stage3(x)
        features['P4'] = x  # P4/16
        
        x = self.stage4(x)
        features['P5'] = x  # P5/32
        
        return features


class PANNeck(nn.Module):
    """
    Path Aggregation Network (PAN) neck for multi-scale feature fusion.
    
    Combines top-down and bottom-up pathways for better feature aggregation.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        depth_multiple: float = 0.33
    ):
        super().__init__()
        
        p2_ch = in_channels['P2']
        p3_ch = in_channels['P3']
        p4_ch = in_channels['P4']
        p5_ch = in_channels['P5']
        
        def depth(d): return max(int(d * depth_multiple), 1)
        
        # Top-down (FPN) pathway
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P5 -> P4
        self.lateral_p5 = ConvBNSiLU(p5_ch, p4_ch, 1)
        self.fpn_p4 = C2f(p4_ch + p4_ch, p4_ch, depth(3), False)
        
        # P4 -> P3
        self.lateral_p4 = ConvBNSiLU(p4_ch, p3_ch, 1)
        self.fpn_p3 = C2f(p3_ch + p3_ch, p3_ch, depth(3), False)
        
        # P3 -> P2 (for P2 detection head)
        self.lateral_p3 = ConvBNSiLU(p3_ch, p2_ch, 1)
        self.fpn_p2 = C2f(p2_ch + p2_ch, p2_ch, depth(3), False)
        
        # Bottom-up (PAN) pathway
        # P2 -> P3
        self.downsample_p2 = ConvBNSiLU(p2_ch, p2_ch, 3, 2)
        self.pan_p3 = C2f(p2_ch + p3_ch, p3_ch, depth(3), False)
        
        # P3 -> P4
        self.downsample_p3 = ConvBNSiLU(p3_ch, p3_ch, 3, 2)
        self.pan_p4 = C2f(p3_ch + p4_ch, p4_ch, depth(3), False)
        
        # P4 -> P5
        self.downsample_p4 = ConvBNSiLU(p4_ch, p4_ch, 3, 2)
        self.pan_p5 = C2f(p4_ch + p5_ch, p5_ch, depth(3), False)
        
        self.out_channels = {
            'P2': p2_ch,
            'P3': p3_ch,
            'P4': p4_ch,
            'P5': p5_ch
        }
    
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Fuse multi-scale features.
        
        Args:
            features: Dictionary with P2, P3, P4, P5 feature maps
        
        Returns:
            Fused feature maps at each scale
        """
        p2 = features['P2']
        p3 = features['P3']
        p4 = features['P4']
        p5 = features['P5']
        
        # Top-down pathway
        p5_up = self.upsample(self.lateral_p5(p5))
        p4 = self.fpn_p4(torch.cat([p4, p5_up], dim=1))
        
        p4_up = self.upsample(self.lateral_p4(p4))
        p3 = self.fpn_p3(torch.cat([p3, p4_up], dim=1))
        
        p3_up = self.upsample(self.lateral_p3(p3))
        p2 = self.fpn_p2(torch.cat([p2, p3_up], dim=1))
        
        # Bottom-up pathway
        p2_down = self.downsample_p2(p2)
        p3 = self.pan_p3(torch.cat([p2_down, p3], dim=1))
        
        p3_down = self.downsample_p3(p3)
        p4 = self.pan_p4(torch.cat([p3_down, p4], dim=1))
        
        p4_down = self.downsample_p4(p4)
        p5 = self.pan_p5(torch.cat([p4_down, p5], dim=1))
        
        return {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5}


class P2DetectionHead(nn.Module):
    """
    Detection head for P2 high-resolution features (160x160).
    
    Optimized for small object detection in thermal imagery.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int = 16
    ):
        """
        Initialize P2 detection head.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of detection classes
            reg_max: Maximum regression value for DFL
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        
        # Classification branch
        self.cls_conv = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3),
            ConvBNSiLU(in_channels, in_channels, 3)
        )
        self.cls_pred = nn.Conv2d(in_channels, num_classes, 1)
        
        # Regression branch (bbox + DFL)
        self.reg_conv = nn.Sequential(
            ConvBNSiLU(in_channels, in_channels, 3),
            ConvBNSiLU(in_channels, in_channels, 3)
        )
        self.reg_pred = nn.Conv2d(in_channels, 4 * reg_max, 1)
        
        # Objectness branch
        self.obj_pred = nn.Conv2d(in_channels, 1, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate predictions from P2 features.
        
        Args:
            x: P2 feature map (B, C, H, W)
        
        Returns:
            Dictionary with cls, reg, obj predictions
        """
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(reg_feat)
        
        return {
            'cls': cls_pred,
            'reg': reg_pred,
            'obj': obj_pred
        }


class DetectionHead(nn.Module):
    """
    Multi-scale detection head for YOLO.
    
    Processes features at P2, P3, P4, P5 scales.
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        reg_max: int = 16,
        include_p2: bool = True
    ):
        """
        Initialize detection head.
        
        Args:
            in_channels: Channel counts for each scale
            num_classes: Number of detection classes
            reg_max: Maximum regression value
            include_p2: Whether to include P2 head
        """
        super().__init__()
        
        self.include_p2 = include_p2
        self.num_classes = num_classes
        
        # Create heads for each scale
        scales = ['P2', 'P3', 'P4', 'P5'] if include_p2 else ['P3', 'P4', 'P5']
        
        self.heads = nn.ModuleDict({
            scale: self._make_head(in_channels[scale], num_classes, reg_max)
            for scale in scales
        })
    
    def _make_head(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int
    ) -> nn.Module:
        """Create a single detection head."""
        return P2DetectionHead(in_channels, num_classes, reg_max)
    
    def forward(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate multi-scale predictions.
        
        Args:
            features: Multi-scale feature maps
        
        Returns:
            Predictions at each scale
        """
        predictions = {}
        
        for scale, head in self.heads.items():
            predictions[scale] = head(features[scale])
        
        return predictions


# ==============================================================================
# Complete YOLOv8 Thermal Model
# ==============================================================================

class YOLOv8Thermal(nn.Module):
    """
    Complete YOLOv8 model modified for thermal imagery.
    
    Key modifications from standard YOLOv8:
    1. Single-channel input (thermal vs 3-channel RGB)
    2. P2 high-resolution detection head (160x160 for small objects)
    3. Vision Transformer neck for global context modeling
    
    Architecture:
        Input (1, 640, 640)
            ↓
        Backbone (CSPDarknet-style)
            ↓
        [P2, P3, P4, P5] features
            ↓
        Transformer Neck (global context)
            ↓
        PAN Neck (multi-scale fusion)
            ↓
        Detection Heads (P2, P3, P4, P5)
            ↓
        Predictions
    
    Example:
        >>> model = YOLOv8Thermal(num_classes=3)
        >>> x = torch.rand(2, 1, 640, 640)
        >>> predictions = model(x)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 80,
        depth_multiple: float = 0.33,
        width_multiple: float = 0.50,
        include_p2: bool = True,
        use_transformer_neck: bool = True,
        transformer_heads: int = 8,
        transformer_blocks: int = 1
    ):
        """
        Initialize YOLOv8 Thermal.
        
        Args:
            in_channels: Number of input channels (1 for thermal)
            num_classes: Number of detection classes
            depth_multiple: Depth multiplier (0.33 for small)
            width_multiple: Width multiplier (0.50 for small)
            include_p2: Include P2 high-res detection head
            use_transformer_neck: Use ViT neck for global context
            transformer_heads: Number of attention heads
            transformer_blocks: Number of transformer blocks
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.include_p2 = include_p2
        self.use_transformer_neck = use_transformer_neck
        
        # Backbone
        self.backbone = YOLOv8ThermalBackbone(
            in_channels=in_channels,
            depth_multiple=depth_multiple,
            width_multiple=width_multiple
        )
        
        # Optional Transformer neck (applied to P4 for global context)
        if use_transformer_neck:
            self.transformer_neck = TransformerNeck(
                in_channels=self.backbone.out_channels['P4'],
                num_heads=transformer_heads,
                num_blocks=transformer_blocks
            )
        else:
            self.transformer_neck = None
        
        # PAN Neck for multi-scale fusion
        self.neck = PANNeck(
            in_channels=self.backbone.out_channels,
            depth_multiple=depth_multiple
        )
        
        # Detection heads
        self.head = DetectionHead(
            in_channels=self.neck.out_channels,
            num_classes=num_classes,
            include_p2=include_p2
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict:
        """
        Forward pass through complete model.
        
        Args:
            x: Input tensor (B, 1, H, W)
            return_features: Also return intermediate features
        
        Returns:
            Dictionary containing predictions (and optionally features)
        """
        # Backbone
        features = self.backbone(x)
        
        # Apply transformer to P4 for global context
        if self.transformer_neck is not None:
            features['P4'] = self.transformer_neck(features['P4'])
        
        # Neck (multi-scale fusion)
        fused_features = self.neck(features)
        
        # Detection heads
        predictions = self.head(fused_features)
        
        output = {'predictions': predictions}
        
        if return_features:
            output['backbone_features'] = features
            output['fused_features'] = fused_features
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature maps (for knowledge distillation)."""
        features = self.backbone(x)
        
        if self.transformer_neck is not None:
            features['P4'] = self.transformer_neck(features['P4'])
        
        fused_features = self.neck(features)
        
        return fused_features


# ==============================================================================
# Factory Functions
# ==============================================================================

def yolov8n_thermal(num_classes: int = 80, **kwargs) -> YOLOv8Thermal:
    """YOLOv8-Nano for thermal (fastest, smallest)."""
    return YOLOv8Thermal(
        num_classes=num_classes,
        depth_multiple=0.33,
        width_multiple=0.25,
        **kwargs
    )


def yolov8s_thermal(num_classes: int = 80, **kwargs) -> YOLOv8Thermal:
    """YOLOv8-Small for thermal (recommended)."""
    return YOLOv8Thermal(
        num_classes=num_classes,
        depth_multiple=0.33,
        width_multiple=0.50,
        **kwargs
    )


def yolov8m_thermal(num_classes: int = 80, **kwargs) -> YOLOv8Thermal:
    """YOLOv8-Medium for thermal (balanced)."""
    return YOLOv8Thermal(
        num_classes=num_classes,
        depth_multiple=0.67,
        width_multiple=0.75,
        **kwargs
    )


def yolov8l_thermal(num_classes: int = 80, **kwargs) -> YOLOv8Thermal:
    """YOLOv8-Large for thermal (teacher model)."""
    return YOLOv8Thermal(
        num_classes=num_classes,
        depth_multiple=1.0,
        width_multiple=1.0,
        **kwargs
    )


# ==============================================================================
# Test Code
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("YOLOv8 Thermal Model Test")
    print("=" * 60)
    
    # Create model
    model = yolov8s_thermal(
        num_classes=3,
        include_p2=True,
        use_transformer_neck=True
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.rand(batch_size, 1, 640, 640)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x, return_features=True)
    
    print("\nBackbone features:")
    for name, feat in output['backbone_features'].items():
        print(f"  {name}: {feat.shape}")
    
    print("\nFused features:")
    for name, feat in output['fused_features'].items():
        print(f"  {name}: {feat.shape}")
    
    print("\nPredictions:")
    for scale, preds in output['predictions'].items():
        print(f"  {scale}:")
        for key, pred in preds.items():
            print(f"    {key}: {pred.shape}")
    
    # Benchmark
    import time
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            _ = model(x)
        elapsed = (time.perf_counter() - start) / iterations * 1000
    
    print(f"\nInference latency: {elapsed:.2f} ms")
    print(f"Throughput: {batch_size / elapsed * 1000:.1f} images/sec")
