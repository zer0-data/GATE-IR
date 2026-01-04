"""
RainRemover: Lightweight Spatial Residual Block for Rain Streak Removal

Component 2 (Stage B) of the GATE-IR pipeline.
Implements LSRB with Depthwise Separable Convolutions for efficient 
rain mask prediction, followed by CLAHE for contrast restoration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import numpy as np

# Try to import OpenCV for CLAHE
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block.
    
    Splits standard convolution into:
    1. Depthwise: One filter per input channel
    2. Pointwise: 1x1 convolution to mix channels
    
    Achieves ~8-9x speedup over standard conv for 3x3 kernels.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = True
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        # Depthwise convolution: groups = in_channels
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LSRB(nn.Module):
    """
    Lightweight Spatial Residual Block (LSRB).
    
    Uses 3 layers of Depthwise Separable Convolutions to predict
    a rain mask. The residual connection allows learning the rain
    component that gets subtracted from the input.
    
    Architecture:
        Input → DSConv → BN → ReLU → DSConv → BN → ReLU → DSConv → Rain Mask
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 3
    ):
        """
        Initialize LSRB.
        
        Args:
            in_channels: Number of input channels (1 for thermal)
            hidden_channels: Intermediate channel count
            kernel_size: Convolution kernel size
            num_layers: Number of depthwise separable conv layers
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        layers = []
        
        # First layer: expand channels
        layers.extend([
            DepthwiseSeparableConv(in_channels, hidden_channels, kernel_size),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Middle layers
        for _ in range(num_layers - 2):
            layers.extend([
                DepthwiseSeparableConv(hidden_channels, hidden_channels, kernel_size),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            ])
        
        # Final layer: reduce to mask (same channels as input)
        layers.append(
            DepthwiseSeparableConv(hidden_channels, in_channels, kernel_size)
        )
        
        self.layers = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small weights to start with near-zero rain mask."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Initialize last layer to output near-zero mask
        last_conv = None
        for m in reversed(list(self.layers.modules())):
            if isinstance(m, nn.Conv2d):
                last_conv = m
                break
        if last_conv is not None:
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict rain mask and compute clean image.
        
        Args:
            x: Input rainy image (B, C, H, W)
        
        Returns:
            clean: Derained image (Input - Rain Mask)
            rain_mask: Predicted rain component
        """
        rain_mask = self.layers(x)
        clean = x - rain_mask
        
        # Clamp output to valid range
        clean = torch.clamp(clean, 0.0, 1.0)
        
        return clean, rain_mask


class CLAHE(nn.Module):
    """
    Contrast Limited Adaptive Histogram Equalization.
    
    Provides both OpenCV-based and pure PyTorch implementations.
    OpenCV is faster but requires CPU transfer. PyTorch version
    is differentiable and stays on GPU.
    """
    
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        use_opencv: bool = True
    ):
        """
        Initialize CLAHE.
        
        Args:
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for adaptive equalization
            use_opencv: Use OpenCV implementation if available
        """
        super().__init__()
        
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.use_opencv = use_opencv and OPENCV_AVAILABLE
        
        if self.use_opencv:
            # Create OpenCV CLAHE object
            self.clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=tile_grid_size
            )
    
    def _opencv_clahe(self, image: torch.Tensor) -> torch.Tensor:
        """Apply CLAHE using OpenCV (CPU-based)."""
        device = image.device
        batch_size = image.shape[0]
        
        # Convert to numpy
        images_np = image.detach().cpu().numpy()
        results = []
        
        for i in range(batch_size):
            img = images_np[i]
            
            # Handle channel dimension
            if img.ndim == 3:
                img = img.squeeze(0)  # Remove channel dim for CLAHE
            
            # Scale to 0-255 for OpenCV
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Apply CLAHE
            enhanced = self.clahe.apply(img_uint8)
            
            # Scale back to 0-1
            enhanced_float = enhanced.astype(np.float32) / 255.0
            results.append(enhanced_float)
        
        # Stack and convert back to tensor
        result = np.stack(results, axis=0)
        result = torch.from_numpy(result).to(device)
        
        # Add channel dimension if needed
        if result.ndim == 3:
            result = result.unsqueeze(1)
        
        return result
    
    def _torch_clahe(self, image: torch.Tensor) -> torch.Tensor:
        """
        Approximate CLAHE using pure PyTorch operations.
        
        This is a simplified version that provides similar effects
        but is differentiable and stays on GPU.
        """
        batch_size, channels, height, width = image.shape
        device = image.device
        
        # Number of tiles
        tiles_y, tiles_x = self.tile_grid_size
        tile_h = height // tiles_y
        tile_w = width // tiles_x
        
        # Process each image in batch
        results = []
        
        for b in range(batch_size):
            img = image[b, 0]  # Single channel
            enhanced = torch.zeros_like(img)
            
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Extract tile
                    y1 = ty * tile_h
                    y2 = min((ty + 1) * tile_h, height)
                    x1 = tx * tile_w
                    x2 = min((tx + 1) * tile_w, width)
                    
                    tile = img[y1:y2, x1:x2]
                    
                    # Compute histogram
                    flat = tile.flatten()
                    hist = torch.histc(flat, bins=256, min=0, max=1)
                    
                    # Clip histogram
                    clip_count = hist.sum() * self.clip_limit / 256
                    excess = torch.clamp(hist - clip_count, min=0).sum()
                    hist = torch.clamp(hist, max=clip_count)
                    hist = hist + excess / 256
                    
                    # Compute CDF
                    cdf = torch.cumsum(hist, dim=0)
                    cdf = cdf / cdf[-1]
                    
                    # Apply mapping
                    indices = (tile.flatten() * 255).long().clamp(0, 255)
                    mapped = cdf[indices].reshape(tile.shape)
                    
                    enhanced[y1:y2, x1:x2] = mapped
            
            results.append(enhanced)
        
        result = torch.stack(results, dim=0).unsqueeze(1)
        return result
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply CLAHE to image.
        
        Args:
            image: Input tensor (B, 1, H, W) normalized to [0, 1]
        
        Returns:
            CLAHE-enhanced tensor
        """
        if self.use_opencv:
            return self._opencv_clahe(image)
        else:
            return self._torch_clahe(image)


class RainRemover(nn.Module):
    """
    Complete rain removal module combining LSRB and CLAHE.
    
    Pipeline:
        Input → LSRB (Rain Mask Prediction) → Subtraction → CLAHE → Output
    
    The LSRB learns to predict rain streak patterns which are subtracted
    from the input. CLAHE then restores thermal contrast lost to rain.
    
    Example:
        >>> remover = RainRemover()
        >>> rainy_image = torch.rand(4, 1, 480, 640)
        >>> clean = remover(rainy_image)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        num_layers: int = 3,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8)
    ):
        """
        Initialize RainRemover.
        
        Args:
            in_channels: Number of input channels
            hidden_channels: LSRB hidden channel count
            num_layers: Number of LSRB layers
            apply_clahe: Whether to apply CLAHE after deraining
            clahe_clip_limit: CLAHE clip limit parameter
            clahe_tile_size: CLAHE tile grid size
        """
        super().__init__()
        
        self.lsrb = LSRB(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        self.apply_clahe = apply_clahe
        if apply_clahe:
            self.clahe = CLAHE(
                clip_limit=clahe_clip_limit,
                tile_grid_size=clahe_tile_size
            )
    
    def forward(
        self,
        image: torch.Tensor,
        return_mask: bool = False,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Remove rain from thermal image.
        
        Args:
            image: Input rainy image (B, 1, H, W)
            return_mask: If True, also return predicted rain mask
            return_intermediate: If True, return pre-CLAHE result too
        
        Returns:
            clean: Derained and enhanced image
            rain_mask: (optional) Predicted rain component
            pre_clahe: (optional) Image after deraining but before CLAHE
        """
        # Ensure proper input range
        input_max = image.max()
        if input_max > 1.0:
            image = image / input_max
            rescale = True
        else:
            rescale = False
        
        # LSRB: Predict and remove rain
        derained, rain_mask = self.lsrb(image)
        
        # CLAHE: Restore contrast
        if self.apply_clahe:
            clean = self.clahe(derained)
        else:
            clean = derained
        
        # Restore original scale
        if rescale:
            clean = clean * input_max
            derained = derained * input_max
            rain_mask = rain_mask * input_max
        
        # Build output
        outputs = [clean]
        if return_mask:
            outputs.append(rain_mask)
        if return_intermediate:
            outputs.append(derained)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def get_rain_mask(self, image: torch.Tensor) -> torch.Tensor:
        """Extract only the rain mask without full processing."""
        _, rain_mask = self.lsrb(image)
        return rain_mask


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("RainRemover Module Test")
    print("=" * 60)
    
    # Create synthetic rainy image with streak patterns
    batch_size = 2
    height, width = 480, 640
    
    # Base thermal image
    base_image = torch.rand(batch_size, 1, height, width) * 0.5 + 0.25
    
    # Add synthetic rain streaks (diagonal high-frequency patterns)
    rain_streaks = torch.zeros_like(base_image)
    for i in range(0, height, 5):
        for j in range(0, width, 3):
            if (i + j) % 7 == 0:
                rain_streaks[:, :, i:min(i+15, height), j:min(j+2, width)] = 0.3
    
    rainy_image = torch.clamp(base_image + rain_streaks, 0, 1)
    
    print(f"Input shape: {rainy_image.shape}")
    print(f"Input range: [{rainy_image.min():.3f}, {rainy_image.max():.3f}]")
    print(f"OpenCV available: {OPENCV_AVAILABLE}")
    
    # Initialize RainRemover
    remover = RainRemover(hidden_channels=32, num_layers=3)
    
    # Count parameters
    num_params = sum(p.numel() for p in remover.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Forward pass
    clean, rain_mask, pre_clahe = remover(
        rainy_image,
        return_mask=True,
        return_intermediate=True
    )
    
    print(f"\nOutput shape: {clean.shape}")
    print(f"Output range: [{clean.min():.3f}, {clean.max():.3f}]")
    print(f"Rain mask range: [{rain_mask.min():.3f}, {rain_mask.max():.3f}]")
    print(f"Rain mask mean: {rain_mask.abs().mean():.4f}")
    
    # Test CLAHE separately
    print("\n--- CLAHE Standalone Test ---")
    clahe = CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Low contrast image
    low_contrast = torch.rand(2, 1, 256, 256) * 0.3 + 0.35
    enhanced = clahe(low_contrast)
    
    print(f"Input contrast (std): {low_contrast.std():.4f}")
    print(f"Output contrast (std): {enhanced.std():.4f}")
    
    # Benchmark
    import time
    remover.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = remover(rainy_image)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            _ = remover(rainy_image)
        elapsed = (time.perf_counter() - start) / iterations * 1000
    
    print(f"\nLatency: {elapsed:.2f} ms per batch (batch={batch_size})")
