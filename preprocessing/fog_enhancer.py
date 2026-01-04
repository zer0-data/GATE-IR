"""
FogEnhancer: Adaptive Gamma Correction for Fog-Degraded Thermal Images

Component 2 (Stage B) of the GATE-IR pipeline.
Applies adaptive gamma correction to restore contrast in foggy thermal images.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class FogEnhancer(nn.Module):
    """
    Fog enhancement module using adaptive gamma correction.
    
    Fog causes:
    - Reduced contrast
    - Lower overall intensity
    - Homogenized temperature distributions
    
    Adaptive gamma correction adjusts brightness non-linearly:
        I_out = I_in ^ gamma
    
    Where gamma is computed based on the mean intensity:
        gamma = gamma_base + k * (mean - target_mean)
    
    Gamma correction behavior:
    - gamma < 1: Brightens the image (shadows lifted)
    - gamma > 1: Darkens the image (highlights compressed)
    
    Therefore:
    - Lower mean (dark/foggy) → gamma < 1 → brighten
    - Higher mean (bright) → gamma > 1 → darken
    
    Example:
        >>> enhancer = FogEnhancer()
        >>> foggy_image = torch.rand(4, 1, 480, 640) * 0.3  # Low intensity fog
        >>> enhanced = enhancer(foggy_image)
        >>> print(enhanced.mean())  # Should be closer to 0.5
    """
    
    def __init__(
        self,
        gamma_base: float = 1.0,
        gamma_min: float = 0.3,
        gamma_max: float = 3.0,
        target_mean: float = 0.5,
        adaptation_rate: float = 2.0,
        epsilon: float = 1e-8
    ):
        """
        Initialize FogEnhancer.
        
        Args:
            gamma_base: Base gamma value when mean equals target
            gamma_min: Minimum gamma (prevents over-brightening)
            gamma_max: Maximum gamma (prevents over-darkening)
            target_mean: Desired output mean intensity
            adaptation_rate: How aggressively to adjust gamma
            epsilon: Small value for numerical stability
        """
        super().__init__()
        
        self.gamma_base = gamma_base
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.target_mean = target_mean
        self.adaptation_rate = adaptation_rate
        self.epsilon = epsilon
    
    def compute_adaptive_gamma(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute per-image adaptive gamma values.
        
        Args:
            image: Normalized image tensor (B, C, H, W) in range [0, 1]
        
        Returns:
            Gamma tensor of shape (B, 1, 1, 1) for broadcasting
        """
        # Compute mean intensity per image
        # Keep dimensions for proper broadcasting
        mean_intensity = image.mean(dim=[1, 2, 3], keepdim=True)
        
        # Adaptive gamma formula:
        # gamma = base + rate * (mean - target)
        # 
        # Dark image (mean=0.2, target=0.5): gamma = 1.0 + 2.0*(-0.3) = 0.4 → brightens
        # Bright image (mean=0.8, target=0.5): gamma = 1.0 + 2.0*(0.3) = 1.6 → darkens
        gamma = self.gamma_base + self.adaptation_rate * (mean_intensity - self.target_mean)
        
        # Clamp to valid range
        gamma = torch.clamp(gamma, self.gamma_min, self.gamma_max)
        
        return gamma
    
    def apply_gamma(
        self,
        image: torch.Tensor,
        gamma: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply gamma correction to image.
        
        Args:
            image: Input image tensor in range [0, 1]
            gamma: Gamma values, broadcastable to image shape
        
        Returns:
            Gamma-corrected image in range [0, 1]
        """
        # Ensure positive values for power operation
        image = torch.clamp(image, self.epsilon, 1.0)
        
        # Apply gamma: I_out = I_in ^ gamma
        corrected = torch.pow(image, gamma)
        
        return corrected
    
    def forward(
        self,
        image: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
        return_gamma: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Apply adaptive fog enhancement.
        
        Args:
            image: Input thermal image (B, 1, H, W) or (B, H, W)
                   Should be normalized to [0, 1]
            gamma: Optional manual gamma override
            return_gamma: If True, also return computed gamma values
        
        Returns:
            enhanced: Enhanced image tensor
            gamma: (optional) Computed gamma values
        """
        # Ensure 4D tensor
        squeeze_output = False
        if image.dim() == 3:
            image = image.unsqueeze(1)
            squeeze_output = True
        
        # Normalize if needed (check if values > 1)
        input_max = image.max()
        if input_max > 1.0:
            image = image / input_max
            renormalize = True
        else:
            renormalize = False
        
        # Compute adaptive gamma if not provided
        if gamma is None:
            gamma = self.compute_adaptive_gamma(image)
        
        # Apply gamma correction
        enhanced = self.apply_gamma(image, gamma)
        
        # Restore original scale if needed
        if renormalize:
            enhanced = enhanced * input_max
        
        # Match input dimensionality
        if squeeze_output:
            enhanced = enhanced.squeeze(1)
        
        if return_gamma:
            return enhanced, gamma
        return enhanced
    
    def extra_repr(self) -> str:
        return (
            f"gamma_base={self.gamma_base}, "
            f"gamma_range=[{self.gamma_min}, {self.gamma_max}], "
            f"target_mean={self.target_mean}"
        )


class AdaptiveContrastEnhancer(nn.Module):
    """
    Extended fog enhancement with additional contrast stretching.
    
    Combines gamma correction with linear contrast stretching
    for more aggressive fog removal.
    """
    
    def __init__(
        self,
        gamma_enhancer: Optional[FogEnhancer] = None,
        contrast_percentile: float = 2.0
    ):
        """
        Initialize AdaptiveContrastEnhancer.
        
        Args:
            gamma_enhancer: FogEnhancer instance (created if None)
            contrast_percentile: Percentile for contrast stretching
        """
        super().__init__()
        
        self.gamma_enhancer = gamma_enhancer or FogEnhancer()
        self.contrast_percentile = contrast_percentile
    
    def contrast_stretch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply percentile-based contrast stretching.
        
        Uses torch.quantile for O(N) performance instead of O(N log N) sorting.
        Fully vectorized across batch dimension.
        
        Args:
            image: Input tensor (B, C, H, W)
        
        Returns:
            Contrast-stretched tensor
        """
        batch_size = image.shape[0]
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, C*H*W)
        flat = image.view(batch_size, -1)
        
        # Compute percentiles using quantile (much faster than sort)
        # quantile is O(N) vs O(N log N) for sort
        low_q = self.contrast_percentile / 100.0
        high_q = 1.0 - low_q
        
        # Vectorized quantile across batch - shape: (B,)
        low_val = torch.quantile(flat, low_q, dim=1, keepdim=True)
        high_val = torch.quantile(flat, high_q, dim=1, keepdim=True)
        
        # Reshape for broadcasting: (B, 1) -> (B, 1, 1, 1)
        low_val = low_val.view(batch_size, 1, 1, 1)
        high_val = high_val.view(batch_size, 1, 1, 1)
        
        # Avoid division by zero
        range_val = high_val - low_val
        range_val = torch.where(range_val > 0, range_val, torch.ones_like(range_val))
        
        # Vectorized contrast stretch
        stretched = (image - low_val) / range_val
        stretched = torch.clamp(stretched, 0.0, 1.0)
        
        return stretched
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply combined gamma correction and contrast stretching."""
        # First apply gamma correction
        enhanced = self.gamma_enhancer(image)
        
        # Then apply contrast stretching
        enhanced = self.contrast_stretch(enhanced)
        
        return enhanced


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("FogEnhancer Module Test")
    print("=" * 60)
    
    # Simulate foggy thermal image (low contrast, shifted dark)
    batch_size = 4
    foggy_images = torch.rand(batch_size, 1, 480, 640) * 0.3 + 0.1  # Range [0.1, 0.4]
    
    print(f"Input shape: {foggy_images.shape}")
    print(f"Input mean: {foggy_images.mean():.3f}")
    print(f"Input range: [{foggy_images.min():.3f}, {foggy_images.max():.3f}]")
    
    # Create enhancer
    enhancer = FogEnhancer()
    print(f"\nEnhancer config: {enhancer.extra_repr()}")
    
    # Apply enhancement
    enhanced, gamma = enhancer(foggy_images, return_gamma=True)
    
    print(f"\nOutput mean: {enhanced.mean():.3f}")
    print(f"Output range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    print(f"Computed gamma values: {gamma.squeeze().tolist()}")
    
    # Test with normal image (should have minimal effect)
    print("\n--- Normal Image Test ---")
    normal_images = torch.rand(batch_size, 1, 480, 640) * 0.6 + 0.2  # Range [0.2, 0.8]
    enhanced_normal, gamma_normal = enhancer(normal_images, return_gamma=True)
    
    print(f"Input mean: {normal_images.mean():.3f}")
    print(f"Output mean: {enhanced_normal.mean():.3f}")
    print(f"Gamma (should be ~1.0): {gamma_normal.squeeze().tolist()}")
