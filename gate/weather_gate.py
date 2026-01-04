"""
WeatherGate: Low-Latency Weather Classification for Thermal Images

Component 1 (Stage A) of the GATE-IR pipeline.
Classifies 14-bit RAW thermal images into Clear, Fog, or Rain conditions
using efficient statistical feature extraction and a lightweight MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WeatherGate(nn.Module):
    """
    Low-latency weather classifier for 14-bit RAW thermal images.
    
    Routes images through weather-specific preprocessing paths based on
    detected conditions (Clear, Fog, Rain).
    
    Architecture:
        Input (5 features) → Linear(64) → ReLU → Linear(64) → ReLU → Linear(3)
    
    Features Extracted:
        1. Thermal Variance - Overall intensity variation
        2. Min Value - Lower bound of dynamic range
        3. Max Value - Upper bound of dynamic range  
        4. Entropy - Texture complexity measure
        5. Laplacian Variance - Edge sharpness (fog vs rain proxy)
    
    Example:
        >>> gate = WeatherGate()
        >>> thermal_batch = torch.randn(4, 1, 480, 640)  # B, C, H, W
        >>> class_ids = gate(thermal_batch)  # Returns tensor of shape (4,)
        >>> print(class_ids)  # tensor([0, 1, 2, 0]) - Clear, Fog, Rain, Clear
    """
    
    # Class labels for interpretability
    CLASSES = {0: "Clear", 1: "Fog", 2: "Rain"}
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_classes: int = 3,
        entropy_bins: int = 256,
        normalize_input: bool = True,
        bit_depth: int = 14
    ):
        """
        Initialize WeatherGate classifier.
        
        Args:
            input_dim: Number of input features (default: 5)
            hidden_dim: Hidden layer dimension (default: 64 for speed)
            num_classes: Number of weather classes (default: 3)
            entropy_bins: Number of bins for entropy calculation (default: 256)
            normalize_input: Whether to normalize 14-bit input to [0, 1]
            bit_depth: Bit depth of input thermal images (default: 14)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.entropy_bins = entropy_bins
        self.normalize_input = normalize_input
        self.bit_depth = bit_depth
        self.max_value = (2 ** bit_depth) - 1  # 16383 for 14-bit
        
        # Laplacian kernel for edge detection (registered as buffer, not parameter)
        laplacian_kernel = torch.tensor([
            [0.,  1., 0.],
            [1., -4., 1.],
            [0.,  1., 0.]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
        # 3-layer MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from thermal images efficiently.
        
        Computes 5 scalar features per image using vectorized PyTorch operations:
        1. Thermal Variance - torch.var()
        2. Min Value - torch.amin()
        3. Max Value - torch.amax()
        4. Entropy - Histogram-based calculation
        5. Laplacian Variance - Convolution-based edge metric
        
        Args:
            image: Batch of thermal images, shape (B, 1, H, W) or (B, H, W)
                   Values should be in range [0, max_value] for 14-bit
        
        Returns:
            Feature tensor of shape (B, 5)
        """
        # Ensure 4D tensor: (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(1)
        
        batch_size = image.shape[0]
        device = image.device
        
        # Normalize to [0, 1] if needed
        if self.normalize_input:
            # Handle both 14-bit raw and pre-normalized inputs
            if image.max() > 1.0:
                image = image.float() / self.max_value
        
        # Flatten spatial dimensions for per-image statistics: (B, H*W)
        flat = image.view(batch_size, -1)
        
        # Feature 1: Thermal Variance
        variance = torch.var(flat, dim=1)
        
        # Feature 2 & 3: Min and Max
        min_val = torch.amin(flat, dim=1)
        max_val = torch.amax(flat, dim=1)
        
        # Feature 4: Entropy (histogram-based)
        entropy = self._compute_entropy_batch(flat)
        
        # Feature 5: Laplacian Variance (edge sharpness)
        laplacian_var = self._compute_laplacian_variance(image)
        
        # Stack features: (B, 5)
        features = torch.stack([
            variance,
            min_val,
            max_val,
            entropy,
            laplacian_var
        ], dim=1)
        
        return features
    
    def _compute_entropy_batch(self, flat: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy for a batch of flattened images.
        
        Uses differentiable soft histogram for GPU-friendly computation.
        
        Args:
            flat: Flattened image tensor of shape (B, N) where N = H * W
                  Values can be raw (0 to max_value) or normalized [0, 1]
        
        Returns:
            Entropy tensor of shape (B,)
        """
        batch_size = flat.shape[0]
        device = flat.device
        
        # CRITICAL: Normalize to [0, 1] based on actual data range BEFORE clamping
        # This prevents raw 14-bit data (0-16383) from being clamped to all 1.0
        flat_min = flat.amin(dim=1, keepdim=True)
        flat_max = flat.amax(dim=1, keepdim=True)
        
        # Avoid division by zero for constant images
        flat_range = flat_max - flat_min
        flat_range = torch.where(flat_range == 0, torch.ones_like(flat_range), flat_range)
        
        # Min-max normalize each image to [0, 1]
        flat_normalized = (flat - flat_min) / flat_range
        
        # Now safe to clamp (should already be [0, 1], this is defensive)
        flat_normalized = torch.clamp(flat_normalized, 0.0, 1.0)
        
        # Compute histogram for each image in batch
        # Using soft binning for differentiability
        bin_edges = torch.linspace(0, 1, self.entropy_bins + 1, device=device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / self.entropy_bins
        
        # Expand for broadcasting: (B, N, 1) and (1, 1, bins)
        flat_expanded = flat_normalized.unsqueeze(-1)  # (B, N, 1)
        centers_expanded = bin_centers.view(1, 1, -1)  # (1, 1, bins)
        
        # Soft histogram: count contributions to each bin
        # Using triangular kernel for soft assignment
        distances = torch.abs(flat_expanded - centers_expanded)  # (B, N, bins)
        weights = torch.clamp(1 - distances / bin_width, min=0)  # (B, N, bins)
        hist = weights.sum(dim=1)  # (B, bins)
        
        # Normalize to probability distribution
        hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-10)
        
        # Compute entropy: H = -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(hist * torch.log2(hist + 1e-10), dim=1)
        
        # Normalize by max entropy (log2 of number of bins)
        max_entropy = torch.log2(torch.tensor(self.entropy_bins, dtype=torch.float32, device=device))
        entropy = entropy / max_entropy
        
        return entropy
    
    def _compute_laplacian_variance(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian variance as an edge sharpness metric.
        
        High variance = sharp edges (clear or rain streaks)
        Low variance = blurred edges (fog)
        
        Args:
            image: Image tensor of shape (B, 1, H, W)
        
        Returns:
            Laplacian variance tensor of shape (B,)
        """
        # Apply Laplacian convolution
        # padding='same' equivalent with manual padding
        laplacian = F.conv2d(
            image,
            self.laplacian_kernel,
            padding=1
        )
        
        # Compute variance of Laplacian response per image
        batch_size = laplacian.shape[0]
        flat_laplacian = laplacian.view(batch_size, -1)
        lap_var = torch.var(flat_laplacian, dim=1)
        
        return lap_var
    
    def forward(
        self,
        image: torch.Tensor,
        return_probs: bool = False,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Classify weather conditions from thermal images.
        
        Args:
            image: Batch of thermal images, shape (B, 1, H, W) or (B, H, W)
            return_probs: If True, also return class probabilities
            return_features: If True, also return extracted features
        
        Returns:
            class_ids: Predicted class indices, shape (B,)
                       0 = Clear, 1 = Fog, 2 = Rain
            probs: (optional) Class probabilities, shape (B, 3)
            features: (optional) Extracted features, shape (B, 5)
        """
        # Extract features
        features = self.extract_features(image)
        
        # Forward through MLP
        logits = self.classifier(features)
        
        # Get class predictions
        probs = F.softmax(logits, dim=1)
        class_ids = torch.argmax(probs, dim=1)
        
        # Build output tuple based on flags
        outputs = [class_ids]
        if return_probs:
            outputs.append(probs)
        if return_features:
            outputs.append(features)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def get_class_name(self, class_id: int) -> str:
        """Get human-readable class name from class ID."""
        return self.CLASSES.get(class_id, "Unknown")
    
    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference-optimized prediction (no gradient computation).
        
        Args:
            image: Batch of thermal images
        
        Returns:
            class_ids: Predicted class indices
            confidence: Confidence scores for predictions
        """
        self.eval()
        class_ids, probs = self.forward(image, return_probs=True)
        confidence = probs.gather(1, class_ids.unsqueeze(1)).squeeze(1)
        return class_ids, confidence


def extract_features(image: torch.Tensor, bit_depth: int = 14) -> torch.Tensor:
    """
    Standalone feature extraction function.
    
    Convenience function for extracting features without instantiating
    the full WeatherGate module.
    
    Args:
        image: Batch of thermal images, shape (B, 1, H, W) or (B, H, W)
        bit_depth: Bit depth of input images
    
    Returns:
        Feature tensor of shape (B, 5)
    """
    gate = WeatherGate(bit_depth=bit_depth)
    gate.eval()
    
    # Move to same device as input
    gate = gate.to(image.device)
    
    with torch.no_grad():
        features = gate.extract_features(image)
    
    return features


# Example usage and testing
if __name__ == "__main__":
    # Create sample 14-bit thermal images
    batch_size = 4
    height, width = 480, 640
    
    # Simulate 14-bit data (0 to 16383)
    thermal_batch = torch.randint(0, 16384, (batch_size, 1, height, width), dtype=torch.float32)
    
    print("=" * 60)
    print("WeatherGate Module Test")
    print("=" * 60)
    print(f"Input shape: {thermal_batch.shape}")
    print(f"Input range: [{thermal_batch.min():.0f}, {thermal_batch.max():.0f}]")
    
    # Initialize model
    gate = WeatherGate()
    print(f"\nModel parameters: {sum(p.numel() for p in gate.parameters()):,}")
    
    # Forward pass
    class_ids, probs, features = gate(thermal_batch, return_probs=True, return_features=True)
    
    print(f"\nOutput:")
    print(f"  Class IDs shape: {class_ids.shape}")
    print(f"  Probabilities shape: {probs.shape}")
    print(f"  Features shape: {features.shape}")
    
    print(f"\nPredictions:")
    for i in range(batch_size):
        class_name = gate.get_class_name(class_ids[i].item())
        conf = probs[i, class_ids[i]].item()
        print(f"  Image {i}: {class_name} (confidence: {conf:.3f})")
    
    print(f"\nFeature breakdown (first image):")
    feature_names = ["Variance", "Min", "Max", "Entropy", "Laplacian Var"]
    for name, val in zip(feature_names, features[0]):
        print(f"  {name}: {val.item():.6f}")
    
    # Benchmark latency
    import time
    gate.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = gate(thermal_batch)
        
        # Benchmark
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = gate(thermal_batch)
        elapsed = (time.perf_counter() - start) / iterations * 1000
        
    print(f"\nLatency: {elapsed:.2f} ms per batch (batch_size={batch_size})")
    print(f"Throughput: {batch_size / elapsed * 1000:.1f} images/sec")
