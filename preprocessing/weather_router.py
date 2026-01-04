"""
WeatherRouter: Conditional Preprocessing Router for Thermal Images

Component 2 (Stage B) of the GATE-IR pipeline.
Routes images through weather-specific preprocessing paths based on
classification results from WeatherGate.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union

from .fog_enhancer import FogEnhancer
from .rain_remover import RainRemover


class PassThrough(nn.Module):
    """
    Identity module for clear weather conditions.
    
    Simply returns the input unchanged. Used for consistency
    in the routing pipeline.
    """
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class WeatherRouter(nn.Module):
    """
    Conditional router for weather-specific preprocessing.
    
    Routes thermal images through appropriate preprocessing modules
    based on detected weather conditions:
        - Clear (class_id=0): PassThrough (no modification)
        - Fog (class_id=1): FogEnhancer (gamma correction)
        - Rain (class_id=2): RainRemover (LSRB + CLAHE)
    
    Supports both per-image routing (heterogeneous batches) and
    batch-level routing (homogeneous batches for efficiency).
    
    Example:
        >>> from gate.weather_gate import WeatherGate
        >>> 
        >>> gate = WeatherGate()
        >>> router = WeatherRouter()
        >>> 
        >>> thermal_batch = torch.rand(8, 1, 480, 640)
        >>> class_ids = gate(thermal_batch)
        >>> processed = router(thermal_batch, class_ids)
    """
    
    # Weather class definitions
    CLEAR = 0
    FOG = 1
    RAIN = 2
    
    CLASS_NAMES = {
        CLEAR: "Clear",
        FOG: "Fog", 
        RAIN: "Rain"
    }
    
    def __init__(
        self,
        fog_enhancer: Optional[FogEnhancer] = None,
        rain_remover: Optional[RainRemover] = None,
        fog_config: Optional[Dict] = None,
        rain_config: Optional[Dict] = None,
        normalize_input: bool = True,
        bit_depth: int = 14
    ):
        """
        Initialize WeatherRouter.
        
        Args:
            fog_enhancer: Pre-configured FogEnhancer (created if None)
            rain_remover: Pre-configured RainRemover (created if None)
            fog_config: Config dict for FogEnhancer if creating new
            rain_config: Config dict for RainRemover if creating new
            normalize_input: Whether to auto-normalize raw input to [0, 1]
            bit_depth: Bit depth for normalization (14-bit = divide by 16383)
        """
        super().__init__()
        
        self.normalize_input = normalize_input
        self.bit_depth = bit_depth
        self.max_value = (2 ** bit_depth) - 1  # 16383 for 14-bit
        
        # Initialize preprocessing modules
        self.passthrough = PassThrough()
        
        if fog_enhancer is not None:
            self.fog_enhancer = fog_enhancer
        else:
            fog_config = fog_config or {}
            self.fog_enhancer = FogEnhancer(**fog_config)
        
        if rain_remover is not None:
            self.rain_remover = rain_remover
        else:
            rain_config = rain_config or {}
            self.rain_remover = RainRemover(**rain_config)
        
        # Module mapping
        self.modules_map = nn.ModuleDict({
            "clear": self.passthrough,
            "fog": self.fog_enhancer,
            "rain": self.rain_remover
        })
    
    def _normalize(
        self, 
        images: torch.Tensor, 
        is_normalized: Optional[bool] = None
    ) -> Tuple[torch.Tensor, bool]:
        """
        Normalize images to [0, 1] range if needed.
        
        Returns:
            normalized_images: Images in [0, 1] range
            was_normalized: Whether normalization was applied (for rescaling output)
        """
        if is_normalized is True:
            return images, False
        
        if is_normalized is False:
            # Explicitly raw - normalize
            return images.float() / self.max_value, True
        
        # Legacy: infer from max value
        if self.normalize_input and images.max() > 1.0:
            return images.float() / images.max(), True
        
        return images, False
    
    def route_single(
        self,
        image: torch.Tensor,
        class_id: int,
        is_normalized: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Route a single image through appropriate preprocessing.
        
        Args:
            image: Single thermal image (1, H, W) or (H, W)
            class_id: Weather class (0=Clear, 1=Fog, 2=Rain)
            is_normalized: Whether input is already normalized to [0, 1]
        
        Returns:
            Preprocessed image
        """
        # Ensure 4D tensor for processing
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Normalize if needed
        image, was_normalized = self._normalize(image, is_normalized)
        
        # Route to appropriate module
        if class_id == self.CLEAR:
            result = self.passthrough(image)
        elif class_id == self.FOG:
            result = self.fog_enhancer(image)
        elif class_id == self.RAIN:
            # Pass is_normalized=True since we already normalized
            result = self.rain_remover(image, is_normalized=True)
        else:
            raise ValueError(f"Unknown class_id: {class_id}")
        
        return result.squeeze(0)
    
    def route_batch_homogeneous(
        self,
        images: torch.Tensor,
        class_id: int,
        is_normalized: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Route entire batch through same preprocessing (efficient).
        
        Use when all images in batch have same weather condition.
        
        Args:
            images: Batch of thermal images (B, 1, H, W)
            class_id: Single weather class for entire batch
            is_normalized: Whether input is already normalized to [0, 1]
        
        Returns:
            Preprocessed batch
        """
        # Normalize if needed
        images, was_normalized = self._normalize(images, is_normalized)
        
        if class_id == self.CLEAR:
            return self.passthrough(images)
        elif class_id == self.FOG:
            return self.fog_enhancer(images)
        elif class_id == self.RAIN:
            # Pass is_normalized=True since we already normalized above
            return self.rain_remover(images, is_normalized=True)
        else:
            raise ValueError(f"Unknown class_id: {class_id}")
    
    def route_batch_heterogeneous(
        self,
        images: torch.Tensor,
        class_ids: torch.Tensor,
        is_normalized: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Route batch with mixed weather conditions (per-image).
        
        Args:
            images: Batch of thermal images (B, 1, H, W)
            class_ids: Per-image class IDs, shape (B,)
            is_normalized: Whether input is already normalized to [0, 1]
        
        Returns:
            Preprocessed batch
        """
        batch_size = images.shape[0]
        results = []
        
        for i in range(batch_size):
            image = images[i:i+1]  # Keep batch dim
            class_id = class_ids[i].item()
            
            processed = self.route_batch_homogeneous(image, class_id, is_normalized=is_normalized)
            results.append(processed)
        
        return torch.cat(results, dim=0)
    
    def route_batch_optimized(
        self,
        images: torch.Tensor,
        class_ids: torch.Tensor,
        is_normalized: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Optimized batch routing using masked operations.
        
        Groups images by class and processes each group in batch,
        then reassembles results in original order.
        
        Args:
            images: Batch of thermal images (B, 1, H, W)
            class_ids: Per-image class IDs, shape (B,)
            is_normalized: Whether input is already normalized to [0, 1]
        
        Returns:
            Preprocessed batch
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Create output tensor - MUST be float to handle preprocessor outputs
        # even if input images are integer type (e.g., raw 14-bit as int16)
        output = torch.zeros_like(images, dtype=torch.float32)
        
        # Process each weather class in batch
        for class_id in [self.CLEAR, self.FOG, self.RAIN]:
            # Find images belonging to this class
            mask = (class_ids == class_id)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Extract and process
            class_images = images[indices]
            processed = self.route_batch_homogeneous(class_images, class_id, is_normalized=is_normalized)
            
            # Place results back
            output[indices] = processed
        
        return output
    
    def forward(
        self,
        images: torch.Tensor,
        class_ids: Union[torch.Tensor, int],
        optimize: bool = True,
        is_normalized: Optional[bool] = None
    ) -> torch.Tensor:
        """
        Route images through weather-specific preprocessing.
        
        Args:
            images: Batch of thermal images (B, 1, H, W) or (B, H, W)
            class_ids: Weather class(es) - single int or tensor of shape (B,)
            optimize: Use optimized batched routing
            is_normalized: Whether input is already normalized to [0, 1]
                          True=normalized, False=raw, None=infer from values
        
        Returns:
            Preprocessed images
        """
        # Ensure 4D tensor
        squeeze_output = False
        if images.dim() == 3:
            images = images.unsqueeze(1)
            squeeze_output = True
        
        # Handle single class_id for entire batch
        if isinstance(class_ids, int):
            result = self.route_batch_homogeneous(images, class_ids, is_normalized=is_normalized)
        elif isinstance(class_ids, torch.Tensor):
            # Check if all same class
            unique_classes = torch.unique(class_ids)
            if len(unique_classes) == 1:
                result = self.route_batch_homogeneous(images, unique_classes[0].item(), is_normalized=is_normalized)
            elif optimize:
                result = self.route_batch_optimized(images, class_ids, is_normalized=is_normalized)
            else:
                result = self.route_batch_heterogeneous(images, class_ids, is_normalized=is_normalized)
        else:
            raise TypeError(f"class_ids must be int or Tensor, got {type(class_ids)}")
        
        if squeeze_output:
            result = result.squeeze(1)
        
        return result
    
    def get_stats(self, class_ids: torch.Tensor) -> Dict[str, int]:
        """Get statistics on class distribution in batch."""
        stats = {}
        for class_id, name in self.CLASS_NAMES.items():
            count = (class_ids == class_id).sum().item()
            stats[name] = count
        return stats


class WeatherPipeline(nn.Module):
    """
    Complete weather-aware preprocessing pipeline.
    
    Combines WeatherGate classification with WeatherRouter preprocessing
    into a single end-to-end module.
    
    Example:
        >>> pipeline = WeatherPipeline()
        >>> thermal_batch = torch.rand(8, 1, 480, 640)
        >>> processed, class_ids = pipeline(thermal_batch)
    """
    
    def __init__(
        self,
        gate_config: Optional[Dict] = None,
        router_config: Optional[Dict] = None
    ):
        """
        Initialize WeatherPipeline.
        
        Args:
            gate_config: Config dict for WeatherGate
            router_config: Config dict for WeatherRouter
        """
        super().__init__()
        
        # Import here to avoid circular imports
        from gate.weather_gate import WeatherGate
        
        gate_config = gate_config or {}
        router_config = router_config or {}
        
        self.gate = WeatherGate(**gate_config)
        self.router = WeatherRouter(**router_config)
    
    def forward(
        self,
        images: torch.Tensor,
        return_classes: bool = True,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Process images through complete weather pipeline.
        
        Args:
            images: Batch of thermal images
            return_classes: Return predicted class IDs
            return_probs: Return class probabilities
        
        Returns:
            processed: Preprocessed images
            class_ids: (optional) Predicted weather classes
            probs: (optional) Class probabilities
        """
        # Classify weather conditions
        if return_probs:
            class_ids, probs = self.gate(images, return_probs=True)
        else:
            class_ids = self.gate(images)
            probs = None
        
        # Route through preprocessing
        processed = self.router(images, class_ids)
        
        # Build output
        outputs = [processed]
        if return_classes:
            outputs.append(class_ids)
        if return_probs:
            outputs.append(probs)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("WeatherRouter Module Test")
    print("=" * 60)
    
    # Create test images
    batch_size = 8
    height, width = 480, 640
    
    # Mix of image types
    images = torch.rand(batch_size, 1, height, width)
    
    # Simulate different weather conditions
    # Class 0: Clear, Class 1: Fog, Class 2: Rain
    class_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    
    print(f"Input shape: {images.shape}")
    print(f"Class distribution: {dict(zip(*torch.unique(class_ids, return_counts=True)))}")
    
    # Initialize router
    router = WeatherRouter()
    
    # Count parameters
    num_params = sum(p.numel() for p in router.parameters())
    print(f"Router parameters: {num_params:,}")
    
    # Test routing
    processed = router(images, class_ids)
    
    print(f"Output shape: {processed.shape}")
    print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Get stats
    stats = router.get_stats(class_ids)
    print(f"Routing stats: {stats}")
    
    # Test homogeneous batch (all same class)
    print("\n--- Homogeneous Batch Test (All Fog) ---")
    fog_class = torch.ones(batch_size, dtype=torch.long)
    processed_fog = router(images, fog_class)
    print(f"Fog processing output range: [{processed_fog.min():.3f}, {processed_fog.max():.3f}]")
    
    # Benchmark
    import time
    router.eval()
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = router(images, class_ids)
        
        # Benchmark heterogeneous
        start = time.perf_counter()
        iterations = 20
        for _ in range(iterations):
            _ = router(images, class_ids)
        elapsed = (time.perf_counter() - start) / iterations * 1000
        print(f"\nHeterogeneous batch latency: {elapsed:.2f} ms")
        
        # Benchmark homogeneous (should be faster)
        start = time.perf_counter()
        for _ in range(iterations):
            _ = router(images, 0)  # All clear
        elapsed = (time.perf_counter() - start) / iterations * 1000
        print(f"Homogeneous batch latency: {elapsed:.2f} ms")
