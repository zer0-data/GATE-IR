"""
GATE-IR: Main Entry Point

Complete pipeline demonstration for the Gated Adaptive Thermal Enhancement
system for Infrared Detection.

Usage:
    python main.py --mode demo
    python main.py --mode test
    python main.py --mode inference --image path/to/thermal.png
"""

import argparse
import os
import sys
import torch
import time
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_components():
    """Test all GATE-IR components."""
    print("=" * 70)
    print("GATE-IR Component Tests")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")
    
    # Test configuration
    batch_size = 2
    height, width = 480, 640
    
    # Create test thermal image (14-bit simulation)
    thermal_image = torch.randint(0, 16384, (batch_size, 1, height, width), dtype=torch.float32)
    thermal_normalized = thermal_image / 16383.0  # Normalize to [0, 1]
    
    print("-" * 70)
    print("Component 1: WeatherGate (Stage A)")
    print("-" * 70)
    
    from gate.weather_gate import WeatherGate
    
    gate = WeatherGate().to(device)
    gate_params = sum(p.numel() for p in gate.parameters())
    print(f"  Parameters: {gate_params:,}")
    
    # Forward pass
    thermal_gpu = thermal_image.to(device)
    start = time.perf_counter()
    class_ids, probs, features = gate(thermal_gpu, return_probs=True, return_features=True)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"  Input shape: {thermal_image.shape}")
    print(f"  Output classes: {class_ids.cpu().tolist()}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  ✓ WeatherGate test passed")
    
    print("\n" + "-" * 70)
    print("Component 2: Preprocessing Modules (Stage B)")
    print("-" * 70)
    
    from preprocessing import FogEnhancer, RainRemover, WeatherRouter
    
    # Test FogEnhancer
    fog_enhancer = FogEnhancer()
    fog_input = thermal_normalized.to(device) * 0.3 + 0.1  # Simulate foggy (dark)
    fog_output = fog_enhancer(fog_input)
    print(f"  FogEnhancer:")
    print(f"    Input mean: {fog_input.mean():.3f}")
    print(f"    Output mean: {fog_output.mean():.3f}")
    
    # Test RainRemover
    rain_remover = RainRemover().to(device)
    rain_params = sum(p.numel() for p in rain_remover.parameters())
    rain_input = thermal_normalized.to(device)
    rain_output = rain_remover(rain_input)
    print(f"  RainRemover:")
    print(f"    Parameters: {rain_params:,}")
    print(f"    Output shape: {rain_output.shape}")
    
    # Test WeatherRouter
    router = WeatherRouter().to(device)
    router_input = thermal_normalized.to(device)
    test_classes = torch.tensor([0, 1], device=device)  # Clear, Fog
    routed = router(router_input, test_classes)
    print(f"  WeatherRouter:")
    print(f"    Input: {test_classes.cpu().tolist()} (Clear, Fog)")
    print(f"    Output shape: {routed.shape}")
    print(f"  ✓ Preprocessing tests passed")
    
    print("\n" + "-" * 70)
    print("Component 3: YOLOv8-Thermal (Stage C)")
    print("-" * 70)
    
    from models.yolov8_thermal import yolov8s_thermal
    
    detector = yolov8s_thermal(
        num_classes=3,
        include_p2=True,
        use_transformer_neck=True
    ).to(device)
    detector_params = sum(p.numel() for p in detector.parameters())
    print(f"  Parameters: {detector_params:,}")
    
    # Forward pass with 640x640 input
    detector_input = torch.rand(batch_size, 1, 640, 640, device=device)
    
    start = time.perf_counter()
    detector.eval()
    with torch.no_grad():
        output = detector(detector_input, return_features=True)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"  Input shape: {detector_input.shape}")
    print(f"  Feature scales:")
    for name, feat in output['fused_features'].items():
        print(f"    {name}: {feat.shape}")
    print(f"  Detection heads:")
    for name, preds in output['predictions'].items():
        print(f"    {name}: cls={preds['cls'].shape}, reg={preds['reg'].shape}")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  ✓ YOLOv8-Thermal test passed")
    
    print("\n" + "-" * 70)
    print("Component 4: CycleGAN (IR → Pseudo-RGB)")
    print("-" * 70)
    
    from training.cyclegan import CycleGAN, convert_ir_to_pseudo_rgb
    
    cyclegan = CycleGAN(
        ir_channels=1,
        rgb_channels=3,
        ngf=64,
        n_residual_blocks=6  # Smaller for testing
    ).to(device)
    cyclegan_params = sum(p.numel() for p in cyclegan.parameters())
    print(f"  Parameters: {cyclegan_params:,}")
    
    # Generate pseudo-RGB
    ir_input = (thermal_normalized.to(device) * 2) - 1  # Scale to [-1, 1]
    
    start = time.perf_counter()
    cyclegan.eval()
    with torch.no_grad():
        pseudo_rgb = cyclegan.generate_pseudo_rgb(ir_input)
    latency = (time.perf_counter() - start) * 1000
    
    print(f"  IR input shape: {ir_input.shape}")
    print(f"  Pseudo-RGB shape: {pseudo_rgb.shape}")
    print(f"  Pseudo-RGB range: [{pseudo_rgb.min():.3f}, {pseudo_rgb.max():.3f}]")
    print(f"  Latency: {latency:.2f} ms")
    print(f"  ✓ CycleGAN test passed")
    
    print("\n" + "-" * 70)
    print("Component 5: Knowledge Distillation")
    print("-" * 70)
    
    from training.distillation import FeatureMimicLoss, ChannelAdapter
    
    # Create mock features for testing
    student_channels = {'P3': 128, 'P4': 256}
    teacher_channels = {'P3': 256, 'P4': 512}
    
    feature_loss = FeatureMimicLoss(
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        layers=['P3', 'P4']
    ).to(device)
    
    adapter_params = sum(p.numel() for p in feature_loss.parameters())
    
    # Mock features
    student_feats = {
        'P3': torch.rand(batch_size, 128, 80, 80, device=device),
        'P4': torch.rand(batch_size, 256, 40, 40, device=device)
    }
    teacher_feats = {
        'P3': torch.rand(batch_size, 256, 80, 80, device=device),
        'P4': torch.rand(batch_size, 512, 40, 40, device=device)
    }
    
    loss, layer_losses = feature_loss(student_feats, teacher_feats)
    
    print(f"  Adapter parameters: {adapter_params:,}")
    print(f"  Total feature loss: {loss.item():.4f}")
    for layer, l in layer_losses.items():
        print(f"    {layer}: {l.item():.4f}")
    print(f"  ✓ Distillation test passed")
    
    print("\n" + "=" * 70)
    print("All Component Tests Passed!")
    print("=" * 70)
    
    # Summary
    total_params = gate_params + rain_params + detector_params + cyclegan_params + adapter_params
    print(f"\nTotal System Parameters: {total_params:,}")
    print(f"  WeatherGate: {gate_params:,}")
    print(f"  RainRemover: {rain_params:,}")
    print(f"  YOLOv8-Thermal: {detector_params:,}")
    print(f"  CycleGAN: {cyclegan_params:,}")
    print(f"  Distillation Adapters: {adapter_params:,}")


def demo_pipeline():
    """Demonstrate the complete GATE-IR pipeline."""
    print("=" * 70)
    print("GATE-IR Pipeline Demonstration")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load components
    print("\n1. Loading pipeline components...")
    
    from gate.weather_gate import WeatherGate
    from preprocessing.weather_router import WeatherRouter
    from models.yolov8_thermal import yolov8s_thermal
    
    gate = WeatherGate().to(device).eval()
    router = WeatherRouter().to(device).eval()
    detector = yolov8s_thermal(num_classes=3, include_p2=True).to(device).eval()
    
    print("   ✓ Components loaded")
    
    # Create synthetic thermal images
    print("\n2. Creating synthetic thermal images...")
    
    batch_size = 4
    thermal_batch = torch.rand(batch_size, 1, 640, 640, device=device)
    
    # Simulate different weather conditions
    thermal_batch[0] *= 0.3  # Dark (fog-like)
    thermal_batch[1] += torch.rand_like(thermal_batch[1]) * 0.5  # High variance
    # thermal_batch[2] and [3] are normal
    
    print(f"   Input shape: {thermal_batch.shape}")
    
    # Stage A: Weather Classification
    print("\n3. Stage A: Weather Classification (WeatherGate)...")
    
    with torch.no_grad():
        class_ids, probs = gate(thermal_batch, return_probs=True)
    
    for i in range(batch_size):
        class_name = gate.get_class_name(class_ids[i].item())
        conf = probs[i, class_ids[i]].item()
        print(f"   Image {i}: {class_name} (confidence: {conf:.2%})")
    
    # Stage B: Weather-Specific Preprocessing
    print("\n4. Stage B: Weather-Specific Preprocessing (WeatherRouter)...")
    
    with torch.no_grad():
        processed = router(thermal_batch, class_ids)
    
    print(f"   Processed shape: {processed.shape}")
    print(f"   Routing summary: {router.get_stats(class_ids)}")
    
    # Stage C: Object Detection
    print("\n5. Stage C: Object Detection (YOLOv8-Thermal)...")
    
    with torch.no_grad():
        detections = detector(processed)
    
    print(f"   Detection scales: {list(detections['predictions'].keys())}")
    for scale, preds in detections['predictions'].items():
        print(f"     {scale}: {preds['cls'].shape[2:]} grid")
    
    # Timing analysis
    print("\n6. Latency Analysis...")
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = gate(thermal_batch)
            _ = router(thermal_batch, class_ids)
            _ = detector(processed)
    
    # Benchmark
    iterations = 50
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            class_ids = gate(thermal_batch)
    gate_time = (time.perf_counter() - start) / iterations * 1000
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            processed = router(thermal_batch, class_ids)
    router_time = (time.perf_counter() - start) / iterations * 1000
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            detections = detector(processed)
    detector_time = (time.perf_counter() - start) / iterations * 1000
    
    total_time = gate_time + router_time + detector_time
    
    print(f"   WeatherGate: {gate_time:.2f} ms")
    print(f"   WeatherRouter: {router_time:.2f} ms")
    print(f"   YOLOv8-Thermal: {detector_time:.2f} ms")
    print(f"   Total pipeline: {total_time:.2f} ms")
    print(f"   FPS: {1000/total_time * batch_size:.1f} (batch={batch_size})")
    
    print("\n" + "=" * 70)
    print("Pipeline Demonstration Complete!")
    print("=" * 70)


def inference(image_path: str, output_path: Optional[str] = None):
    """Run inference on a single image."""
    print(f"Running inference on: {image_path}")
    
    # This is a placeholder - in practice, you would:
    # 1. Load the image
    # 2. Preprocess to 14-bit format
    # 3. Run through the pipeline
    # 4. Post-process detections
    # 5. Save/visualize results
    
    print("Inference mode requires trained model weights.")
    print("Please train the models first using:")
    print("  1. python training/train_cyclegan.py")
    print("  2. python training/train_distillation.py")


def main():
    parser = argparse.ArgumentParser(
        description='GATE-IR: Gated Adaptive Thermal Enhancement for IR Detection'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='demo',
        choices=['demo', 'test', 'inference'],
        help='Run mode: demo, test, or inference'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='Path to thermal image (for inference mode)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for inference results'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_pipeline()
    elif args.mode == 'test':
        test_components()
    elif args.mode == 'inference':
        if args.image is None:
            print("Error: --image required for inference mode")
            sys.exit(1)
        inference(args.image, args.output)


if __name__ == '__main__':
    main()
