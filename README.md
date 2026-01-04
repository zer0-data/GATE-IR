# GATE-IR

**Gated Adaptive Thermal Enhancement for Infrared Detection**

A multi-stage deep learning pipeline for thermal image processing with weather-adaptive preprocessing and optimized object detection.

## 🎯 Overview

GATE-IR addresses the challenge of object detection in adverse weather conditions for thermal/infrared imagery through:

1. **Stage A (Gating)**: Lightweight weather classification (Clear/Fog/Rain)
2. **Stage B (Preprocessing)**: Weather-specific image enhancement
3. **Stage C (Detection)**: Custom YOLOv8-Small optimized for thermal targets

```
Thermal Image → WeatherGate → [Clear|Fog|Rain] → Preprocessing → YOLOv8-Thermal → Detections
```

## 📁 Project Structure

```
GATE-IR/
├── gate/                    # Weather classification
│   └── weather_gate.py      # WeatherGate classifier
├── preprocessing/           # Image enhancement
│   ├── fog_enhancer.py      # Adaptive gamma correction
│   ├── rain_remover.py      # LSRB + CLAHE
│   └── weather_router.py    # Conditional routing
├── models/                  # Detection architecture
│   ├── yolov8_thermal.py    # Custom YOLOv8 with P2 head + ViT neck
│   └── yolov8_thermal.yaml  # Ultralytics config
├── training/                # Training pipelines
│   ├── cyclegan.py          # IR → Pseudo-RGB translation
│   ├── train_cyclegan.py    # CycleGAN training script
│   └── distillation.py      # Teacher-Student distillation
├── docs/
│   └── theory.md            # Theoretical background
├── main.py                  # Demo and testing
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Test Components

```bash
python main.py --mode test
```

### Run Demo

```bash
python main.py --mode demo
```

## 🧩 Components

### 1. WeatherGate (Stage A)

Low-latency classifier using 5 statistical features:
- Thermal Variance, Min, Max, Entropy, Laplacian Variance

```python
from gate.weather_gate import WeatherGate

gate = WeatherGate()
class_ids = gate(thermal_batch)  # 0=Clear, 1=Fog, 2=Rain
```

### 2. Preprocessing (Stage B)

| Module | Method | Use Case |
|--------|--------|----------|
| FogEnhancer | Adaptive Gamma | Low contrast foggy images |
| RainRemover | LSRB + CLAHE | Rain streak removal |
| WeatherRouter | Conditional routing | Automatic path selection |

```python
from preprocessing import WeatherRouter

router = WeatherRouter()
processed = router(thermal_batch, class_ids)
```

### 3. YOLOv8-Thermal (Stage C)

Modified YOLOv8-Small for thermal imagery:
- **Single-channel input** (vs 3-channel RGB)
- **P2 detection head** (160×160 for small objects)
- **ViT neck** (global context recovery)

```python
from models.yolov8_thermal import yolov8s_thermal

model = yolov8s_thermal(num_classes=3, include_p2=True)
detections = model(processed_batch)
```

### 4. CycleGAN Training

Train IR → Pseudo-RGB generator for knowledge distillation:

```bash
python training/train_cyclegan.py \
    --ir_dir ./data/thermal \
    --rgb_dir ./data/rgb \
    --epochs 200
```

### 5. Knowledge Distillation

Transfer knowledge from RGB-pretrained teacher to thermal student:

```python
from training.distillation import DistillationTrainer

trainer = DistillationTrainer(student, teacher, cyclegan_generator)
losses = trainer.train_step(thermal_batch, targets)
```

## 📖 Documentation

See [docs/theory.md](docs/theory.md) for detailed theoretical background on:
- Thermal imaging fundamentals
- Weather degradation models
- Feature extraction rationale
- Architecture design decisions

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV (optional, for CLAHE acceleration)

## 📄 License

MIT License
