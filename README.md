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
- Ultralytics (required for YOLO loss during training)

---

## 🔬 Ablation Studies

The following ablation experiments are proposed to validate design decisions and explore alternative approaches:

### A. Pipeline Architecture Ablations

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| A1 | **Pseudo-RGB at Test Time** | Convert IR→Pseudo-RGB via CycleGAN before detection, train student on RGB | Higher accuracy but +50-100ms latency | Tests if cross-modal features help |
| A2 | **No Weather Gating** | Remove WeatherGate, apply all preprocessing | Baseline without gating overhead | May over-process clear images |
| A3 | **Joint Preprocessing** | Apply fog+rain preprocessing to all images | Simpler pipeline | Risk of artifacts on clear images |
| A4 | **Our Pipeline** | Do selective processing based on our classification | Simpler pipeline | Risk of artifacts on clear images |

### B. Gating Strategy Ablations

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| B1 | **Soft Gating** | Use classification probabilities as blend weights: `out = p_clear×identity + p_fog×fog_enhance + p_rain×rain_remove` | Smoother transitions, fewer artifacts | Increases compute 3× |
| B2 | **Top-2 Soft Gating** | Blend only top-2 predictions | Balance between hard/soft | Reduces soft gating overhead |
| B3 | **Learned Gating** | Replace MLP with learnable attention over preprocessing outputs | End-to-end optimization | Requires labeled weather data |
| B4 | **Temperature Scaling** | Adjust softmax temperature for softer/harder decisions | Tune decision sharpness | `probs = softmax(logits/T)` |

### C. Preprocessing Ablations (Fog Enhancement)

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| C1 | **Histogram Equalization** | Replace adaptive gamma with HE | Stronger contrast, risk of over-enhancement | Simpler, non-parametric |
| C2 | **CLAHE for Fog** | Use CLAHE instead of gamma correction | Local contrast improvement | Compare with gamma |
| C3 | **Dehaze Networks** | Use learned dehazing (DCP, AOD-Net) | Better fog removal | Higher latency |
| C4 | **Multi-Scale Retinex** | Apply MSR for illumination normalization | Robust to varying fog density | Compute intensive |
| C5 | **No Fog Enhancement** | Skip fog preprocessing entirely | Establish enhancement value | Baseline |

### D. Preprocessing Ablations (Rain Removal)

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| D1 | **Rain Mask as Channel** | Concatenate rain mask as 2nd channel instead of subtracting: `[IR, mask]` → YOLO | Model learns to use mask | 2-channel input, arch changes |
| D2 | **Attention Masking** | Use rain mask as spatial attention: `IR × (1 - α×mask)` | Soft suppression | Tunable α |
| D3 | **Deeper LSRB** | Increase LSRB layers (3→5→7) | Better rain extraction | More params, slower |
| D4 | **U-Net for Rain** | Replace LSRB with U-Net architecture | Multi-scale rain removal | Heavier network |
| D5 | **No Rain Preprocessing** | Skip rain removal entirely | Establish enhancement value | Baseline |

### E. Detection Architecture Ablations

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| E1 | **No P2 Head** | Remove 160×160 detection head | Fewer params, lower accuracy on small objects | Standard YOLOv8 |
| E2 | **No Transformer Neck** | Remove ViT neck, use PAN only | Faster inference | Less global context |
| E3 | **Dual-Head (CIoU + GIoU)** | Use both IoU variants in loss | May improve localization | Minor overhead |
| E4 | **EfficientViT Neck** | Replace TransformerBlock with EfficientViT | Better speed/accuracy | Different attention |
| E5 | **SPPF Position** | Move SPPF to neck instead of backbone | Different receptive field | Structural change |

### F. Knowledge Distillation Ablations

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| F1 | **Feature Mimic Only** | Remove detection loss, train only with L_mimic | Tests distillation effectiveness | No GT boxes needed |
| F2 | **Detection Loss Only** | No distillation, train with boxes only | Baseline without teacher | Standard training |
| F3 | **Layer Selection** | Distill different layers (P3+P4 vs P4+P5) | Find optimal transfer point | Affects what knowledge transfers |
| F4 | **Temperature in KD** | Use softmax temperature for soft targets | Smoother knowledge transfer | Standard KD technique |
| F5 | **Self-Distillation** | Use larger thermal model as teacher | No RGB teacher needed | Simpler pipeline |

### G. Input/Output Ablations

| # | Experiment | Description | Expected Outcome | Notes |
|---|------------|-------------|------------------|-------|
| G1 | **8-bit vs 14-bit** | Compare 8-bit and 14-bit thermal input | Measure HDR benefit | Data preprocessing |
| G2 | **Multi-Frame Input** | Stack temporal frames as channels | Temporal context | 3-5 frame input |
| G3 | **Resolution Scaling** | Test 320×320, 480×480, 640×640, 1280×1280 | Speed/accuracy tradeoff | Standard ablation |

---

### Recommended Priority

| Priority | Experiment | Rationale |
|----------|------------|-----------|
| 🔴 High | B1 (Soft Gating) | Likely to improve edge cases |
| 🔴 High | D1 (Rain Mask as Channel) | Novel approach, may help detector |
| 🟡 Medium | A1 (Pseudo-RGB Test) | Validates distillation approach |
| 🟡 Medium | C2 (CLAHE for Fog) | Simple implementation change |
| 🟢 Low | E1 (No P2), E2 (No ViT) | Ablate key architecture choices |

---

### Ablation Results Template

```markdown
| Experiment | mAP@0.5 | mAP@0.5:0.95 | Latency (ms) | Notes |
|------------|---------|--------------|--------------|-------|
| Baseline   | --.-    | --.-         | --.-         | Full pipeline |
| B1: Soft Gating | | | | |
| D1: Rain Mask Channel | | | | |
| ...        |         |              |              |       |
```

---

## 📄 License

MIT License
