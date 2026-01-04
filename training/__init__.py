"""
GATE-IR Training Module

Training utilities for CycleGAN and Knowledge Distillation.
"""

from .cyclegan import (
    CycleGAN,
    Generator,
    Discriminator,
    convert_ir_to_pseudo_rgb
)
from .distillation import (
    DistillationTrainer,
    FeatureMimicLoss,
    ChannelAdapter
)

__all__ = [
    "CycleGAN",
    "Generator", 
    "Discriminator",
    "convert_ir_to_pseudo_rgb",
    "DistillationTrainer",
    "FeatureMimicLoss",
    "ChannelAdapter",
]
