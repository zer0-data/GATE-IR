"""
GATE-IR Models Module

Custom object detection architectures for thermal imagery.
"""

from .yolov8_thermal import (
    YOLOv8Thermal,
    YOLOv8ThermalBackbone,
    TransformerNeck,
    P2DetectionHead
)

__all__ = [
    "YOLOv8Thermal",
    "YOLOv8ThermalBackbone",
    "TransformerNeck",
    "P2DetectionHead",
]
