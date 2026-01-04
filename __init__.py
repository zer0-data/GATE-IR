"""
GATE-IR: Gated Adaptive Thermal Enhancement for IR Detection

A multi-stage pipeline for thermal image processing with weather-adaptive 
preprocessing and custom object detection optimized for infrared imagery.
"""

__version__ = "1.0.0"
__author__ = "GATE-IR Team"

from gate.weather_gate import WeatherGate
from preprocessing.weather_router import WeatherRouter
from preprocessing.fog_enhancer import FogEnhancer
from preprocessing.rain_remover import RainRemover

__all__ = [
    "WeatherGate",
    "WeatherRouter", 
    "FogEnhancer",
    "RainRemover",
]
