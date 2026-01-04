"""
GATE-IR Preprocessing Module

Weather-specific preprocessing modules for thermal images.
"""

from .fog_enhancer import FogEnhancer
from .rain_remover import RainRemover, CLAHE, LocalContrastNormalization
from .weather_router import WeatherRouter

__all__ = [
    "FogEnhancer",
    "RainRemover",
    "CLAHE",
    "LocalContrastNormalization",
    "WeatherRouter",
]
