"""
GATE-IR Gate Module

Weather classification and gating mechanism for routing thermal images.
"""

from .weather_gate import WeatherGate, extract_features

__all__ = ["WeatherGate", "extract_features"]
