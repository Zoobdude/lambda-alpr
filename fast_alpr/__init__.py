"""
FastALPR package.
"""

from fast_alpr.alpr import ALPR, ALPRResult, DrawPredictionsResult
from fast_alpr.base import BaseDetector, BaseOCR, BoundingBox, DetectionResult, OcrResult

__all__ = [
    "ALPR",
    "ALPRResult",
    "BaseDetector",
    "BaseOCR",
    "BoundingBox",
    "DetectionResult",
    "DrawPredictionsResult",
    "OcrResult",
]
