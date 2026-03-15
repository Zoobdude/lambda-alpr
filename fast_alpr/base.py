"""
Base module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from open_image_models.detection.core.base import DetectionResult


@dataclass(frozen=True)
class OcrResult:
    """OCR output for one cropped plate image."""

    text: str
    """Recognized plate text."""
    confidence: float | list[float]
    """OCR confidence as one value or one value per character."""
    region: str | None = None
    """Optional region or country prediction."""
    region_confidence: float | None = None
    """Confidence for the region prediction."""


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """Perform detection on the input frame and return a list of detections."""


class BaseOCR(ABC):
    @abstractmethod
    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        """Perform OCR on the cropped plate image and return the recognized text and character
        probabilities."""
