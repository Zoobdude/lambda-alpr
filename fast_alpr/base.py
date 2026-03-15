"""
Base module.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates in pixel space."""

    x1: int
    """Left coordinate."""
    y1: int
    """Top coordinate."""
    x2: int
    """Right coordinate."""
    y2: int
    """Bottom coordinate."""


@dataclass(frozen=True)
class DetectionResult:
    """One plate detection from the detector."""

    label: str
    """Detected class label."""
    confidence: float
    """Detection confidence from 0.0 to 1.0."""
    bounding_box: BoundingBox
    """Plate location in the image."""


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
