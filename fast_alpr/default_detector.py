"""
Default Detector module.
"""

import os
from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from open_image_models import LicensePlateDetector
from open_image_models.detection.core.hub import PlateDetectorModel
from open_image_models.detection.core.yolo_v9.inference import YoloV9ObjectDetector

from fast_alpr.base import BaseDetector, DetectionResult


class DefaultDetector(BaseDetector):
    """
    Default detector class for license plate detection using ONNX models.

    This class utilizes the `LicensePlateDetector` from the `open_image_models` package
    to perform detection on input frames. When a custom model path is supplied, the ONNX
    model is loaded directly via `YoloV9ObjectDetector` without downloading from the hub.
    """

    def __init__(
        self,
        model_name: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        conf_thresh: float = 0.4,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions = None,
        model_path: str | os.PathLike | None = None,
    ) -> None:
        """
        Initialize the DefaultDetector with the specified parameters. Uses `open-image-models`'s
        `LicensePlateDetector`.

        Parameters:
            model_name: The name of the detector model. See `PlateDetectorModel` for the available
                models. Ignored when `model_path` is provided.
            conf_thresh: Confidence threshold for the detector. Defaults to 0.25.
            providers: The execution providers to use in ONNX Runtime. If None, the default
                providers are used.
            sess_options: Custom session options for ONNX Runtime. If None, default session options
                are used.
            model_path: Path to a custom ONNX detector model file. When provided, the model is
                loaded directly from this path instead of being downloaded from the hub, and
                `model_name` is ignored.
        """
        if model_path is not None:
            self.detector = YoloV9ObjectDetector(
                model_path=model_path,
                class_labels=["License Plate"],
                conf_thresh=conf_thresh,
                providers=providers,
                sess_options=sess_options,
            )
        else:
            self.detector = LicensePlateDetector(
                detection_model=model_name,
                conf_thresh=conf_thresh,
                providers=providers,
                sess_options=sess_options,
            )

    def predict(self, frame: np.ndarray) -> list[DetectionResult]:
        """
        Perform detection on the input frame and return a list of detections.

        Parameters:
            frame: The input image/frame in which to detect license plates.

        Returns:
            A list of detection results, each containing the label,
            confidence, and bounding box of a detected license plate.
        """
        return self.detector.predict(frame)
