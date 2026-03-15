# Reference

This page shows the public API of FastALPR.

## Common Inputs

`ALPR.predict()` and `ALPR.draw_predictions()` accept:

- A NumPy image in BGR format
- A string path to an image file

## Common Returns

- `ALPR.predict(...)` returns `list[ALPRResult]`
- `ALPR.draw_predictions(...)` returns `DrawPredictionsResult`

`ALPRResult` contains:

- `detection`: box, label, and detection confidence
- `ocr`: recognized text and OCR confidence, or `None`

`DrawPredictionsResult` contains:

- `image`: the image with boxes and text drawn on it
- `results`: the same ALPR results used for drawing

## Available Models

See the available detection models in [open-image-models](https://ankandrew.github.io/open-image-models/0.4/reference/#open_image_models.detection.core.hub.PlateDetectorModel)
and OCR models in [fast-plate-ocr](https://ankandrew.github.io/fast-plate-ocr/1.0/inference/model_zoo/).

## Main API

::: fast_alpr.alpr

## Base Types

::: fast_alpr.base
