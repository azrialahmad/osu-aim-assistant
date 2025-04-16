# Model Directory

Place your trained YOLOv11 model files (.pt or .pth) in this directory.

## Pre-trained Models

You have a few options for models:

1. Train your own model using the provided `train_model.py` script
2. Use the Ultralytics YOLOv8 models (compatible with our code)
3. Convert existing object detection models to the YOLO format

## Recommended Model Settings for OSU

For detecting OSU game elements (circles, sliders, spinners), we recommend:

- Model size: YOLOv11n or YOLOv11s (good balance of speed and accuracy)
- Input resolution: 640×640 (standard) or 1280×1280 (higher accuracy)
- Confidence threshold: Start with 0.5 and adjust based on performance

## Model Compatibility

This application is primarily designed for YOLOv11 models, but it's also compatible with:

- YOLOv8 models from Ultralytics
- YOLOv5 models (with minor modifications)

## Custom Training Tips

When training your own model for OSU:

1. Capture diverse gameplay scenarios
2. Include different skins and visual styles
3. Ensure balanced representation of all target types (circles, sliders, spinners)
4. Use data augmentation to improve robustness

## Performance Optimization

If experiencing performance issues:
- Use a smaller model variant (YOLOv11n)
- Reduce inference resolution
- Consider half-precision (FP16) if your GPU supports it 