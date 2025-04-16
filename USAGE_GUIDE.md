# OSU AI Aim Assistant - Usage Guide

This guide explains how to use the OSU AI Aim Assistant application for enhancing your gameplay. This tool uses computer vision and AI to detect game elements and provide aiming assistance.

## Installation

1. Ensure you have Python 3.8+ installed on your system
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. If you have a CUDA-compatible GPU, make sure you have the appropriate CUDA drivers installed

## Model Training (Optional)

If you want to train your own custom detection model instead of using a pre-trained one:

1. Capture screenshots from the game:
   ```
   python train_model.py capture --output ./data/images --num 1000
   ```

2. Label the captured images following the generated instructions:
   ```
   python train_model.py label_instructions
   ```

3. Organize your labeled data into train/val/test folders
   ```
   data/
     ├── train/
     │   ├── images/
     │   └── labels/
     ├── val/
     │   ├── images/
     │   └── labels/
     └── test/
         ├── images/
         └── labels/
   ```

4. Create the dataset YAML configuration:
   ```
   python train_model.py dataset --data_dir ./data
   ```

5. Train the model:
   ```
   python train_model.py train --data ./data/dataset.yaml --epochs 100 --model-size n
   ```

The best model will be saved in the `osu_model/train` directory.

## Running the Aim Assistant

1. Start the application:
   ```
   python main.py
   ```

2. In the application interface:
   - Click "Select Model" to load your trained YOLOv11 model
   - Adjust the confidence threshold to control detection sensitivity
   - Set the assist strength to your preference
   - Choose an aim mode:
     - **Snap**: Direct movement to target (most noticeable)
     - **Smooth**: Gradual movement to target
     - **Interpolate**: Natural-looking movement with easing
   - Configure capture resolution to match your game window
   - Toggle the detection overlay for visual feedback

3. Press the "Enable Aim Assist" button or F8 key to toggle the assist on/off

## Settings Explanation

| Setting | Description |
|---------|-------------|
| **Confidence Threshold** | Minimum probability (0-1) for detecting targets. Higher values mean fewer but more certain detections. |
| **Assist Strength** | How much the cursor is moved towards the target (0-1). Higher values provide more assistance. |
| **Aim Mode** | How the cursor moves towards the target (snap, smooth, interpolate). |
| **Capture Width/Height** | The size of the screen area to capture. Match to your game window. |
| **Target FPS** | How many times per second the assistance updates. Higher values are more responsive but use more CPU. |
| **Show Detection Overlay** | Display bounding boxes around detected targets. |
| **Always On Top** | Keep the application window on top of other windows. |

## Troubleshooting

- **No detections**: Ensure confidence threshold isn't too high and the model is properly loaded
- **Performance issues**: Try reducing capture resolution or target FPS
- **Model fails to load**: Check that the model file is a valid YOLOv11 model
- **Application crashes**: Verify all dependencies are installed correctly

## Disclaimer

This tool is for educational purposes only. Using aim assist software may violate the terms of service for OSU and result in account penalties. Use at your own risk. 