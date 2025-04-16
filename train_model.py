import os
import argparse
import yaml
import torch
from ultralytics import YOLO

def create_dataset_yaml(
    data_dir, 
    class_names=["osu_circle", "osu_slider", "osu_spinner"], 
    train_val_test_ratio=(80, 10, 10)
):
    """
    Create a YAML configuration file for YOLOv11 training
    
    Args:
        data_dir: Directory containing images and labels
        class_names: List of class names
        train_val_test_ratio: Ratio for train/val/test split
    """
    
    # Create paths
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Create dataset YAML config
    dataset_config = {
        "path": data_dir,
        "train": train_dir,
        "val": val_dir,
        "test": test_dir,
        "names": {i: name for i, name in enumerate(class_names)}
    }
    
    # Save YAML config
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset config at {yaml_path}")
    return yaml_path

def train_model(
    data_yaml,
    epochs=100, 
    batch_size=16, 
    image_size=640, 
    model_size="n",  # n, s, m, l, x
    workers=4,
    device=0
):
    """
    Train YOLOv11 model on custom OSU dataset
    
    Args:
        data_yaml: Path to dataset YAML config
        epochs: Number of training epochs
        batch_size: Batch size
        image_size: Input image size
        model_size: Model size (n, s, m, l, x)
        workers: Number of workers for data loading
        device: GPU device ID or 'cpu'
    """
    # Load a pretrained YOLO model
    model = YOLO(f"yolov11{model_size}.pt")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        workers=workers,
        device=device,
        patience=20,  # Early stopping patience
        augment=True,  # Use data augmentation
        save=True,    # Save best model
        project="osu_model",
        name="train"
    )
    
    print(f"Training completed. Model saved to {os.path.join('osu_model', 'train')}")
    return results

def capture_osu_screenshots(
    output_dir,
    num_samples=1000,
    capture_delay=0.5,
    resolution=(1280, 720)
):
    """
    Capture screenshots from OSU for dataset creation
    
    Args:
        output_dir: Directory to save screenshots
        num_samples: Number of screenshots to capture
        capture_delay: Delay between captures in seconds
        resolution: Resolution for capture
    """
    try:
        import time
        import dxcam
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize screen capture
        camera = dxcam.create()
        region = (0, 0, resolution[0], resolution[1])
        
        print(f"Starting capture of {num_samples} screenshots.")
        print("Press Ctrl+C to stop early.")
        
        try:
            for i in range(num_samples):
                # Capture screen
                frame = camera.grab(region=region)
                
                if frame is not None:
                    # Save frame
                    filename = os.path.join(output_dir, f"screenshot_{i:04d}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                
                # Delay between captures
                time.sleep(capture_delay)
        
        except KeyboardInterrupt:
            print("Capture interrupted by user.")
        
        print(f"Capture completed. {i+1} screenshots saved to {output_dir}")
    
    except ImportError:
        print("Error: dxcam or opencv-python not installed.")
        print("Please install with: pip install dxcam opencv-python")

def create_data_labeling_instructions(output_file="labeling_instructions.txt"):
    """
    Create instructions for manually labeling the dataset
    """
    instructions = """
## OSU Dataset Labeling Instructions

1. Install a labeling tool like LabelImg (https://github.com/heartexlabs/labelImg)
2. Open the captured screenshots folder
3. For each image, label:
   - osu_circle: The clickable circles
   - osu_slider: Slider objects 
   - osu_spinner: Spinner objects
4. Save labels in YOLO format (TXT files)
5. Organize the labeled data into train/val/test folders

Example YOLO format (each line in a .txt file):
<class_id> <x_center> <y_center> <width> <height>

- class_id: 0 for osu_circle, 1 for osu_slider, 2 for osu_spinner
- x_center, y_center: Center coordinates normalized to [0,1]
- width, height: Size normalized to [0,1]
"""
    with open(output_file, "w") as f:
        f.write(instructions)
    
    print(f"Created labeling instructions at {output_file}")

def main():
    parser = argparse.ArgumentParser(description="OSU YOLOv11 Model Trainer")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Capture command
    capture_parser = subparsers.add_parser("capture", help="Capture OSU screenshots")
    capture_parser.add_argument("--output", "-o", default="./data/images", help="Output directory")
    capture_parser.add_argument("--num", "-n", type=int, default=1000, help="Number of screenshots")
    capture_parser.add_argument("--delay", "-d", type=float, default=0.5, help="Delay between captures")
    capture_parser.add_argument("--width", "-W", type=int, default=1280, help="Capture width")
    capture_parser.add_argument("--height", "-H", type=int, default=720, help="Capture height")
    
    # Label instructions command
    label_parser = subparsers.add_parser("label_instructions", help="Create labeling instructions")
    label_parser.add_argument("--output", "-o", default="labeling_instructions.txt", help="Output file")
    
    # Dataset command
    dataset_parser = subparsers.add_parser("dataset", help="Create dataset YAML")
    dataset_parser.add_argument("--data_dir", "-d", required=True, help="Data directory")
    dataset_parser.add_argument("--classes", "-c", nargs="+", default=["osu_circle", "osu_slider", "osu_spinner"], 
                                help="Class names")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLOv11 model")
    train_parser.add_argument("--data", "-d", required=True, help="Path to dataset YAML")
    train_parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--batch", "-b", type=int, default=16, help="Batch size")
    train_parser.add_argument("--img-size", "-i", type=int, default=640, help="Image size")
    train_parser.add_argument("--model-size", "-m", default="n", choices=["n", "s", "m", "l", "x"], 
                              help="Model size")
    train_parser.add_argument("--workers", "-w", type=int, default=4, help="Number of workers")
    train_parser.add_argument("--device", default=0, help="Device (0, 1, cpu)")
    
    args = parser.parse_args()
    
    if args.command == "capture":
        import cv2  # Import here to avoid unnecessary import if not using capture
        capture_osu_screenshots(
            args.output, 
            args.num, 
            args.delay, 
            (args.width, args.height)
        )
    
    elif args.command == "label_instructions":
        create_data_labeling_instructions(args.output)
    
    elif args.command == "dataset":
        create_dataset_yaml(args.data_dir, args.classes)
    
    elif args.command == "train":
        train_model(
            args.data, 
            args.epochs, 
            args.batch, 
            args.img_size, 
            args.model_size, 
            args.workers, 
            args.device
        )
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 