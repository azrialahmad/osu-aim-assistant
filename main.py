import sys
import os
import time
import threading
import cv2
import numpy as np
import torch
import dxcam
import win32api
import win32con
import ctypes
import mouse  # For tablet support
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QLabel, QSlider, QPushButton, QCheckBox, 
                           QGroupBox, QComboBox, QSpinBox, QFileDialog, QMessageBox,
                           QRadioButton, QButtonGroup, QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer
from PyQt5.QtGui import QPixmap, QImage
from pynput import keyboard
import json

# Global settings
SETTINGS = {
    "confidence_threshold": 0.6,
    "assist_strength": 0.5,
    "enabled": False,
    "aim_mode": 0,
    "capture_width": 1280,
    "capture_height": 720,
    "model_path": "",
    "show_overlay": True,
    "target_fps": 144,
    "always_on_top": False,
    "preview_enabled": True,
    "target_class": 0,  # 0 for circles
    "input_device": "mouse",
    "use_play_area": False,
    "simple_capture": True,
    "debug_logging": False,
    "profile_name": "Default",
    "half_precision": False
}

# Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(APP_DIR, "profiles")
SETTINGS_FILE = os.path.join(APP_DIR, "settings.json")

# Class mapping for the model
CLASS_MAPPING = []

class DetectionThread(QThread):
    update_frame = pyqtSignal(np.ndarray, list)
    update_fps = pyqtSignal(float)  # Add FPS signal
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.paused = False
        self.model = None
        self.pause_lock = threading.Lock()
        self.frame_times = []  # For FPS calculation
        self.max_frame_times = 30  # Number of frames to average for FPS
        self.cuda_available = False  # Will be updated during model load
        
    def debug_print(self, message):
        """Print debug messages only if debug logging is enabled"""
        if SETTINGS["debug_logging"]:
            print(message)
            
    def load_model(self, model_path):
        try:
            # Make sure the model file exists
            if not os.path.exists(model_path):
                print(f"Error: Model file not found: {model_path}")
                return False
                
            # Check for CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
                print(f"CUDA is available! Devices: {device_count}, Name: {device_name}")
            else:
                print("CUDA is not available. Using CPU.")
            
            # Store CUDA status
            self.cuda_available = cuda_available
            
            # Load model using YOLO's direct method instead of torch.hub
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            
            # Apply half precision if enabled and CUDA is available 
            # This can significantly improve performance on compatible GPUs
            if SETTINGS["half_precision"] and cuda_available:
                print("Using half precision (FP16) for improved performance")
                self.model.model.half()  # Convert model to half precision
            
            # Set initial confidence threshold
            self.conf = SETTINGS["confidence_threshold"]  # Store locally
            
            # Print model info
            print(f"Model loaded: {model_path}")
            
            # Map class names
            global CLASS_MAPPING
            if hasattr(self.model, 'names'):
                CLASS_MAPPING = self.model.names
                print(f"Model classes: {CLASS_MAPPING}")
                
                # Try to find the active_note class
                if isinstance(CLASS_MAPPING, dict):
                    for idx, name in CLASS_MAPPING.items():
                        if "active" in name.lower():
                            SETTINGS["target_class"] = idx
                            print(f"Setting target class to {idx}: {name}")
                            break
            
            return True
        except Exception as e:
            detailed_error = str(e)
            print(f"Error loading model: {detailed_error}")
            
            # Add more detailed debugging for common issues
            if "No module named 'ultralytics'" in detailed_error:
                print("The ultralytics package is not installed. Try: pip install ultralytics")
            elif "CUDA out of memory" in detailed_error:
                print("CUDA out of memory error. Try using a smaller model or reducing batch size.")
            elif "ModuleNotFoundError" in detailed_error:
                print("Missing dependency. Check that all required packages are installed.")
            
            return False
    
    def pause(self):
        """Pause the detection thread"""
        with self.pause_lock:
            self.paused = True
            print("Detection thread paused")
    
    def resume(self):
        """Resume the detection thread"""
        with self.pause_lock:
            self.paused = False
            print("Detection thread resumed")
            
    def update_confidence(self, value):
        """Update the model's confidence threshold safely"""
        if self.model is not None:
            # Pause detection to avoid issues during update
            was_paused = self.paused
            if not was_paused:
                self.pause()
                
            # Update the confidence threshold (stored locally, used in predict call)
            self.conf = value
            print(f"Updated model confidence to {value:.2f}")
            
            # Resume detection if it wasn't paused before
            if not was_paused:
                self.resume()
    
    def calculate_fps(self):
        """Calculate the current FPS based on recent frame times"""
        if len(self.frame_times) < 2:
            return 0.0
            
        # Calculate average time between frames
        time_diffs = []
        for i in range(1, len(self.frame_times)):
            time_diffs.append(self.frame_times[i] - self.frame_times[i-1])
            
        avg_time = sum(time_diffs) / len(time_diffs)
        fps = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return fps
        
    def calculate_play_area(self, window_rect):
        """Calculate the osu! play area (4:3 ratio centered in the window)"""
        left, top, right, bottom = window_rect
        window_width = right - left
        window_height = bottom - top
        
        # Get the maximum 4:3 area that fits in the window
        if window_width / window_height > 4 / 3:  # Wider than 4:3
            # Height is the limiting factor
            play_height = window_height
            play_width = int(play_height * 4 / 3)
            # Center horizontally
            play_left = left + (window_width - play_width) // 2
            play_top = top
        else:  # Taller than 4:3
            # Width is the limiting factor
            play_width = window_width
            play_height = int(play_width * 3 / 4)
            # Center vertically
            play_left = left
            play_top = top + (window_height - play_height) // 2
        
        play_right = play_left + play_width
        play_bottom = play_top + play_height
        
        return (play_left, play_top, play_right, play_bottom)
    
    def validate_region(self, region, screen_width, screen_height):
        """Validate and fix region to ensure it's within screen bounds"""
        left, top, right, bottom = region
        
        # Ensure region is within screen bounds
        left = max(0, min(left, screen_width - 1))
        top = max(0, min(top, screen_height - 1))
        right = max(left + 1, min(right, screen_width))
        bottom = max(top + 1, min(bottom, screen_height))
        
        # Ensure region has minimum size
        if right - left < 10:
            right = min(left + 10, screen_width)
        if bottom - top < 10:
            bottom = min(top + 10, screen_height)
            
        return (left, top, right, bottom)
    
    def initialize_dxcam(self):
        """Initialize a DXcam instance with the appropriate parameters"""
        try:
            # Get the screen dimensions
            import ctypes
            user32 = ctypes.windll.user32
            screen_width = user32.GetSystemMetrics(0)
            screen_height = user32.GetSystemMetrics(1)
            self.debug_print(f"Screen dimensions: {screen_width}x{screen_height}")
            
            # Create a DXcam instance with appropriate output dimensions
            camera = dxcam.create(output_idx=0, output_color="RGB")
            if camera is None:
                print("Error: Failed to initialize DXcam camera")
                return None, 0, 0
                
            return camera, screen_width, screen_height
            
        except Exception as e:
            print(f"Error initializing DXcam: {e}")
            return None, 0, 0
    
    def run(self):
        # Initialize screen capture
        try:
            # Try to find the osu! window
            import win32gui
            
            def find_osu_window():
                """Find the osu! window and return its handle and rect"""
                def callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd) and "osu!" in win32gui.GetWindowText(hwnd):
                        windows.append(hwnd)
                    return True
                
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if not windows:
                    return None, None
                
                hwnd = windows[0]  # Use first found window
                rect = win32gui.GetWindowRect(hwnd)
                return hwnd, rect
            
            # Initialize camera and get screen dimensions
            camera, screen_width, screen_height = self.initialize_dxcam()
            if camera is None:
                print("Failed to create DXcam instance")
                return
                
            self.running = True
            while self.running:
                # Record frame start time for FPS calculation
                frame_start_time = time.time()
                
                # Check if thread is paused
                if self.paused or not SETTINGS["enabled"]:
                    time.sleep(1 / SETTINGS["target_fps"])
                    continue
                    
                if self.model is None:
                    # Skip processing if no model is loaded, but don't exit the thread
                    time.sleep(1 / SETTINGS["target_fps"])
                    continue
                    
                # Try to capture the osu! window
                try:
                    # Find osu! window
                    hwnd, rect = find_osu_window()
                    
                    try:
                        # Check if we should use simple capture mode
                        if SETTINGS["simple_capture"]:
                            # Use full screen in simple mode
                            region = (0, 0, screen_width, screen_height)
                            self.debug_print(f"Using simple capture mode: {region}")
                        else:
                            # Normal capture mode
                            if hwnd:
                                # osu! window found
                                if SETTINGS["use_play_area"]:
                                    # Calculate play area (4:3 ratio)
                                    play_area = self.calculate_play_area(rect)
                                    region = play_area
                                else:
                                    # Capture the entire window
                                    region = rect
                            else:
                                # Fallback to configured region
                                region = (0, 0, SETTINGS["capture_width"], SETTINGS["capture_height"])
                            
                            # Validate region to ensure it's within screen bounds
                            region = self.validate_region(region, screen_width, screen_height)
                            
                        # Only print region if debug logging is enabled
                        self.debug_print(f"Capture region: {region}")
                            
                        # Capture screen with validated region
                        frame = camera.grab(region=region)
                    except ValueError as ve:
                        # If region is invalid, try a simple capture of the center of the screen
                        print(f"Region error: {ve}, falling back to simple capture")
                        try:
                            # Use a safe central region
                            fallback_width = min(800, screen_width)
                            fallback_height = min(600, screen_height)
                            left = (screen_width - fallback_width) // 2
                            top = (screen_height - fallback_height) // 2
                            safe_region = (left, top, left + fallback_width, top + fallback_height)
                            print(f"Using fallback region: {safe_region}")
                            frame = camera.grab(region=safe_region)
                            # If we successfully captured using fallback, update the region
                            if frame is not None:
                                region = safe_region
                        except Exception as e2:
                            # Emergency fallback - use the simplest possible capture
                            print(f"Emergency fallback needed: {e2}")
                            try:
                                # Try with a very small region in the center
                                emergency_width = 400
                                emergency_height = 300
                                left = (screen_width - emergency_width) // 2
                                top = (screen_height - emergency_height) // 2
                                emergency_region = (left, top, left + emergency_width, top + emergency_height)
                                print(f"Using emergency region: {emergency_region}")
                                frame = camera.grab(region=emergency_region)
                                if frame is not None:
                                    region = emergency_region
                            except Exception as e3:
                                print(f"All capture methods failed: {e3}")
                                frame = None
                    
                    if frame is not None:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Run detection with the current confidence threshold
                        results = self.model.predict(
                            source=frame_rgb,
                            conf=self.conf,
                            verbose=False
                        )
                        
                        # Extract detections
                        detections = []
                        if len(results) > 0 and len(results[0].boxes.xyxy) > 0:
                            boxes = results[0].boxes
                            for i in range(len(boxes.xyxy)):
                                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                                conf = boxes.conf[i].item()
                                cls_idx = int(boxes.cls[i].item())
                                
                                # Only consider the target class (active_note) or all if target_class is -1
                                if SETTINGS["target_class"] == -1 or cls_idx == SETTINGS["target_class"]:
                                    if conf >= SETTINGS["confidence_threshold"]:
                                        detections.append({
                                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                            'confidence': float(conf),
                                            'class': cls_idx
                                        })
                        
                        # Update the UI with the frame and detections (only if preview is enabled)
                        if SETTINGS["preview_enabled"]:
                            self.update_frame.emit(frame_rgb, detections)
                        
                        # Apply aim assist if enabled
                        if SETTINGS["enabled"] and detections:
                            self.apply_aim_assist(detections, region=region)
                        
                        # Record frame time for FPS calculation
                        self.frame_times.append(time.time())
                        # Keep only recent frames for FPS calculation
                        if len(self.frame_times) > self.max_frame_times:
                            self.frame_times.pop(0)
                        
                        # Calculate and emit FPS
                        fps = self.calculate_fps()
                        self.update_fps.emit(fps)
                        
                except Exception as e:
                    print(f"Error during screen capture or processing: {e}")
                    import traceback
                    traceback.print_exc()
                    # On error, wait a bit longer to avoid rapid error loops
                    time.sleep(0.5)
            
                # Control FPS - take into account processing time
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                sleep_time = max(0, (1 / SETTINGS["target_fps"]) - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            print(f"Fatal error in detection thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Make sure to clean up DXcam resources
            if 'camera' in locals() and camera is not None:
                camera.stop()
    
    def send_mouse_input(self, dx, dy):
        """
        More direct way to inject mouse movement using the Windows Input API
        Provides lower latency than SetCursorPos
        """
        # Define input structure
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("mi", MOUSEINPUT)
            ]
        
        # Prepare input object
        x = int(dx)
        y = int(dy)
        
        # Create input object
        extra = ctypes.c_ulong(0)
        ii_ = INPUT(
            type=ctypes.c_ulong(0),  # INPUT_MOUSE
            mi=MOUSEINPUT(
                dx=x, 
                dy=y, 
                mouseData=ctypes.c_ulong(0), 
                dwFlags=ctypes.c_ulong(0x0001),  # MOUSEEVENTF_MOVE
                time=ctypes.c_ulong(0),
                dwExtraInfo=ctypes.pointer(extra)
            )
        )
        
        # Send input
        ctypes.windll.user32.SendInput(
            1,  # Number of inputs
            ctypes.byref(ii_),  # Input object
            ctypes.sizeof(ii_)  # Size of the input object
        )
    
    def move_mouse_relative(self, dx, dy):
        """
        Move the mouse using the appropriate method based on input device setting.
        Optimized for both mouse and tablet inputs with low latency.
        """
        try:
            if SETTINGS["input_device"] == "tablet":
                # For tablets, absolute positioning works better than relative
                # Get current position first
                curr_x, curr_y = win32api.GetCursorPos()
                new_x = int(curr_x + dx)
                new_y = int(curr_y + dy)
                
                # Use direct Windows API for lowest latency
                win32api.SetCursorPos((new_x, new_y))
            else:
                # For mouse, use SendInput for lower latency relative movement
                self.send_mouse_input(dx, dy)
        except Exception as e:
            print(f"Mouse movement failed ({e}), falling back to SetCursorPos")
            # Fallback to SetCursorPos
            curr_x, curr_y = win32api.GetCursorPos()
            new_x = int(curr_x + dx)
            new_y = int(curr_y + dy)
            win32api.SetCursorPos((new_x, new_y))
    
    def apply_aim_assist(self, detections, region):
        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        if sorted_detections:
            # Get the highest confidence detection
            target = sorted_detections[0]
            bbox = target['bbox']
            
            # Calculate center of the target
            target_x = int((bbox[0] + bbox[2]) / 2)
            target_y = int((bbox[1] + bbox[3]) / 2)
            
            # Get current mouse position
            curr_x, curr_y = win32api.GetCursorPos()
            
            # Adjust for screen region offset
            region_left = region[0]
            region_top = region[1]
            
            # Calculate vector to target
            dx = target_x + region_left - curr_x
            dy = target_y + region_top - curr_y
            
            # Enhanced EMA smoothing for more natural movement
            if not hasattr(self, 'last_dx'):
                self.last_dx = 0
                self.last_dy = 0
            
            # EMA smoothing factor - lower value = more smoothing
            # Reduced from 0.3 to 0.2 for smoother movement
            ema_factor = 0.2
            
            # Apply enhanced EMA smoothing
            smoothed_dx = ema_factor * dx + (1 - ema_factor) * self.last_dx
            smoothed_dy = ema_factor * dy + (1 - ema_factor) * self.last_dy
            
            # Update last values
            self.last_dx = smoothed_dx
            self.last_dy = smoothed_dy
            
            # Get strength - increased multiplier for stronger effect at 100%
            strength = SETTINGS["assist_strength"] * 1.5  # Multiplier increased from 1.0 to 1.5
            
            # Apply strength based on selected aim mode
            if SETTINGS["aim_mode"] == "snap":
                # Direct snap toward target with increased strength factor
                move_x = smoothed_dx * strength
                move_y = smoothed_dy * strength
                
            elif SETTINGS["aim_mode"] == "smooth":
                # Gentle movement - increased from 0.1 to 0.15 for smoother control
                move_x = smoothed_dx * strength * 0.15
                move_y = smoothed_dy * strength * 0.15
                
            else:  # "interpolate" - default
                # Balanced approach - increased from 0.05 to 0.08
                move_x = smoothed_dx * strength * 0.08
                move_y = smoothed_dy * strength * 0.08
            
            # Apply additional movement smoothing
            # This helps reduce jitter when small movements are applied
            if abs(move_x) < 5 and abs(move_y) < 5:
                # For very small movements, apply extra smoothing to reduce jitter
                move_x *= 0.8
                move_y *= 0.8
            
            # Apply movement if significant enough
            if abs(move_x) > 0.2 or abs(move_y) > 0.2:  # Increased threshold slightly
                self.move_mouse_relative(move_x, move_y)
    
    def stop(self):
        self.running = False
        self.wait()

class KeyListener(threading.Thread):
    def __init__(self, toggle_callback):
        super().__init__()
        self.toggle_callback = toggle_callback
        self.daemon = True
        
    def run(self):
        def on_press(key):
            try:
                if key == keyboard.Key.f8:
                    self.toggle_callback()
            except Exception as e:
                print(f"Error in key listener: {e}")
        
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.detection_thread = None
        
        # Force detection to be disabled on startup - making this more explicit
        SETTINGS["enabled"] = False
        
        # Set up the UI
        self.init_ui()
        
        # Load settings (must be done after UI setup)
        self.load_settings()
        
        # Force detection to be disabled again after loading settings 
        # This ensures it's disabled even if a profile had it enabled
        SETTINGS["enabled"] = False
        self.update_detection_status()
        self.update_toggle_button_style()
        
        # Initialize the detection thread (must be done after settings are loaded)
        self.init_detection_thread()
        
        # Start key listener
        self.key_listener = KeyListener(self.toggle_assist)
        self.key_listener.start()
        
        # Show a startup message
        self.statusBar().showMessage("Application started", 3000)
    
    def init_ui(self):
        self.setWindowTitle("OSU AI Aim Assistant")
        self.setMinimumSize(1000, 600)
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Add CUDA status to status bar (right side)
        self.cuda_indicator = QLabel("CUDA: Unknown")
        self.cuda_indicator.setStyleSheet("color: gray;")
        self.statusBar().addPermanentWidget(self.cuda_indicator)
        
        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for preview
        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setText("Preview disabled")
        
        # Preview toggle button and status indicators
        preview_controls_layout = QHBoxLayout()
        
        # Left side - controls
        preview_buttons_layout = QVBoxLayout()
        self.preview_toggle_btn = QPushButton("Enable Preview")
        self.preview_toggle_btn.setCheckable(True)
        self.preview_toggle_btn.setChecked(SETTINGS["preview_enabled"])
        self.preview_toggle_btn.clicked.connect(self.toggle_preview)
        preview_buttons_layout.addWidget(self.preview_toggle_btn)
        
        # Right side - status indicators
        status_layout = QVBoxLayout()
        self.status_label = QLabel("Detection: Stopped")
        self.status_label.setStyleSheet("color: gray;")
        
        # Add FPS counter
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: blue; font-weight: bold;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        
        # Add both to the preview controls layout
        preview_controls_layout.addLayout(preview_buttons_layout)
        preview_controls_layout.addLayout(status_layout)
        
        preview_layout.addWidget(self.preview_label)
        preview_layout.addLayout(preview_controls_layout)
        
        # Right panel for settings
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Model selection
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        self.model_path_label = QLabel("No model selected")
        model_select_btn = QPushButton("Select Model")
        model_select_btn.clicked.connect(self.select_model)
        
        # Target class selection
        target_class_layout = QHBoxLayout()
        target_class_layout.addWidget(QLabel("Target Class:"))
        self.target_class_combo = QComboBox()
        self.target_class_combo.addItem("All Classes", -1)
        
        # Check if CLASS_MAPPING is a dictionary
        if isinstance(CLASS_MAPPING, dict):
            for idx, name in CLASS_MAPPING.items():
                self.target_class_combo.addItem(f"{name} ({idx})", idx)
                
        self.target_class_combo.setCurrentIndex(0)
        self.target_class_combo.currentIndexChanged.connect(self.update_target_class)
        target_class_layout.addWidget(self.target_class_combo)
        
        model_layout.addWidget(self.model_path_label)
        model_layout.addWidget(model_select_btn)
        model_layout.addLayout(target_class_layout)
        
        # Confidence threshold
        confidence_group = QGroupBox("Confidence Threshold")
        confidence_layout = QVBoxLayout(confidence_group)
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(SETTINGS["confidence_threshold"] * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_label = QLabel(f"Threshold: {SETTINGS['confidence_threshold']:.2f}")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        
        # Assist strength
        strength_group = QGroupBox("Assist Strength")
        strength_layout = QVBoxLayout(strength_group)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(SETTINGS["assist_strength"] * 100))
        self.strength_slider.valueChanged.connect(self.update_strength)
        self.strength_label = QLabel(f"Strength: {SETTINGS['assist_strength']:.2f}")
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_slider)
        
        # Aim mode
        aim_mode_group = QGroupBox("Aim Mode")
        aim_mode_layout = QVBoxLayout(aim_mode_group)
        self.aim_mode_combo = QComboBox()
        
        # Add aim modes
        aim_modes = ["snap", "smooth", "interpolate"]
        self.aim_mode_combo.addItems(aim_modes)
        
        # Set current aim mode - handle both string and integer values
        if isinstance(SETTINGS["aim_mode"], int):
            # If it's an integer index, use setCurrentIndex
            if 0 <= SETTINGS["aim_mode"] < len(aim_modes):
                self.aim_mode_combo.setCurrentIndex(SETTINGS["aim_mode"])
        else:
            # If it's a string, use setCurrentText
            self.aim_mode_combo.setCurrentText(SETTINGS["aim_mode"])
            
        self.aim_mode_combo.currentTextChanged.connect(self.update_aim_mode)
        aim_mode_layout.addWidget(self.aim_mode_combo)
        
        # Capture settings
        capture_group = QGroupBox("Capture Settings")
        capture_layout = QVBoxLayout(capture_group)
        
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Width:"))
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(320, 3840)
        self.width_spinbox.setValue(SETTINGS["capture_width"])
        self.width_spinbox.valueChanged.connect(self.update_capture_width)
        width_layout.addWidget(self.width_spinbox)
        
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Height:"))
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(240, 2160)
        self.height_spinbox.setValue(SETTINGS["capture_height"])
        self.height_spinbox.valueChanged.connect(self.update_capture_height)
        height_layout.addWidget(self.height_spinbox)
        
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(30, 240)
        self.fps_spinbox.setValue(SETTINGS["target_fps"])
        self.fps_spinbox.valueChanged.connect(self.update_target_fps)
        fps_layout.addWidget(self.fps_spinbox)
        
        capture_layout.addLayout(width_layout)
        capture_layout.addLayout(height_layout)
        capture_layout.addLayout(fps_layout)
        
        # Checkboxes
        checks_group = QGroupBox("Options")
        checks_layout = QVBoxLayout(checks_group)
        
        self.show_overlay_check = QCheckBox("Show Detection Overlay")
        self.show_overlay_check.setChecked(SETTINGS["show_overlay"])
        self.show_overlay_check.toggled.connect(self.update_show_overlay)
        
        self.always_top_check = QCheckBox("Always On Top")
        self.always_top_check.setChecked(SETTINGS["always_on_top"])
        self.always_top_check.toggled.connect(self.update_always_on_top)
        
        self.play_area_check = QCheckBox("Optimize for Play Area (4:3)")
        self.play_area_check.setChecked(SETTINGS["use_play_area"])
        self.play_area_check.toggled.connect(self.update_play_area)
        self.play_area_check.setToolTip("Only capture the 4:3 play area for better performance")
        
        self.simple_capture_check = QCheckBox("Simple Capture Mode (Fix Errors)")
        self.simple_capture_check.setChecked(SETTINGS["simple_capture"])
        self.simple_capture_check.toggled.connect(self.update_simple_capture)
        self.simple_capture_check.setToolTip("Use a simple capture mode to fix region errors")
        
        self.debug_logging_check = QCheckBox("Debug Logging")
        self.debug_logging_check.setChecked(SETTINGS["debug_logging"])
        self.debug_logging_check.toggled.connect(self.update_debug_logging)
        self.debug_logging_check.setToolTip("Enable debug output (may reduce performance)")
        
        self.half_precision_check = QCheckBox("Use Half Precision (FP16)")
        self.half_precision_check.setChecked(SETTINGS["half_precision"])
        self.half_precision_check.toggled.connect(self.update_half_precision)
        self.half_precision_check.setToolTip("Use FP16 for better performance (requires CUDA)")
        
        checks_layout.addWidget(self.show_overlay_check)
        checks_layout.addWidget(self.always_top_check)
        checks_layout.addWidget(self.play_area_check)
        checks_layout.addWidget(self.simple_capture_check)
        checks_layout.addWidget(self.debug_logging_check)
        checks_layout.addWidget(self.half_precision_check)
        
        # Config profile management
        profile_group = QGroupBox("Configuration Profiles")
        profile_layout = QVBoxLayout(profile_group)
        
        # Profile selection
        profile_combo_layout = QHBoxLayout()
        profile_combo_layout.addWidget(QLabel("Current Profile:"))
        self.profile_combo = QComboBox()
        self.profile_combo.setEditable(False)
        self.profile_combo.currentTextChanged.connect(self.load_profile)
        profile_combo_layout.addWidget(self.profile_combo, 1)  # 1 = stretch factor
        
        # Profile buttons
        profile_buttons_layout = QHBoxLayout()
        
        self.save_profile_btn = QPushButton("Save")
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.save_profile_btn.setToolTip("Save current settings to the selected profile")
        
        self.save_as_profile_btn = QPushButton("Save As")
        self.save_as_profile_btn.clicked.connect(self.save_profile_as)
        self.save_as_profile_btn.setToolTip("Save current settings to a new profile")
        
        self.delete_profile_btn = QPushButton("Delete")
        self.delete_profile_btn.clicked.connect(self.delete_profile)
        self.delete_profile_btn.setToolTip("Delete the selected profile")
        
        profile_buttons_layout.addWidget(self.save_profile_btn)
        profile_buttons_layout.addWidget(self.save_as_profile_btn)
        profile_buttons_layout.addWidget(self.delete_profile_btn)
        
        profile_layout.addLayout(profile_combo_layout)
        profile_layout.addLayout(profile_buttons_layout)
        
        # Input device selection
        input_device_group = QGroupBox("Input Device")
        input_device_layout = QVBoxLayout(input_device_group)
        
        self.mouse_radio = QRadioButton("Mouse (Default)")
        self.tablet_radio = QRadioButton("Tablet")
        
        # Set initial selection based on settings
        if SETTINGS["input_device"] == "tablet":
            self.tablet_radio.setChecked(True)
        else:
            self.mouse_radio.setChecked(True)
            
        # Connect signals
        self.mouse_radio.toggled.connect(lambda checked: self.update_input_device("mouse") if checked else None)
        self.tablet_radio.toggled.connect(lambda checked: self.update_input_device("tablet") if checked else None)
        
        input_device_layout.addWidget(self.mouse_radio)
        input_device_layout.addWidget(self.tablet_radio)
        
        checks_layout.addWidget(self.show_overlay_check)
        checks_layout.addWidget(self.always_top_check)
        
        # Toggle button
        self.toggle_button = QPushButton("Enable Aim Assist (F8)")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(SETTINGS["enabled"])
        self.toggle_button.clicked.connect(self.toggle_assist)
        self.update_toggle_button_style()
        
        # Add all groups to settings layout
        settings_layout.addWidget(model_group)
        settings_layout.addWidget(confidence_group)
        settings_layout.addWidget(strength_group)
        settings_layout.addWidget(aim_mode_group)
        settings_layout.addWidget(capture_group)
        settings_layout.addWidget(checks_group)
        settings_layout.addWidget(input_device_group)  # Add input device group
        settings_layout.addWidget(self.toggle_button)
        settings_layout.addWidget(profile_group)
        
        # Add stretch to push everything up
        settings_layout.addStretch()
        
        # Add to main layout
        main_layout.addWidget(preview_group, 2)
        main_layout.addWidget(settings_group, 1)
        
        # Update preview visibility
        self.update_preview_state()
    
    def init_detection_thread(self):
        self.detection_thread = DetectionThread(self)
        self.detection_thread.update_frame.connect(self.update_preview)
        self.detection_thread.update_fps.connect(self.update_fps)  # Connect FPS signal
        
        # Try to load the model if a path is already specified in settings
        if SETTINGS["model_path"]:
            if os.path.exists(SETTINGS["model_path"]):
                success = self.detection_thread.load_model(SETTINGS["model_path"])
                if success:
                    self.update_class_dropdown()
                    self.show_status_message(f"Model loaded successfully: {os.path.basename(SETTINGS['model_path'])}")
                    # Update CUDA indicator after model load
                    self.update_cuda_indicator()
                else:
                    self.show_status_message(f"Failed to load model from settings: {os.path.basename(SETTINGS['model_path'])}", is_error=True)
                    self.model_path_label.setText("Error loading model")
            else:
                self.show_status_message(f"Model file not found: {os.path.basename(SETTINGS['model_path'])}", is_error=True)
                self.model_path_label.setText("Model file not found")
                # Clear the invalid path
                SETTINGS["model_path"] = ""
        
        # Start the detection thread
        self.detection_thread.start()
        
        # Update the status indicator
        self.update_detection_status()
    
    def update_preview(self, frame, detections):
        if frame is None or not SETTINGS["preview_enabled"]:
            return
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Draw detections if overlay is enabled
        if SETTINGS["show_overlay"]:
            for det in detections:
                bbox = det['bbox']
                conf = det['confidence']
                cls = det['class']
                
                # Get class name - handle either dict or list
                if isinstance(CLASS_MAPPING, dict):
                    class_name = CLASS_MAPPING.get(cls, f"Class {cls}")
                else:
                    class_name = f"Class {cls}"
                
                # Convert bbox coordinates to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Draw bounding box (green for active note, blue for others)
                color = (0, 255, 0) if cls == SETTINGS["target_class"] else (255, 0, 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw confidence and class
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert to QImage
        h, w, c = display_frame.shape
        q_img = QImage(display_frame.data, w, h, w * c, QImage.Format.Format_RGB888)
        
        # Display in the label
        pixmap = QPixmap.fromImage(q_img)
        self.preview_label.setPixmap(pixmap.scaled(self.preview_label.width(), 
                                                 self.preview_label.height(),
                                                 Qt.AspectRatioMode.KeepAspectRatio))
    
    def toggle_preview(self):
        SETTINGS["preview_enabled"] = self.preview_toggle_btn.isChecked()
        self.update_preview_state()
        # Save setting
        self.save_settings()
    
    def update_preview_state(self):
        if SETTINGS["preview_enabled"]:
            self.preview_toggle_btn.setText("Disable Preview")
        else:
            self.preview_toggle_btn.setText("Enable Preview")
            self.preview_label.clear()
            self.preview_label.setText("Preview disabled")
    
    def select_model(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, "Select YOLOv11 Model", "", "Model Files (*.pt *.pth)")
        
        if model_path:
            SETTINGS["model_path"] = model_path
            self.model_path_label.setText(os.path.basename(model_path))
            
            # Make sure detection thread is initialized
            if not hasattr(self, 'detection_thread') or self.detection_thread is None:
                self.show_status_message("Detection thread not initialized. Please restart the application.", is_error=True)
                return
            
            # Load the model
            success = self.detection_thread.load_model(model_path)
            if success:
                # Update class dropdown after model load
                self.update_class_dropdown()
                self.show_status_message(f"Model loaded successfully: {os.path.basename(model_path)}")
                
                # Update CUDA indicator
                self.update_cuda_indicator()
            else:
                self.model_path_label.setText("Error loading model")
                self.show_status_message("Failed to load model. Please check if the model is valid.", is_error=True)
    
    def update_cuda_indicator(self):
        """Update CUDA status indicator in the UI"""
        if hasattr(self, 'detection_thread') and self.detection_thread is not None:
            if hasattr(self.detection_thread, 'cuda_available'):
                cuda_available = self.detection_thread.cuda_available
                if cuda_available:
                    self.cuda_indicator.setText("CUDA: Enabled")
                    self.cuda_indicator.setStyleSheet("color: green; font-weight: bold;")
                else:
                    self.cuda_indicator.setText("CUDA: Disabled (CPU)")
                    self.cuda_indicator.setStyleSheet("color: orange;")
            else:
                self.cuda_indicator.setText("CUDA: Unknown")
                self.cuda_indicator.setStyleSheet("color: gray;")
    
    def update_class_dropdown(self):
        # Clear current items
        self.target_class_combo.clear()
        
        # Add "All Classes" option
        self.target_class_combo.addItem("All Classes", -1)
        
        # Add classes from model - handle the case where CLASS_MAPPING is not a dict
        if isinstance(CLASS_MAPPING, dict):
            for idx, name in CLASS_MAPPING.items():
                self.target_class_combo.addItem(f"{name} ({idx})", idx)
            
            # Try to select active_note
            for i in range(self.target_class_combo.count()):
                if "active" in self.target_class_combo.itemText(i).lower():
                    self.target_class_combo.setCurrentIndex(i)
                    break
    
    def update_target_class(self, index):
        SETTINGS["target_class"] = self.target_class_combo.currentData()
        print(f"Target class changed to: {SETTINGS['target_class']}")
    
    def update_confidence(self):
        value = self.confidence_slider.value() / 100.0
        SETTINGS["confidence_threshold"] = value
        self.confidence_label.setText(f"Threshold: {value:.2f}")
        
        # Update model confidence if detection_thread exists and model is loaded
        if hasattr(self, 'detection_thread') and self.detection_thread is not None:
            if self.detection_thread.model is not None:
                # Show that we're updating confidence
                self.statusBar().showMessage(f"Updating confidence threshold to {value:.2f}...", 2000)
                
                # Update confidence (this will pause and resume the thread internally)
                self.detection_thread.update_confidence(value)
                
                # Update the status indicator
                self.update_detection_status()
                
                # Flash the confidence label briefly to indicate change
                original_style = self.confidence_label.styleSheet()
                self.confidence_label.setStyleSheet("color: green; font-weight: bold;")
                
                # Use a timer to reset the style after a brief period
                QTimer.singleShot(500, lambda: self.confidence_label.setStyleSheet(original_style))
    
    def update_strength(self):
        value = self.strength_slider.value() / 100.0
        SETTINGS["assist_strength"] = value
        self.strength_label.setText(f"Strength: {value:.2f}")
    
    def update_aim_mode(self, mode):
        SETTINGS["aim_mode"] = mode
    
    def update_capture_width(self, width):
        SETTINGS["capture_width"] = width
    
    def update_capture_height(self, height):
        SETTINGS["capture_height"] = height
    
    def update_target_fps(self, fps):
        SETTINGS["target_fps"] = fps
    
    def update_show_overlay(self, state):
        SETTINGS["show_overlay"] = state
    
    def update_always_on_top(self, state):
        SETTINGS["always_on_top"] = state
        flags = self.windowFlags()
        
        if state:
            self.setWindowFlags(flags | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(flags & ~Qt.WindowType.WindowStaysOnTopHint)
        
        self.show()
    
    def update_detection_status(self):
        """Update the status label to reflect the current detection state"""
        # Check if the detection thread exists
        if not hasattr(self, 'detection_thread') or self.detection_thread is None:
            self.status_label.setText("Detection: Not initialized")
            self.status_label.setStyleSheet("color: gray;")
            return
            
        if not SETTINGS["enabled"]:
            self.status_label.setText("Detection: Stopped")
            self.status_label.setStyleSheet("color: gray;")
        elif self.detection_thread.paused:
            self.status_label.setText("Detection: Paused")
            self.status_label.setStyleSheet("color: orange;")
        else:
            self.status_label.setText("Detection: Running")
            self.status_label.setStyleSheet("color: green;")
            
    def show_status_message(self, message, is_error=False):
        """Show a status message in the status bar"""
        if is_error:
            self.statusBar().setStyleSheet("color: red;")
        else:
            self.statusBar().setStyleSheet("")
            
        self.statusBar().showMessage(message, 3000)
        
        # Reset style after 3 seconds
        QTimer.singleShot(3000, lambda: self.statusBar().setStyleSheet(""))

    def toggle_assist(self):
        SETTINGS["enabled"] = not SETTINGS["enabled"]
        self.toggle_button.setChecked(SETTINGS["enabled"])
        self.update_toggle_button_style()
        
        # Update status indicator
        self.update_detection_status()
        
        # Show a status message
        if SETTINGS["enabled"]:
            self.show_status_message("Aim assist enabled")
        else:
            self.show_status_message("Aim assist disabled")
    
    def update_toggle_button_style(self):
        if SETTINGS["enabled"]:
            self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.toggle_button.setText("Disable Aim Assist (F8)")
        else:
            self.toggle_button.setStyleSheet("")
            self.toggle_button.setText("Enable Aim Assist (F8)")
    
    def load_settings(self):
        try:
            if os.path.exists(SETTINGS_FILE):
                with open(SETTINGS_FILE, 'r') as f:
                    saved_settings = json.load(f)
                    
                # Update global settings with saved values
                for key, value in saved_settings.items():
                    if key in SETTINGS:
                        # Skip the "enabled" key to ensure it starts disabled
                        if key != "enabled":
                            SETTINGS[key] = value
                
                # Load available profiles
                self.load_profiles()
                
                # Update UI with loaded settings
                self.confidence_slider.setValue(int(SETTINGS["confidence_threshold"] * 100))
                self.strength_slider.setValue(int(SETTINGS["assist_strength"] * 100))
                
                # Handle aim_mode properly
                if isinstance(SETTINGS["aim_mode"], int):
                    # If it's an integer index, use setCurrentIndex
                    if 0 <= SETTINGS["aim_mode"] < self.aim_mode_combo.count():
                        self.aim_mode_combo.setCurrentIndex(SETTINGS["aim_mode"])
                else:
                    # If it's a string, use setCurrentText
                    self.aim_mode_combo.setCurrentText(SETTINGS["aim_mode"])
                
                self.width_spinbox.setValue(SETTINGS["capture_width"])
                self.height_spinbox.setValue(SETTINGS["capture_height"])
                
                self.show_overlay_check.setChecked(SETTINGS["show_overlay"])
                self.always_top_check.setChecked(SETTINGS["always_on_top"])
                self.play_area_check.setChecked(SETTINGS["use_play_area"])
                self.simple_capture_check.setChecked(SETTINGS["simple_capture"])
                self.debug_logging_check.setChecked(SETTINGS["debug_logging"])
                self.half_precision_check.setChecked(SETTINGS["half_precision"])
                
                # Select the current profile in the combobox
                index = self.profile_combo.findText(SETTINGS["profile_name"])
                if index >= 0:
                    self.profile_combo.setCurrentIndex(index)
                
                print("Settings loaded successfully")
                
                # Make sure enabled is always false on startup
                SETTINGS["enabled"] = False
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        try:
            # Make sure profiles directory exists
            os.makedirs(PROFILES_DIR, exist_ok=True)
            
            # Save current settings to the main settings file
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(SETTINGS, f, indent=4)
            
            # Save current profile settings
            self.save_current_profile()
            
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("Settings saved", 2000)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def load_profiles(self):
        """Load available configuration profiles"""
        try:
            # Create profiles directory if it doesn't exist
            os.makedirs(PROFILES_DIR, exist_ok=True)
            
            # Disconnect signal temporarily to prevent triggering load_profile
            self.profile_combo.blockSignals(True)
            
            # Clear existing items
            self.profile_combo.clear()
            
            # Get profile files
            profile_files = [f for f in os.listdir(PROFILES_DIR) 
                            if f.endswith('.json') and os.path.isfile(os.path.join(PROFILES_DIR, f))]
            
            # Add default profile if no profiles exist
            if not profile_files:
                self.profile_combo.addItem("Default")
                # Save default profile
                self.save_current_profile()
            else:
                # Add available profiles
                for profile_file in profile_files:
                    profile_name = profile_file.replace('.json', '')
                    self.profile_combo.addItem(profile_name)
            
            # Select current profile
            index = self.profile_combo.findText(SETTINGS["profile_name"])
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
            else:
                # If current profile doesn't exist, select first profile
                self.profile_combo.setCurrentIndex(0)
                SETTINGS["profile_name"] = self.profile_combo.currentText()
            
            # Reconnect signal
            self.profile_combo.blockSignals(False)
        except Exception as e:
            print(f"Error loading profiles: {e}")
    
    def load_profile(self, profile_name):
        """Load a specific profile by name"""
        if not profile_name or self.profile_combo.signalsBlocked():
            return
            
        try:
            profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
            
            # Check if profile exists
            if os.path.exists(profile_path):
                with open(profile_path, 'r') as f:
                    profile_settings = json.load(f)
                
                # Update settings with profile values but ensure detection remains disabled
                for key, value in profile_settings.items():
                    if key in SETTINGS:
                        # Skip the "enabled" key to ensure it stays disabled
                        if key != "enabled":
                            SETTINGS[key] = value
                
                # Update profile name
                SETTINGS["profile_name"] = profile_name
                
                # Update UI with loaded settings
                self.confidence_slider.setValue(int(SETTINGS["confidence_threshold"] * 100))
                self.strength_slider.setValue(int(SETTINGS["assist_strength"] * 100))
                
                # Handle aim_mode properly
                if isinstance(SETTINGS["aim_mode"], int):
                    # If it's an integer index, use setCurrentIndex
                    if 0 <= SETTINGS["aim_mode"] < self.aim_mode_combo.count():
                        self.aim_mode_combo.setCurrentIndex(SETTINGS["aim_mode"])
                else:
                    # If it's a string, use setCurrentText
                    self.aim_mode_combo.setCurrentText(SETTINGS["aim_mode"])
                
                self.width_spinbox.setValue(SETTINGS["capture_width"])
                self.height_spinbox.setValue(SETTINGS["capture_height"])
                
                self.show_overlay_check.setChecked(SETTINGS["show_overlay"])
                self.always_top_check.setChecked(SETTINGS["always_on_top"])
                self.play_area_check.setChecked(SETTINGS["use_play_area"])
                self.simple_capture_check.setChecked(SETTINGS["simple_capture"])
                self.debug_logging_check.setChecked(SETTINGS["debug_logging"])
                self.half_precision_check.setChecked(SETTINGS["half_precision"])
                
                # Make sure enabled is always false when loading a profile
                SETTINGS["enabled"] = False
                self.update_detection_status()
                self.update_toggle_button_style()
                
                # Save updated settings
                self.save_settings()
                
                self.statusBar().showMessage(f"Loaded profile: {profile_name}", 2000)
        except Exception as e:
            print(f"Error loading profile '{profile_name}': {e}")
            self.statusBar().showMessage(f"Error loading profile", 2000)
    
    def save_profile(self):
        """Save current settings to selected profile"""
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            return
            
        try:
            self.save_current_profile()
            self.statusBar().showMessage(f"Saved profile: {profile_name}", 2000)
        except Exception as e:
            print(f"Error saving profile '{profile_name}': {e}")
            self.statusBar().showMessage(f"Error saving profile", 2000)
    
    def save_profile_as(self):
        """Save current settings to a new profile"""
        profile_name, ok = QInputDialog.getText(
            self, "Save Profile As", "Enter profile name:")
        
        if ok and profile_name:
            # Update current profile name
            SETTINGS["profile_name"] = profile_name
            
            # Add to combobox if not exists
            index = self.profile_combo.findText(profile_name)
            if index < 0:
                self.profile_combo.addItem(profile_name)
            
            # Select the new profile
            index = self.profile_combo.findText(profile_name)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
            
            # Save profile
            self.save_current_profile()
            
            self.statusBar().showMessage(f"Created profile: {profile_name}", 2000)
    
    def delete_profile(self):
        """Delete the selected profile"""
        profile_name = self.profile_combo.currentText()
        
        # Confirm deletion
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Are you sure you want to delete the profile '{profile_name}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                # Remove from combobox
                index = self.profile_combo.findText(profile_name)
                if index >= 0:
                    self.profile_combo.removeItem(index)
                
                # Delete profile file
                profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
                if os.path.exists(profile_path):
                    os.remove(profile_path)
                
                # Set new current profile
                if self.profile_combo.count() > 0:
                    SETTINGS["profile_name"] = self.profile_combo.currentText()
                else:
                    # Add default profile if no profiles exist
                    self.profile_combo.addItem("Default")
                    SETTINGS["profile_name"] = "Default"
                
                self.save_settings()
                self.statusBar().showMessage(f"Deleted profile: {profile_name}", 2000)
            except Exception as e:
                print(f"Error deleting profile '{profile_name}': {e}")
                self.statusBar().showMessage(f"Error deleting profile", 2000)
    
    def save_current_profile(self):
        """Save current settings to a profile file"""
        profile_name = SETTINGS["profile_name"]
        if not profile_name:
            profile_name = "Default"
            SETTINGS["profile_name"] = profile_name
        
        try:
            # Make sure profiles directory exists
            os.makedirs(PROFILES_DIR, exist_ok=True)
            
            # Save profile to file
            profile_path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
            with open(profile_path, 'w') as f:
                json.dump(SETTINGS, f, indent=4)
        except Exception as e:
            print(f"Error saving profile '{profile_name}': {e}")
    
    def update_input_device(self, device_type):
        """Update the input device setting"""
        SETTINGS["input_device"] = device_type
        self.statusBar().showMessage(f"Using {device_type.title()} as input device", 2000)
        print(f"Input device changed to: {device_type}")
        self.save_settings()

    def update_fps(self, fps):
        """Update the FPS counter label with the current FPS"""
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        # Color coding based on FPS
        if fps < 15:
            # Low FPS - red
            self.fps_label.setStyleSheet("color: red; font-weight: bold;")
        elif fps < 30:
            # Medium FPS - orange
            self.fps_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            # Good FPS - green
            self.fps_label.setStyleSheet("color: green; font-weight: bold;")

    def update_play_area(self, state):
        SETTINGS["use_play_area"] = state
        self.save_settings()

    def update_simple_capture(self, state):
        SETTINGS["simple_capture"] = state
        self.save_settings()

    def update_debug_logging(self, state):
        SETTINGS["debug_logging"] = state
        self.save_settings()

    def update_half_precision(self, state):
        SETTINGS["half_precision"] = state
        self.save_settings()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())