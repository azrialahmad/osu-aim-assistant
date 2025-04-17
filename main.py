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
import _ctypes  # Add explicit import for _ctypes for COM error handling
import mouse  # For tablet support
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                           QWidget, QLabel, QSlider, QPushButton, QCheckBox, 
                           QGroupBox, QComboBox, QSpinBox, QFileDialog, QMessageBox,
                           QRadioButton, QButtonGroup, QInputDialog, QTabWidget)
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
    "capture_width": 1920,
    "capture_height": 1080,
    "model_path": "",
    "show_overlay": True,
    "target_fps": 144,
    "always_on_top": False,
    "preview_enabled": False,  # Set to False by default for better performance
    "preview_process_disabled": True,  # Completely disable preview processing
    "target_class": 0,  # 0 for active_note
    "multi_target_tracking": False,  # Allow tracking multiple target types
    "track_slider_ball": False,      # Track slider_ball in addition to primary target
    "slider_ball_class": 2,          # Default class index for slider_ball
    "input_device": "mouse",
    "use_play_area": False,
    "simple_capture": True,
    "debug_logging": False,
    "profile_name": "Default",
    "half_precision": False,
    # Advanced smoothing options
    "advanced_smoothing_enabled": False,  # Toggle for advanced smoothing
    "smoothing_factor": 0.2,     # Default EMA smoothing factor
    "bezier_weight": 0.5,        # Bezier curve weight
    "catmull_tension": 0.5,      # Catmull-Rom tension parameter
    "speed_scaling": True,       # Enable speed scaling based on distance
    "min_distance_threshold": 5  # Minimum distance threshold for movement
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
                elif isinstance(CLASS_MAPPING, list):
                    for idx, name in enumerate(CLASS_MAPPING):
                        if isinstance(name, str) and "active" in name.lower():
                            SETTINGS["target_class"] = idx
                            print(f"Setting target class to {idx}: {name}")
                            break
            else:
                print("Model has no 'names' attribute")
                CLASS_MAPPING = {}  # Reset to empty
            
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
                                
                                # Check class for detection:
                                # - If target_class is -1, detect all classes
                                # - Otherwise, detect target_class and slider_ball (class 2) if enabled
                                is_target_class = SETTINGS["target_class"] == -1 or cls_idx == SETTINGS["target_class"]
                                is_slider_ball = SETTINGS["track_slider_ball"] and cls_idx == SETTINGS["slider_ball_class"]
                                
                                if (is_target_class or is_slider_ball) and conf >= SETTINGS["confidence_threshold"]:
                                    detections.append({
                                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                        'confidence': float(conf),
                                        'class': cls_idx
                                    })
                        
                        # Update the UI with the frame and detections (only if preview is enabled)
                        if SETTINGS["preview_enabled"] and not SETTINGS["preview_process_disabled"]:
                            try:
                                self.update_frame.emit(frame_rgb, detections)
                            except Exception as e:
                                # If emitting the frame fails, log but continue
                                print(f"Error updating preview frame: {e}")
                        
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
                except _ctypes.COMError as com_error:
                    # Handle DXCAM specific COM errors
                    if "AcquireNextFrame" in str(com_error):
                        # This is likely a transient error with screen capture, pause briefly
                        print("DXCAM frame acquisition error. Recreating camera...")
                        time.sleep(0.5)
                        # Recreate the camera
                        if 'camera' in locals() and camera is not None:
                            try:
                                camera.stop()
                            except:
                                pass
                        camera, screen_width, screen_height = self.initialize_dxcam()
                    else:
                        # Other COM errors, log and continue
                        print(f"COM Error in detection thread: {com_error}")
                        time.sleep(0.5)
                except Exception as e:
                    # Handle general exceptions
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
        # Sort detections by class first (prioritize target class over slider ball),
        # then by confidence (highest first)
        def detection_priority(det):
            # Lower value = higher priority
            # Give preference to the target class (usually active notes)
            class_priority = 1 if det['class'] == SETTINGS["target_class"] else 0
            
            # If target_class is -1 (all classes), just use confidence
            if SETTINGS["target_class"] == -1:
                return det['confidence']
                
            # Return combined priority (class is more important than confidence)
            return (class_priority, det['confidence'])
            
        # Sort detections by priority
        sorted_detections = sorted(detections, key=detection_priority, reverse=True)
        
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
            
            # Calculate distance to target
            distance = ((dx**2 + dy**2)**0.5)
            
            # Check if we need minimum distance threshold filtering
            if SETTINGS["advanced_smoothing_enabled"] and distance < SETTINGS["min_distance_threshold"]:
                return  # Skip if target is too close and advanced smoothing is on
            
            # Get strength - use a consistent base multiplier
            strength = SETTINGS["assist_strength"]
            
            # Check if advanced smoothing is enabled
            advanced_smoothing = SETTINGS["advanced_smoothing_enabled"]
            
            # Apply different aim modes with improved algorithms
            if SETTINGS["aim_mode"] == "snap":
                # Direct snap toward target with no smoothing
                # For snap mode, we don't smooth the movement at all and use higher multiplier
                move_x = strength * dx * 1.0  # Much stronger effect for snappier movement
                move_y = strength * dy * 1.0
                
            elif SETTINGS["aim_mode"] == "smooth":
                # Apply EMA smoothing for the smooth mode
                if not hasattr(self, 'last_dx'):
                    self.last_dx = 0
                    self.last_dy = 0
                
                # Get EMA smoothing factor from settings if advanced smoothing enabled
                # Otherwise use a default value
                ema_factor = SETTINGS["smoothing_factor"] if advanced_smoothing else 0.2
                
                # Apply EMA smoothing
                smoothed_dx = ema_factor * dx + (1 - ema_factor) * self.last_dx
                smoothed_dy = ema_factor * dy + (1 - ema_factor) * self.last_dy
                
                # Update last values
                self.last_dx = smoothed_dx
                self.last_dy = smoothed_dy
                
                # Calculate speed reduction factor - maintain direction but adjust speed
                move_x = smoothed_dx * strength * 0.15
                move_y = smoothed_dy * strength * 0.15
                
            elif SETTINGS["aim_mode"] == "bezier" and advanced_smoothing:
                # Bezier curve smoothing (only available with advanced smoothing)
                # We'll use a simple quadratic Bezier curve between current position and target
                
                # Initialize control points array if not exists
                if not hasattr(self, 'control_points'):
                    self.control_points = []
                    
                # Add current target as control point
                self.control_points.append((dx, dy))
                
                # Keep only the last 3 points for Bezier calculation
                if len(self.control_points) > 3:
                    self.control_points.pop(0)
                
                # If we have enough points, calculate Bezier interpolation
                if len(self.control_points) >= 2:
                    # Use the current vector and previous vector for interpolation
                    current = self.control_points[-1]
                    previous = self.control_points[-2]
                    
                    # Calculate Bezier point (linear if only 2 points, quadratic if 3)
                    if len(self.control_points) == 2:
                        # Linear Bezier (weighted average)
                        t = SETTINGS["bezier_weight"]  # Get weight from settings
                        smoothed_dx = (1-t)*previous[0] + t*current[0]
                        smoothed_dy = (1-t)*previous[1] + t*current[1]
                    else:
                        # Quadratic Bezier with 3 points
                        t = SETTINGS["bezier_weight"]  # Get weight from settings
                        p0 = self.control_points[0]
                        p1 = self.control_points[1]
                        p2 = self.control_points[2]
                        
                        # B(t) = (1-t)²P₀ + 2(1-t)tP₁ + t²P₂
                        t_inv = 1 - t
                        smoothed_dx = t_inv*t_inv*p0[0] + 2*t_inv*t*p1[0] + t*t*p2[0]
                        smoothed_dy = t_inv*t_inv*p0[1] + 2*t_inv*t*p1[1] + t*t*p2[1]
                else:
                    # Not enough points, use raw vector
                    smoothed_dx = dx
                    smoothed_dy = dy
                
                # Apply strength modifier - maintain path but adjust speed
                move_x = smoothed_dx * strength * 0.1
                move_y = smoothed_dy * strength * 0.1
            
            elif SETTINGS["aim_mode"] == "catmull" and advanced_smoothing:
                # Catmull-Rom spline interpolation for smoother curve
                
                # Initialize control points array if not exists
                if not hasattr(self, 'catmull_points'):
                    self.catmull_points = []
                
                # Add current target as control point
                self.catmull_points.append((dx, dy))
                
                # Keep only the last 4 points for Catmull-Rom calculation
                if len(self.catmull_points) > 4:
                    self.catmull_points.pop(0)
                
                # If we have enough points, calculate Catmull-Rom interpolation
                if len(self.catmull_points) >= 4:
                    # Use 4 points for Catmull-Rom interpolation
                    p0 = self.catmull_points[0]
                    p1 = self.catmull_points[1]
                    p2 = self.catmull_points[2]
                    p3 = self.catmull_points[3]
                    
                    # Catmull-Rom interpolation parameter
                    t = SETTINGS["catmull_tension"]
                    
                    # Catmull-Rom formula:
                    # S(t) = 0.5 * ((2*P1) + (-P0 + P2) * t + (2*P0 - 5*P1 + 4*P2 - P3) * t^2 + (-P0 + 3*P1 - 3*P2 + P3) * t^3)
                    t2 = t * t
                    t3 = t2 * t
                    
                    # X component
                    smoothed_dx = 0.5 * ((2*p1[0]) + 
                                        (-p0[0] + p2[0]) * t + 
                                        (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2 + 
                                        (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                    
                    # Y component
                    smoothed_dy = 0.5 * ((2*p1[1]) + 
                                        (-p0[1] + p2[1]) * t + 
                                        (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2 + 
                                        (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                else:
                    # Not enough points for Catmull-Rom, use raw or simple smoothing
                    if len(self.catmull_points) <= 1:
                        smoothed_dx = dx
                        smoothed_dy = dy
                    else:
                        # Simple average of available points
                        sum_x = sum(p[0] for p in self.catmull_points)
                        sum_y = sum(p[1] for p in self.catmull_points)
                        count = len(self.catmull_points)
                        smoothed_dx = sum_x / count
                        smoothed_dy = sum_y / count
                
                # Apply strength modifier
                move_x = smoothed_dx * strength * 0.1
                move_y = smoothed_dy * strength * 0.1
                
            else:  # Default to interpolate or fallback for non-advanced modes
                # Basic smoothing approach
                if not hasattr(self, 'last_dx'):
                    self.last_dx = 0
                    self.last_dy = 0
                
                # Use a light smoothing factor
                ema_factor = SETTINGS["smoothing_factor"] * 1.5 if advanced_smoothing else 0.3
                ema_factor = min(0.5, ema_factor)  # Cap at 0.5
                
                # Apply light smoothing
                smoothed_dx = ema_factor * dx + (1 - ema_factor) * self.last_dx
                smoothed_dy = ema_factor * dy + (1 - ema_factor) * self.last_dy
                
                # Update last values
                self.last_dx = smoothed_dx
                self.last_dy = smoothed_dy
                
                # Apply movement with speed adjustment
                move_x = smoothed_dx * strength * 0.08
                move_y = smoothed_dy * strength * 0.08
            
            # Only apply speed scaling if enabled in settings and advanced smoothing is on
            if advanced_smoothing and SETTINGS["speed_scaling"]:
                # Normalize the movement vector to preserve direction
                # This ensures we only modify speed, not the direction of movement
                magnitude = (move_x**2 + move_y**2)**0.5
                if magnitude > 0:
                    # Get original direction
                    dir_x = move_x / magnitude
                    dir_y = move_y / magnitude
                    
                    # Apply a speed curve based on distance
                    # Closer = slower, farther = faster
                    speed_factor = min(1.0, distance / 500.0)  # Cap speed boost
                    
                    # Recalculate movement with speed adjustment
                    adjusted_speed = magnitude * (0.5 + 0.5 * speed_factor)
                    move_x = dir_x * adjusted_speed
                    move_y = dir_y * adjusted_speed
            
            # Apply movement if significant enough
            if abs(move_x) > 0.2 or abs(move_y) > 0.2:
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

class PreviewWindow(QMainWindow):
    """Separate window for preview display"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detection Preview")
        self.setMinimumSize(800, 600)
        
        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Preview label
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setText("Waiting for detection...")
        
        # Status indicators
        status_layout = QHBoxLayout()
        
        # FPS counter
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # Status label
        self.status_label = QLabel("Detection: Stopped")
        self.status_label.setStyleSheet("color: gray;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)
        
        main_layout.addWidget(self.preview_label, 1)
        main_layout.addLayout(status_layout)
        
        # Set window flags
        self.setWindowFlags(Qt.WindowType.Window)
    
    def update_preview(self, frame, detections):
        """Update the preview with detection results"""
        if frame is None:
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
    
    def update_fps(self, fps):
        """Update the FPS counter label"""
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
    
    def update_detection_status(self):
        """Update the status label"""
        if not SETTINGS["enabled"]:
            self.status_label.setText("Detection: Stopped")
            self.status_label.setStyleSheet("color: gray;")
        else:
            self.status_label.setText("Detection: Running")
            self.status_label.setStyleSheet("color: green;")

class TabWidget(QTabWidget):
    """Main tab widget that contains all settings panels"""
    settings_changed = pyqtSignal()  # Signal when settings change
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create tabs for different settings
        self.settings_tab = QWidget()
        self.capture_tab = QWidget()
        self.advanced_tab = QWidget()
        
        # Add tabs to widget
        self.addTab(self.settings_tab, "Model & Aim")
        self.addTab(self.capture_tab, "Capture")
        self.addTab(self.advanced_tab, "Advanced")
        
        # Initialize tab UIs
        self.init_settings_tab()
        self.init_capture_tab()
        self.init_advanced_tab()
    
    def init_settings_tab(self):
        """Initialize the combined model and aim settings tab"""
        layout = QVBoxLayout(self.settings_tab)
        
        # Model section
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)
        
        # Model path and selection
        model_path_layout = QHBoxLayout()
        self.model_path_label = QLabel("No model selected")
        model_select_btn = QPushButton("Select Model")
        model_select_btn.clicked.connect(self.select_model)
        model_path_layout.addWidget(self.model_path_label, 1)
        model_path_layout.addWidget(model_select_btn)
        
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
        target_class_layout.addWidget(self.target_class_combo, 1)
        
        # Confidence threshold
        confidence_layout = QVBoxLayout()
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(0, 100)
        self.confidence_slider.setValue(int(SETTINGS["confidence_threshold"] * 100))
        self.confidence_slider.valueChanged.connect(self.update_confidence)
        self.confidence_label = QLabel(f"Threshold: {SETTINGS['confidence_threshold']:.2f}")
        confidence_layout.addWidget(self.confidence_label)
        confidence_layout.addWidget(self.confidence_slider)
        
        # Add model layouts
        model_layout.addLayout(model_path_layout)
        model_layout.addLayout(target_class_layout)
        model_layout.addLayout(confidence_layout)
        
        # Aim section
        aim_group = QGroupBox("Aim Assist")
        aim_layout = QVBoxLayout(aim_group)
        
        # Assist strength
        strength_layout = QVBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(SETTINGS["assist_strength"] * 100))
        self.strength_slider.valueChanged.connect(self.update_strength)
        self.strength_label = QLabel(f"Strength: {SETTINGS['assist_strength']:.2f}")
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_slider)
        
        # Aim mode
        aim_mode_layout = QHBoxLayout()
        aim_mode_layout.addWidget(QLabel("Aim Mode:"))
        self.aim_mode_combo = QComboBox()
        
        # Add aim modes with descriptions
        aim_modes = [
            "snap",          # No smoothing, direct movement
            "smooth",        # EMA smoothing
            "bezier",        # Bezier curve interpolation
            "catmull",       # Catmull-Rom spline smoothing
            "interpolate"    # Basic interpolation
        ]
        aim_mode_descriptions = {
            "snap": "Direct movement with no smoothing",
            "smooth": "Smooth movement with EMA filtering",
            "bezier": "Bezier curve for fluid, predictable motion",
            "catmull": "Catmull-Rom spline for natural curves",
            "interpolate": "Basic interpolation (balanced)"
        }
        
        self.aim_mode_combo.addItems(aim_modes)
        
        # Add tooltip with descriptions
        for i, mode in enumerate(aim_modes):
            self.aim_mode_combo.setItemData(i, aim_mode_descriptions[mode], Qt.ItemDataRole.ToolTipRole)
        
        # Set current aim mode - handle both string and integer values
        if isinstance(SETTINGS["aim_mode"], int):
            # If it's an integer index, use setCurrentIndex
            if 0 <= SETTINGS["aim_mode"] < len(aim_modes):
                self.aim_mode_combo.setCurrentIndex(SETTINGS["aim_mode"])
        else:
            # If it's a string, use setCurrentText
            self.aim_mode_combo.setCurrentText(SETTINGS["aim_mode"])
            
        self.aim_mode_combo.currentTextChanged.connect(self.update_aim_mode)
        aim_mode_layout.addWidget(self.aim_mode_combo, 1)
        
        # Input device selection
        input_device_layout = QHBoxLayout()
        input_device_layout.addWidget(QLabel("Input Device:"))
        self.input_device_combo = QComboBox()
        self.input_device_combo.addItems(["Mouse", "Tablet"])
        self.input_device_combo.setCurrentText("Tablet" if SETTINGS["input_device"] == "tablet" else "Mouse")
        self.input_device_combo.currentTextChanged.connect(
            lambda text: self.update_input_device("tablet" if text == "Tablet" else "mouse"))
        input_device_layout.addWidget(self.input_device_combo, 1)
        
        # Add aim layouts
        aim_layout.addLayout(strength_layout)
        aim_layout.addLayout(aim_mode_layout)
        aim_layout.addLayout(input_device_layout)
        
        # Add toggle button at the bottom
        self.toggle_button = QPushButton("Enable Aim Assist (F8)")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(SETTINGS["enabled"])
        self.toggle_button.clicked.connect(self.toggle_assist)
        self.update_toggle_button_style()
        
        # Add all to the layout
        layout.addWidget(model_group)
        layout.addWidget(aim_group)
        layout.addWidget(self.toggle_button)
        layout.addStretch(1)
    
    def init_capture_tab(self):
        """Initialize the capture settings tab"""
        layout = QVBoxLayout(self.capture_tab)
        
        # Capture settings
        capture_group = QGroupBox("Screen Capture")
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
        
        # Capture options
        self.play_area_check = QCheckBox("Optimize for Play Area (4:3)")
        self.play_area_check.setChecked(SETTINGS["use_play_area"])
        self.play_area_check.toggled.connect(self.update_play_area)
        self.play_area_check.setToolTip("Only capture the 4:3 play area for better performance")
        
        self.simple_capture_check = QCheckBox("Simple Capture Mode (Fix Errors)")
        self.simple_capture_check.setChecked(SETTINGS["simple_capture"])
        self.simple_capture_check.toggled.connect(self.update_simple_capture)
        self.simple_capture_check.setToolTip("Use a simple capture mode to fix region errors")
        
        self.show_overlay_check = QCheckBox("Show Detection Overlay")
        self.show_overlay_check.setChecked(SETTINGS["show_overlay"])
        self.show_overlay_check.toggled.connect(self.update_show_overlay)
        
        self.always_top_check = QCheckBox("Always On Top")
        self.always_top_check.setChecked(SETTINGS["always_on_top"])
        self.always_top_check.toggled.connect(self.update_always_on_top)
        
        # Preview disable option
        self.disable_preview_process_check = QCheckBox("Disable Preview Processing (Performance)")
        self.disable_preview_process_check.setChecked(SETTINGS["preview_process_disabled"])
        self.disable_preview_process_check.toggled.connect(self.update_disable_preview_process)
        self.disable_preview_process_check.setToolTip("Completely disable preview processing for maximum performance")
        
        # Preview button
        self.preview_toggle_btn = QPushButton("Show Preview Window")
        self.preview_toggle_btn.clicked.connect(self.toggle_preview)
        
        # Add all to capture layout
        capture_layout.addLayout(width_layout)
        capture_layout.addLayout(height_layout)
        capture_layout.addLayout(fps_layout)
        capture_layout.addWidget(self.play_area_check)
        capture_layout.addWidget(self.simple_capture_check)
        capture_layout.addWidget(self.show_overlay_check)
        capture_layout.addWidget(self.always_top_check)
        capture_layout.addWidget(self.disable_preview_process_check)
        capture_layout.addWidget(self.preview_toggle_btn)
        
        # Add to main layout
        layout.addWidget(capture_group)
        layout.addStretch(1)
    
    def init_advanced_tab(self):
        """Initialize the advanced settings tab"""
        layout = QVBoxLayout(self.advanced_tab)
        
        # Multi-target tracking options
        multi_target_group = QGroupBox("Multi-Target Tracking")
        multi_target_layout = QVBoxLayout(multi_target_group)
        
        # Enable multi-target tracking
        self.multi_target_check = QCheckBox("Enable Multi-Target Tracking")
        self.multi_target_check.setChecked(SETTINGS["multi_target_tracking"])
        self.multi_target_check.toggled.connect(self.update_multi_target)
        self.multi_target_check.setToolTip("Track multiple object types simultaneously")
        
        # Track slider ball
        self.slider_ball_check = QCheckBox("Track Slider Ball")
        self.slider_ball_check.setChecked(SETTINGS["track_slider_ball"])
        self.slider_ball_check.toggled.connect(self.update_track_slider_ball)
        self.slider_ball_check.setToolTip("Track slider ball in addition to primary target")
        self.slider_ball_check.setEnabled(SETTINGS["multi_target_tracking"])  # Only enabled if multi-target is on
        
        # Slider ball class selector
        slider_class_layout = QHBoxLayout()
        slider_class_layout.addWidget(QLabel("Slider Ball Class:"))
        self.slider_class_combo = QComboBox()
        
        # Set default slider ball class to 2
        SETTINGS["slider_ball_class"] = 2
        
        if isinstance(CLASS_MAPPING, dict):
            for idx, name in CLASS_MAPPING.items():
                self.slider_class_combo.addItem(f"{name} ({idx})", idx)
        
        # Set current selection to slider_ball (class 2)
        for i in range(self.slider_class_combo.count()):
            if self.slider_class_combo.itemData(i) == 2:
                self.slider_class_combo.setCurrentIndex(i)
                break
        
        self.slider_class_combo.currentIndexChanged.connect(self.update_slider_ball_class)
        self.slider_class_combo.setEnabled(SETTINGS["multi_target_tracking"] and SETTINGS["track_slider_ball"])
        slider_class_layout.addWidget(self.slider_class_combo, 1)
        
        # Advanced smoothing options
        adv_smoothing_group = QGroupBox("Advanced Smoothing")
        adv_smoothing_layout = QVBoxLayout(adv_smoothing_group)
        
        self.adv_smoothing_check = QCheckBox("Enable Advanced Smoothing")
        self.adv_smoothing_check.setChecked(SETTINGS["advanced_smoothing_enabled"])
        self.adv_smoothing_check.toggled.connect(self.toggle_advanced_smoothing)
        
        # Smoothing factor slider
        smoothing_layout = QVBoxLayout()
        self.smoothing_label = QLabel(f"Smoothing Factor: {SETTINGS['smoothing_factor']:.2f}")
        self.smoothing_slider = QSlider(Qt.Orientation.Horizontal)
        self.smoothing_slider.setRange(1, 99)
        self.smoothing_slider.setValue(int(SETTINGS['smoothing_factor'] * 100))
        self.smoothing_slider.valueChanged.connect(self.update_smoothing_factor)
        smoothing_layout.addWidget(self.smoothing_label)
        smoothing_layout.addWidget(self.smoothing_slider)
        
        # Bezier weight slider
        bezier_layout = QVBoxLayout()
        self.bezier_label = QLabel(f"Bezier Weight: {SETTINGS['bezier_weight']:.2f}")
        self.bezier_slider = QSlider(Qt.Orientation.Horizontal)
        self.bezier_slider.setRange(1, 99)
        self.bezier_slider.setValue(int(SETTINGS['bezier_weight'] * 100))
        self.bezier_slider.valueChanged.connect(self.update_bezier_weight)
        bezier_layout.addWidget(self.bezier_label)
        bezier_layout.addWidget(self.bezier_slider)
        
        # Catmull tension slider
        catmull_layout = QVBoxLayout()
        self.catmull_label = QLabel(f"Catmull Tension: {SETTINGS['catmull_tension']:.2f}")
        self.catmull_slider = QSlider(Qt.Orientation.Horizontal)
        self.catmull_slider.setRange(1, 99)
        self.catmull_slider.setValue(int(SETTINGS['catmull_tension'] * 100))
        self.catmull_slider.valueChanged.connect(self.update_catmull_tension)
        catmull_layout.addWidget(self.catmull_label)
        catmull_layout.addWidget(self.catmull_slider)
        
        # Min distance threshold
        min_dist_layout = QVBoxLayout()
        self.min_dist_label = QLabel(f"Min Distance Threshold: {SETTINGS['min_distance_threshold']}")
        self.min_dist_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_dist_slider.setRange(0, 50)
        self.min_dist_slider.setValue(SETTINGS["min_distance_threshold"])
        self.min_dist_slider.valueChanged.connect(self.update_min_distance)
        min_dist_layout.addWidget(self.min_dist_label)
        min_dist_layout.addWidget(self.min_dist_slider)
        
        self.speed_scaling_check = QCheckBox("Speed Scaling (Distance-based)")
        self.speed_scaling_check.setChecked(SETTINGS["speed_scaling"])
        self.speed_scaling_check.toggled.connect(self.update_speed_scaling)
        self.speed_scaling_check.setToolTip("Adjust cursor speed based on distance to target")
        
        # Model options
        model_options_group = QGroupBox("Model Options")
        model_options_layout = QVBoxLayout(model_options_group)
        
        self.half_precision_check = QCheckBox("Use Half Precision (FP16)")
        self.half_precision_check.setChecked(SETTINGS["half_precision"])
        self.half_precision_check.toggled.connect(self.update_half_precision)
        self.half_precision_check.setToolTip("Use FP16 for better performance (requires CUDA)")
        
        self.debug_logging_check = QCheckBox("Debug Logging")
        self.debug_logging_check.setChecked(SETTINGS["debug_logging"])
        self.debug_logging_check.toggled.connect(self.update_debug_logging)
        self.debug_logging_check.setToolTip("Enable debug output (may reduce performance)")
        
        model_options_layout.addWidget(self.half_precision_check)
        model_options_layout.addWidget(self.debug_logging_check)
        
        # Add to multi-target layout
        multi_target_layout.addWidget(self.multi_target_check)
        multi_target_layout.addWidget(self.slider_ball_check)
        multi_target_layout.addLayout(slider_class_layout)
        
        # Add to smoothing layout
        adv_smoothing_layout.addWidget(self.adv_smoothing_check)
        adv_smoothing_layout.addLayout(smoothing_layout)
        adv_smoothing_layout.addLayout(bezier_layout)
        adv_smoothing_layout.addLayout(catmull_layout)
        adv_smoothing_layout.addLayout(min_dist_layout)
        adv_smoothing_layout.addWidget(self.speed_scaling_check)
        
        # Add groups to main layout
        layout.addWidget(multi_target_group)
        layout.addWidget(adv_smoothing_group)
        layout.addWidget(model_options_group)
        layout.addStretch(1)
    
    # All the event handlers for settings changes
    def select_model(self):
        file_dialog = QFileDialog()
        model_path, _ = file_dialog.getOpenFileName(self, "Select YOLOv11 Model", "", "Model Files (*.pt *.pth)")
        
        if model_path:
            SETTINGS["model_path"] = model_path
            self.model_path_label.setText(os.path.basename(model_path))
            self.settings_changed.emit()  # Signal that settings changed
    
    def update_target_class(self, index):
        SETTINGS["target_class"] = self.target_class_combo.currentData()
        self.settings_changed.emit()
    
    def update_multi_target(self, state):
        SETTINGS["multi_target_tracking"] = state
        # Update dependent UI elements
        self.slider_ball_check.setEnabled(state)
        self.slider_class_combo.setEnabled(state and SETTINGS["track_slider_ball"])
        self.settings_changed.emit()
    
    def update_track_slider_ball(self, state):
        SETTINGS["track_slider_ball"] = state
        # Update dependent UI elements
        self.slider_class_combo.setEnabled(state and SETTINGS["multi_target_tracking"])
        self.settings_changed.emit()
    
    def update_slider_ball_class(self, index):
        SETTINGS["slider_ball_class"] = self.slider_class_combo.currentData()
        self.settings_changed.emit()
    
    def update_confidence(self, value):
        confidence = value / 100.0
        SETTINGS["confidence_threshold"] = confidence
        self.confidence_label.setText(f"Threshold: {confidence:.2f}")
        self.settings_changed.emit()
    
    def update_strength(self, value):
        strength = value / 100.0
        SETTINGS["assist_strength"] = strength
        self.strength_label.setText(f"Strength: {strength:.2f}")
        self.settings_changed.emit()
    
    def update_aim_mode(self, mode):
        SETTINGS["aim_mode"] = mode
        self.settings_changed.emit()
    
    def update_input_device(self, device_type):
        SETTINGS["input_device"] = device_type
        self.settings_changed.emit()
    
    def update_capture_width(self, width):
        SETTINGS["capture_width"] = width
        self.settings_changed.emit()
    
    def update_capture_height(self, height):
        SETTINGS["capture_height"] = height
        self.settings_changed.emit()
    
    def update_target_fps(self, fps):
        SETTINGS["target_fps"] = fps
        self.settings_changed.emit()
    
    def update_play_area(self, state):
        SETTINGS["use_play_area"] = state
        self.settings_changed.emit()
    
    def update_simple_capture(self, state):
        SETTINGS["simple_capture"] = state
        self.settings_changed.emit()
    
    def update_show_overlay(self, state):
        SETTINGS["show_overlay"] = state
        self.settings_changed.emit()
    
    def update_always_on_top(self, state):
        SETTINGS["always_on_top"] = state
        self.settings_changed.emit()
    
    def update_disable_preview_process(self, state):
        """Update the preview process disabled setting"""
        SETTINGS["preview_process_disabled"] = state
        
        # If disabling preview processing, also hide the preview if it's visible
        if state and self.preview_window.isVisible():
            self.preview_window.hide()
            self.preview_btn.setText("Show Preview")
            SETTINGS["preview_enabled"] = False
            
        self.settings_changed.emit()
        
        # Show a status message
        if state:
            self.show_status_message("Preview processing disabled for better performance")
        else:
            self.show_status_message("Preview processing enabled")
    
    def toggle_advanced_smoothing(self, enabled):
        SETTINGS["advanced_smoothing_enabled"] = enabled
        self.settings_changed.emit()
    
    def update_smoothing_factor(self, value):
        factor = value / 100.0
        SETTINGS["smoothing_factor"] = factor
        self.smoothing_label.setText(f"Smoothing Factor: {factor:.2f}")
        self.settings_changed.emit()
    
    def update_bezier_weight(self, value):
        weight = value / 100.0
        SETTINGS["bezier_weight"] = weight
        self.bezier_label.setText(f"Bezier Weight: {weight:.2f}")
        self.settings_changed.emit()
    
    def update_catmull_tension(self, value):
        tension = value / 100.0
        SETTINGS["catmull_tension"] = tension
        self.catmull_label.setText(f"Catmull Tension: {tension:.2f}")
        self.settings_changed.emit()
    
    def update_min_distance(self, value):
        SETTINGS["min_distance_threshold"] = value
        self.min_dist_label.setText(f"Min Distance Threshold: {value}")
        self.settings_changed.emit()
    
    def update_speed_scaling(self, state):
        SETTINGS["speed_scaling"] = state
        self.settings_changed.emit()
    
    def update_half_precision(self, state):
        SETTINGS["half_precision"] = state
        self.settings_changed.emit()
    
    def update_debug_logging(self, state):
        SETTINGS["debug_logging"] = state
        self.settings_changed.emit()
    
    def toggle_assist(self):
        SETTINGS["enabled"] = not SETTINGS["enabled"]
        self.toggle_button.setChecked(SETTINGS["enabled"])
        self.update_toggle_button_style()
        self.settings_changed.emit()
    
    def update_toggle_button_style(self):
        if SETTINGS["enabled"]:
            self.toggle_button.setStyleSheet("background-color: #4CAF50; color: white;")
            self.toggle_button.setText("Disable Aim Assist (F8)")
        else:
            self.toggle_button.setStyleSheet("")
            self.toggle_button.setText("Enable Aim Assist (F8)")
    
    def toggle_preview(self):
        # This will be connected to the MainWindow's toggle_preview method
        pass

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.detection_thread = None
        
        # Force detection to be disabled on startup
        SETTINGS["enabled"] = False
        
        # Create preview window
        self.preview_window = PreviewWindow(self)
        
        # Set up the UI
        self.init_ui()
        
        # Load settings (must be done after UI setup)
        self.load_settings()
        
        # Force detection to be disabled again after loading settings 
        SETTINGS["enabled"] = False
        self.update_detection_status()
        
        # Update preview button text based on settings
        if SETTINGS["preview_enabled"]:
            self.preview_btn.setText("Hide Preview")
        else:
            self.preview_btn.setText("Show Preview")
        
        # Initialize the detection thread (must be done after settings are loaded)
        self.init_detection_thread()
        
        # Start key listener
        self.key_listener = KeyListener(self.toggle_assist)
        self.key_listener.start()
        
        # Show a startup message
        self.statusBar().showMessage("Application started", 3000)
    
    def init_ui(self):
        self.setWindowTitle("OSU AI Aim Assistant")
        self.setMinimumSize(600, 700)  # Larger to accommodate tabs
        
        # Create status bar
        self.statusBar().showMessage("Ready")
        
        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create title and description
        title_label = QLabel("OSU AI Aim Assistant")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Status indicator and FPS display
        status_fps_layout = QHBoxLayout()
        
        # FPS display
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet("color: green; font-weight: bold;")
        fps_layout.addWidget(self.fps_label)
        
        # Status indicator
        self.status_label = QLabel("Detection: Stopped")
        self.status_label.setStyleSheet("color: gray;")
        
        status_fps_layout.addWidget(self.status_label)
        status_fps_layout.addStretch(1)
        status_fps_layout.addLayout(fps_layout)
        
        # Preview button
        preview_btn = QPushButton("Show Preview")
        preview_btn.clicked.connect(self.toggle_preview)
        self.preview_btn = preview_btn
        
        # Create tabbed interface
        self.tab_widget = TabWidget(self)
        self.tab_widget.settings_changed.connect(self.settings_updated)
        
        # Connect preview button from tab widget
        self.tab_widget.toggle_preview = self.toggle_preview
        
        # Profile controls
        profile_group = QGroupBox("Configuration Profiles")
        profile_layout = QVBoxLayout(profile_group)
        
        # Profile selection
        profile_combo_layout = QHBoxLayout()
        profile_combo_layout.addWidget(QLabel("Profile:"))
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
        
        # Add everything to main layout
        main_layout.addWidget(title_label)
        main_layout.addLayout(status_fps_layout)
        main_layout.addWidget(preview_btn)
        main_layout.addWidget(self.tab_widget, 1)  # Give the tab widget more space
        main_layout.addWidget(profile_group)
    
    def settings_updated(self):
        """Handle settings updates from the tab widget"""
        self.save_settings()
        
        # Update detection thread if it's running
        if self.detection_thread is not None:
            # Update confidence threshold
            self.detection_thread.update_confidence(SETTINGS["confidence_threshold"])
        
        # Update status label
        self.update_detection_status()
    
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
        
        # Also update status in preview window
        if hasattr(self, 'preview_window'):
            self.preview_window.update_detection_status()
        
        # Update toggle button style in tab widget
        if hasattr(self, 'tab_widget'):
            self.tab_widget.update_toggle_button_style()
    
    def init_detection_thread(self):
        self.detection_thread = DetectionThread(self)
        
        # Connect signals to preview window
        self.detection_thread.update_frame.connect(self.preview_window.update_preview)
        self.detection_thread.update_fps.connect(self.preview_window.update_fps)
        
        # Connect FPS signal to main window's FPS label
        self.detection_thread.update_fps.connect(self.update_fps)
        
        # Try to load the model if a path is already specified in settings
        if SETTINGS["model_path"]:
            if os.path.exists(SETTINGS["model_path"]):
                success = self.detection_thread.load_model(SETTINGS["model_path"])
                if success:
                    self.update_class_dropdown()
                    self.show_status_message(f"Model loaded successfully: {os.path.basename(SETTINGS['model_path'])}")
                else:
                    self.show_status_message(f"Failed to load model from settings: {os.path.basename(SETTINGS['model_path'])}", is_error=True)
            else:
                self.show_status_message(f"Model file not found: {os.path.basename(SETTINGS['model_path'])}", is_error=True)
                # Clear the invalid path
                SETTINGS["model_path"] = ""
        
        # Start the detection thread
        self.detection_thread.start()
        
        # Update the status indicator
        self.update_detection_status()
    
    def update_class_dropdown(self):
        # Update class dropdown in the tab widget
        if hasattr(self.tab_widget, 'target_class_combo'):
            # Clear current items
            self.tab_widget.target_class_combo.clear()
            
            # Add "All Classes" option
            self.tab_widget.target_class_combo.addItem("All Classes", -1)
            
            # Add classes from model
            if hasattr(self.detection_thread, 'model') and self.detection_thread.model is not None:
                # Check if model has names attribute
                if hasattr(self.detection_thread.model, 'names'):
                    class_mapping = self.detection_thread.model.names
                    
                    # Update global class mapping
                    global CLASS_MAPPING
                    CLASS_MAPPING = class_mapping
                    
                    # Populate dropdown
                    if isinstance(class_mapping, dict):
                        for idx, name in class_mapping.items():
                            self.tab_widget.target_class_combo.addItem(f"{name} ({idx})", idx)
                    elif isinstance(class_mapping, list):
                        for idx, name in enumerate(class_mapping):
                            self.tab_widget.target_class_combo.addItem(f"{name} ({idx})", idx)
                    
                    # Try to select active_note
                    for i in range(self.tab_widget.target_class_combo.count()):
                        if "active" in self.tab_widget.target_class_combo.itemText(i).lower():
                            self.tab_widget.target_class_combo.setCurrentIndex(i)
                            # Update target class setting
                            SETTINGS["target_class"] = self.tab_widget.target_class_combo.currentData()
                            break
            
            # Also update the slider class dropdown
            if hasattr(self.tab_widget, 'slider_class_combo'):
                self.tab_widget.slider_class_combo.clear()
                
                # Populate with classes
                if isinstance(CLASS_MAPPING, dict):
                    for idx, name in CLASS_MAPPING.items():
                        self.tab_widget.slider_class_combo.addItem(f"{name} ({idx})", idx)
                
                # Set current selection to slider_ball (class 2)
                for i in range(self.tab_widget.slider_class_combo.count()):
                    if self.tab_widget.slider_class_combo.itemData(i) == 2:  # Force slider_ball class to 2
                        self.tab_widget.slider_class_combo.setCurrentIndex(i)
                        SETTINGS["slider_ball_class"] = 2
                        break
    
    def update_fps(self, fps):
        """Update the FPS display in the main window"""
        self.fps_label.setText(f"{fps:.1f}")
        
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
        
        # Update tab widget toggle button
        if hasattr(self.tab_widget, 'toggle_button'):
            self.tab_widget.toggle_button.setChecked(SETTINGS["enabled"])
            self.tab_widget.update_toggle_button_style()
        
        # Update status indicator
        self.update_detection_status()
        
        # Show a status message
        if SETTINGS["enabled"]:
            self.show_status_message("Aim assist enabled")
        else:
            self.show_status_message("Aim assist disabled")
            
    def toggle_preview(self):
        """Toggle the preview window visibility and process"""
        if self.preview_window.isVisible():
            # Hide the preview window
            self.preview_window.hide()
            self.preview_btn.setText("Show Preview")
            SETTINGS["preview_enabled"] = False
            
            # Ask user if they want to completely disable preview processing
            reply = QMessageBox.question(
                self, "Disable Preview Processing",
                "Do you want to completely disable preview processing for better performance?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                SETTINGS["preview_process_disabled"] = True
                self.show_status_message("Preview processing disabled for better performance")
            else:
                SETTINGS["preview_process_disabled"] = False
        else:
            # Show the preview window and enable preview processing
            SETTINGS["preview_enabled"] = True
            SETTINGS["preview_process_disabled"] = False
            self.preview_window.show()
            self.preview_btn.setText("Hide Preview")
        
        # Save setting
        self.save_settings()
    
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
                
                # Make sure enabled is always false on startup
                SETTINGS["enabled"] = False
                
                # Configure preview button text
                if SETTINGS["preview_enabled"]:
                    self.preview_btn.setText("Hide Preview")
                else:
                    self.preview_btn.setText("Show Preview")
                
                # After loading settings, let's make sure slider_ball class is 2
                SETTINGS["slider_ball_class"] = 2
                
                print("Settings loaded successfully")
        except Exception as e:
            print(f"Error loading settings: {e}")
            import traceback
            traceback.print_exc()
    
    def save_settings(self):
        try:
            # Make sure profiles directory exists
            os.makedirs(PROFILES_DIR, exist_ok=True)
            
            # Save current settings to the main settings file
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(SETTINGS, f, indent=4)
            
            # Save current profile settings
            self.save_current_profile()
        except Exception as e:
            print(f"Error saving settings: {e}")
            import traceback
            traceback.print_exc()
    
    # Profile management methods
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
                was_enabled = SETTINGS["enabled"]
                for key, value in profile_settings.items():
                    if key in SETTINGS:
                            SETTINGS[key] = value
                
                # Update profile name
                SETTINGS["profile_name"] = profile_name
                
                # Restore enabled state
                SETTINGS["enabled"] = was_enabled
                
                # Make sure slider_ball class is 2
                SETTINGS["slider_ball_class"] = 2
                
                # Notify tab widget of changes
                self.settings_updated()
                
                self.show_status_message(f"Loaded profile: {profile_name}")
        except Exception as e:
            print(f"Error loading profile '{profile_name}': {e}")
            self.show_status_message(f"Error loading profile", True)
    
    def save_profile(self):
        """Save current settings to selected profile"""
        profile_name = self.profile_combo.currentText()
        if not profile_name:
            return
            
        try:
            self.save_current_profile()
            self.show_status_message(f"Saved profile: {profile_name}")
        except Exception as e:
            print(f"Error saving profile '{profile_name}': {e}")
            self.show_status_message(f"Error saving profile", True)
    
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
            
            self.show_status_message(f"Created profile: {profile_name}")
    
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
                self.show_status_message(f"Deleted profile: {profile_name}")
            except Exception as e:
                print(f"Error deleting profile '{profile_name}': {e}")
                self.show_status_message(f"Error deleting profile", True)
    
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
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Save settings
        self.save_settings()

        # Stop detection thread
        if self.detection_thread is not None:
            self.detection_thread.stop()
        
        # Close preview window if open
        if hasattr(self, 'preview_window') and self.preview_window.isVisible():
            self.preview_window.close()
        
        # Accept the close event
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())