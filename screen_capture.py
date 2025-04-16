import numpy as np
import cv2
import time
import logging
import dxcam

class ScreenCapture:
    def __init__(self, window_title="osu!", log_level=logging.INFO):
        """
        Initialize the screen capture module using DXcam for high-performance capture.
        
        Args:
            window_title (str): Title of the game window to capture
            log_level: Logging level
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.window_title = window_title
        self.region = None
        self.camera = None
        
        # Initialize DXcam
        self.initialize_dxcam()
        
    def initialize_dxcam(self):
        """Initialize the DXcam camera for fast screen capture."""
        try:
            # Create a DXcam camera instance
            self.camera = dxcam.create()
            self.logger.info("DXcam initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing DXcam: {e}")
            self.camera = None
    
    def find_game_window(self):
        """
        Find the game window and set the capture region.
        Returns True if successful, False otherwise.
        """
        import win32gui
        
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd) and self.window_title in win32gui.GetWindowText(hwnd):
                windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if not windows:
            self.logger.warning(f"Game window '{self.window_title}' not found")
            return False
        
        # Use the first window that matches
        hwnd = windows[0]
        try:
            # Get window position and dimensions
            rect = win32gui.GetWindowRect(hwnd)
            x, y, right, bottom = rect
            width = right - x
            height = bottom - y
            
            # Set the region for DXcam to capture
            self.region = (x, y, right, bottom)
            self.logger.info(f"Game window found: {self.window_title} at {self.region}")
            return True
        except Exception as e:
            self.logger.error(f"Error getting window dimensions: {e}")
            return False
    
    def capture_window(self):
        """
        Capture the game window using DXcam.
        Returns the captured image or None if capture failed.
        """
        if self.camera is None:
            self.logger.error("DXcam not initialized")
            return None
        
        if self.region is None:
            if not self.find_game_window():
                return None
        
        try:
            # Capture the screen region
            image = self.camera.grab(region=self.region)
            if image is None:
                self.logger.warning("Failed to capture screen")
                return None
            
            return image
        except Exception as e:
            self.logger.error(f"Error capturing screen: {e}")
            return None
    
    def process_image(self, image):
        """
        Process the captured image to detect game elements.
        
        Args:
            image: The captured image
            
        Returns:
            tuple: (processed_image, mask) or (None, None) if processing failed
        """
        if image is None:
            return None, None
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Example: Create a mask for a specific color range
            # This is a placeholder - adjust color ranges based on what you want to detect
            lower_bound = np.array([0, 50, 50])
            upper_bound = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Apply the mask to the original image
            processed_image = cv2.bitwise_and(image, image, mask=mask)
            
            return processed_image, mask
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None, None
    
    def get_game_state(self):
        """
        Get the current game state by capturing and processing the game window.
        
        Returns:
            dict: A dictionary containing the processed image, mask, and timestamp
                  or None if capture or processing failed
        """
        # Capture the game window
        image = self.capture_window()
        if image is None:
            return None
        
        # Process the image
        processed_image, mask = self.process_image(image)
        if processed_image is None:
            return None
        
        # Create a game state dictionary
        game_state = {
            'raw_image': image,
            'processed_image': processed_image,
            'mask': mask,
            'timestamp': time.time()
        }
        
        return game_state
    
    def close(self):
        """Close the DXcam camera and clean up resources."""
        if self.camera:
            self.camera.stop()
            self.logger.info("DXcam camera closed") 