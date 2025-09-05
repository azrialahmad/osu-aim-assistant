# OSU! AI Aim Assistant

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

An AI-powered aim assistant designed to analyze OSU! gameplay in real-time. This tool uses the YOLOv11 object detection model to identify targets and a high-performance screen capture library to minimize latency, serving as a practical exploration of computer vision in a high-speed gaming environment.

---

### Showcase

[showcase](showcase.gif)


---

### 🎯 Project Goal

The primary goal of this project was to address the steep learning curve in the high-precision rhythm game OSU! by applying computer vision. I wanted to build a tool that could parse the visual information of the game in real-time, facing the core challenge of achieving extremely low latency to keep up with the fast-paced gameplay.

---

### ✨ Key Features

* **Real-Time Target Detection:** Utilizes a YOLOv11 model specifically trained to identify OSU! hit circles, sliders, and spinners.
* **High-Performance Screen Capture:** Built with `dxcam` for GPU-accelerated screen capture, ensuring minimal performance impact and low latency.
* **Intuitive GUI:** A simple graphical user interface allows for easy configuration of all settings.
* **Customizable Assistance:**
    * **Multiple Aim Modes:** Choose between three distinct assistance algorithms:
        * `Interpolated`: Linearly moves the cursor towards the target.
        * `Smooth`: Uses a weighted average for a more natural, less robotic cursor path.
        * `Snap`: Instantly moves the cursor to the target once detected (for testing).
    * **AI Confidence Threshold:** Adjust the AI's detection confidence to filter out false positives.
    * **Assist Strength:** Control how strongly the assistant influences cursor movement.

---

### 🛠️ How It Works

The application operates in a tight, high-performance loop to ensure real-time responsiveness:

1.  **Screen Capture:** A dedicated thread uses `dxcam` to continuously capture the game window. This method is significantly faster than traditional screen capture libraries as it leverages the GPU.
2.  **Object Detection:** The captured frame is fed into the YOLOv11 model, which processes the image and returns the coordinates of all detected targets (hit circles).
3.  **Cursor Guidance:** Based on the selected mode, the application calculates the vector from the current cursor position to the nearest detected target and applies a calibrated force to guide the user's aim.

---

### ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/osu-aim-assistant.git](https://github.com/your-username/osu-aim-assistant.git)
    cd osu-aim-assistant
    ```

2.  **Install Dependencies**
    *It is highly recommended to use a virtual environment.*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    python main.py
    ```

4.  **Configure Settings**
    * Launch OSU! in windowed or borderless windowed mode.
    * Use the GUI to adjust the confidence, assist strength, and aim mode to your preference.

---

### 🖥️ System Requirements

* **OS:** Windows 10 / 11
* **Python:** Version 3.8+
* **GPU:** An NVIDIA GPU compatible with CUDA is highly recommended for optimal performance.

---

### ## ⚠️ Known Limitations & Disclaimer

* **Tablet Absolute Positioning:** This tool does not currently function well with tablets that use absolute positioning for cursor input.
* **Educational Purpose Only:** This tool was developed as a personal project to explore computer vision and AI in a real-time application. Using aim-assist software may violate the terms of service for OSU! and is not recommended for competitive play.
