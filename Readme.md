# Hand Gesture Control with OpenCV, MediaPipe, and PyAutoGUI

This code demonstrates how to implement hand gesture control using OpenCV, MediaPipe, and PyAutoGUI. It detects hand landmarks, translates them into gestures and performs corresponding actions like clicking and dragging on the computer.

## Requirements

- OpenCV: `pip install opencv-python`
- MediaPipe: `pip install mediapipe`
- PyAutoGUI: `pip install pyautogui`

## How It Works

1. The code initializes a webcam feed using OpenCV.
2. It processes each frame to detect hand landmarks using MediaPipe's hand solutions.
3. The detected landmarks are used to infer hand gestures. The gestures include:
    - Pinch for dragging capabilities.
    - Bending index finger for a left click.
    - Bending middle finger for a right click.
4. PyAutoGUI is used to simulate the mouse actions based on the detected gestures.

## Code Documentation

### Importing Necessary Modules
```python
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time
```

### Initializing Necessary Variables and Objects

- `is_holding`: Represents the state of mouse drag.
- `threshold_angle`: Threshold angle for detecting a bent finger. You can adjust this value based on your needs.
- `ACTION_COOLDOWN`: Time interval between each mouse action to avoid spamming.
- `last_action_time`: Keeps track of when the last action was performed.
- `capture`: Video capture object for accessing the webcam.
- `width`, `height`: Dimensions of the webcam feed.
- `screen_width`, `screen_height`: Dimensions of the computer screen.

### Helper Functions

- `calculate_distance(landmark1, landmark2)`: Calculates the Euclidean distance between two landmarks in a 2D space.
  
- `detect_gestures(landmarks)`: Detects hand gestures based on landmarks and simulates corresponding mouse actions using PyAutoGUI.

### Main Loop

The main loop captures frames from the webcam feed, processes them to detect hand landmarks, and invokes the `detect_gestures` function to infer and act upon the detected gestures. The output frame with drawn landmarks is displayed using OpenCV.

Press `q` to exit the loop and close the application.

## Running the Code

To run the code, save the above code in a Python file (e.g., `hand_gesture_control.py`) and execute:

```bash
python hand_gesture_control.py
```

Ensure you have the necessary packages installed and a webcam connected to your system. Adjust the camera index in `cv2.VideoCapture(0)` if using an external camera.

## Notes

1. Ensure proper lighting conditions for accurate hand detection.
2. Adjust the `threshold_angle` and other parameters for better gesture detection based on your environment and needs.

## License

This code is provided under the MIT License. Make sure you mention the original source and author if you use or modify this code.
This code is provided under the MIT License. Make sure you mention the original source and author if you use or modify this code.