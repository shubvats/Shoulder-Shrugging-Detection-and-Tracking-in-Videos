---

# Shoulder Shrugging Detection and Tracking in Videos

This repository contains Python code for detecting and tracking shoulder movements, specifically targeting the action of shoulder shrugging, in videos. Utilizing computer vision techniques and machine learning models, the code provides robust detection and visualization methods for analyzing upper body gestures.

## Features:

- **Face Detection**: Utilizes existing methods like Viola-Jones or Convolutional Neural Networks for detecting head positions in each frame of the video.
  
- **Shoulder Line Detection and Tracking**: Detects and tracks shoulder lines based on the detected head position, providing options for detecting each shoulder independently or together.
  
- **Shoulder Shrugging Detection**: Identifies shoulder shrugging events throughout the video, ranging from subtle to exaggerated, and highlights them in the corresponding frames.

## Instructions:

1. **Setup**: Ensure you have Python installed along with necessary libraries such as OpenCV (`cv2`) and NumPy (`numpy`).

2. **Usage**: Run the provided Python script (`shoulder_shrugging_detection.py`) with a video file as input.

3. **Customization**: Adjust parameters like detection thresholds and visualization options as needed for your specific application.

## Example Usage:

```python
python shoulder_shrugging_detection.py --video_path <path_to_video_file>
```

## Requirements:

- Python 3.x
- OpenCV (cv2)
- NumPy

## Credits:

This project was developed by Shubham Vats

