# Project 8: Object Tracking in Video

This project demonstrates how to build an object tracker that identifies objects in a video stream and follows their movement over time. It builds on a foundational object detection model and uses a simple **Intersection over Union (IoU)** algorithm to assign and maintain a unique ID for each detected object.

### Features

- **Video Processing:** Reads and processes video files frame by frame.
- **Object Detection:** Uses a pre-trained **TensorFlow** model to detect objects.
- **Basic Object Tracking:** Implements an IoU-based method to link detections across frames.
- **Unique IDs:** Assigns a persistent ID to each tracked object.

### Installation & Usage

- Requires the TensorFlow object detection model, which can be downloaded separately.
- Run the `object_tracker.py` script and provide a video file to see the tracker in action.
