Real-time Object Detection using YOLOv5 
This project demonstrates real-time object detection using a pre-trained YOLOv5 model and a live webcam feed. It serves as a foundational project for computer vision applications, showcasing the ability to process live video data and identify multiple objects within each frame.

Features ✨
Real-time Inference: Processes a live video stream from a webcam for instant object detection.

Pre-trained Model: Leverages a pre-trained YOLOv5s model for fast and accurate detection of common objects (e.g., person, car, dog, etc.).

Visual Output: Draws bounding boxes and labels around detected objects directly onto the video feed.

User-friendly Interface: Includes a simple exit condition (pressing the 'q' key) to stop the program gracefully.

Installation 🛠️
To set up this project, you'll need Python 3.8 or higher. Follow these steps to create a virtual environment and install the necessary libraries.

Clone the repository:

Bash

git clone https://github.com/AdamKBetts/realtime-object-detection.git
cd realtime-object-detection
Create and activate a virtual environment:

Bash

python -m venv ai_project
ai_project\Scripts\activate  # On Windows
Install the required libraries:

Bash

pip install opencv-python torch torchvision ultralytics seaborn
Usage ▶️
With your virtual environment active, run the detector.py script from your project directory.

Bash

python detector.py
A window will pop up showing your webcam feed with real-time object detection. To stop the program, press the 'q' key.

Next Steps & Enhancements 💡
This project is a solid foundation, but it can be expanded and improved upon. Some ideas for further development include:

Performance Optimization: Implement the project on a GPU for faster inference speeds.

Custom Object Detection: Fine-tune the model on a custom dataset to detect specific objects not included in the original training data.

Object Counting and Tracking: Add logic to count specific objects or track their movement over time.
