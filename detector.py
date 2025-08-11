import cv2
import torch

# Load a pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Start the webcam feed using default webcam
cap = cv2.VideoCapture(0)

# Process frames in a loop
while True:
    # cap.read - reads a frame from the video stream, 'ret' is boolean that is true if frame was read successfully
    ret, frame = cap.read()

    # If the frame was not read successfully break
    if not ret:
        break

    # The model processes the image and returns an object containing the detection results.
    results = model(frame)

    # The 'results.render()' method draws bounding boxes, labels, and confidence scores
    # directly onto the frame, making visualization very simple.
    rendered_frame = results.render()[0]

    # 'cv2.imshow()' displays the processed frame in a window
    cv2.imshow('YOLOv5 Object Detection', rendered_frame)

    # 'cv2.waitKey(1)' waits for 1 millisecond for a key press.
    # '0xFF == ord('q')' checks if the pressed key is 'q'.
    # If the user presses 'q', the loop breaks.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 'cap.release()' releases the webcam, freeing it up for other applications.
cap.release()

# 'cv2.destroyAllWindows()' closes all the OpenCV windows.
cv2.destroyAllWindows()