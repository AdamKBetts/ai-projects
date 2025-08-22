import cv2
import numpy as np
import tensorflow as tf

# Load a pre-trained model
model = tf.saved_model.load("ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")

# Load the video file
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open file.")
    exit()

# List of class labels
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Simple IoU (Intersection over Union) function
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Tracking variables
trackers = {} # Dict for ID store and last know position of tracked object
next_id = 0
iou_threshold = 0.5 # To determine if detection belongs to an existing tracker

while True:
    ret, frame = cap.read()
    if not ret:
        break

    input_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    # Perform detection
    detections = model(tf.constant(input_tensor, dtype=tf.uint8))
    
    current_frame_detections = []

    # Process detections
    for i in range(int(detections['num_detections'][0])):
        score = detections['detection_scores'][0][i].numpy()
        bbox = detections['detection_boxes'][0][i].numpy()
        class_id = int(detections['detection_classes'][0][i].numpy())

        if score > 0.5:
            ymin, xmin, ymax, xmax = bbox
            (h, w, _) = frame.shape
            (xmin, xmax, ymin, ymax) = (int(xmin * w), int(xmax * w), int(ymin * h), int(ymax * h))
            
            label = f"{classes[class_id]}: {score:.2f}"
            current_frame_detections.append({'box': (xmin, ymin, xmax, ymax), 'label': label})

    # Tracking logic
    new_detections_match = [False] * len(current_frame_detections)
    
    # Handle existing trackers
    if trackers:
        for obj_id, last_box in list(trackers.items()):
            best_match_iou = 0
            best_match_idx = -1
            
            for idx, current_det in enumerate(current_frame_detections):
                if not new_detections_match[idx]:
                    current_iou = iou(last_box, current_det['box'])
                    if current_iou > best_match_iou and current_iou > iou_threshold:
                        best_match_iou = current_iou
                        best_match_idx = idx

            if best_match_idx != -1:
                # Update existing tracker
                trackers[obj_id] = current_frame_detections[best_match_idx]['box']
                current_frame_detections[best_match_idx]['id'] = obj_id
                new_detections_match[best_match_idx] = True
            else:
                # Remove tracker if no match found
                del trackers[obj_id]

    # Handle new detections (ones that did not match an existing tracker)
    for idx, det in enumerate(current_frame_detections):
        if not new_detections_match[idx]:
            det['id'] = next_id
            trackers[next_id] = det['box']
            next_id += 1
            
    # Draw results with IDs
    for det in current_frame_detections:
        (xmin, ymin, xmax, ymax) = det['box']
        label_with_id = f"{det['label']} | ID:{det['id']}"
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, label_with_id, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Object Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()