import cv2
import numpy as np
import urllib.request

# Download YOLOv3 weights and configuration files
yolo_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
yolo_cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
yolo_classes_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
urllib.request.urlretrieve(yolo_weights_url, 'yolov3.weights')
urllib.request.urlretrieve(yolo_cfg_url, 'yolov3.cfg')
urllib.request.urlretrieve(yolo_classes_url, 'coco.names')

# Load YOLOv3 model
net = cv2.dnn.readNet('yolov3.cfg', 'yolov3.weights')
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load MiDaS model
mides_url = 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pt'
urllib.request.urlretrieve(mides_url, 'mides.onnx')
mides_net = cv2.dnn.readNetFromONNX('mides.onnx')

# Load class labels for YOLOv3
classes = []
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for input to YOLOv3
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = frame.shape

    # Perform object detection using YOLOv3
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists for bounding boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Scale bounding box coordinates to the original frame
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Calculate bounding box coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to eliminate redundant overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Initialize list for object distances
    distances = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Crop ROI for depth estimation using MiDaS
            roi = frame[y:y + h, x:x + w]
            blob = cv2.dnn.blobFromImage(roi, 1/255.0, (384, 384), 0, swapRB=True, crop=False)
            mides_net.setInput(blob)
            mides_output = mides_net.forward()
            avg_depth = np.average(mides_output)
            distance = 1 / avg_depth

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {distance:.2f} meters", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Append distance to the list
            distances.append(distance)

    # Display the frame
    cv2.imshow('Object Detection and Depth Estimation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
