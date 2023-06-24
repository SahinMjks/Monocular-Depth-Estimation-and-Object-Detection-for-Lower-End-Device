import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time

# Camera parameters
focal_length = 640
sensor_width = 3.6
sensor_height = 2.7

# Calculate pixel size (in cm)
pixel_size = sensor_width / focal_length

# Calculate sensor size (in pixels)
sensor_size_x = round(sensor_width / pixel_size)
sensor_size_y = round(sensor_height / pixel_size)

# Load the MiDaS Small model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
midas.eval()

# Set the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((sensor_size_y, sensor_size_x)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set up video capture
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0
fps = 0

# Load YOLOv3 weights and configuration
weights_file = 'yolov3.weights'
config_file = 'yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Manually specify label names
label_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the transformation
    input_tensor = transform(frame_rgb).unsqueeze(0)
    input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run the inference for depth estimation
    with torch.no_grad():
        prediction = midas(input_tensor)

    # Convert the prediction tensor to a numpy array
    depth_map = prediction.squeeze().cpu().numpy()

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Resize the frame for object detection
    resized_frame = cv2.resize(frame, (sensor_size_x, sensor_size_y))

    # Preprocess the input frame for YOLOv3
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (sensor_size_x, sensor_size_y), swapRB=True, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Forward pass through the network of YOLO object detection model
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process the network outputs
    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:  # Adjust confidence threshold as desired
                # Scale the bounding box coordinates to the original frame size
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate the top-left corner coordinates
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-maximum suppression to remove overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.3, nms_threshold=0.4)

    # Draw bounding boxes and labels, and calculate distance for each object
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]

            if class_id < len(label_names):
                label = label_names[class_id]
            else:
                label = 'unknown'

            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw label and confidence
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop the object region from the depth map
            depth_crop = depth_map[y:y+h, x:x+w]

            # Calculate the average depth within the object region
            object_depth = 1 / np.mean(depth_crop)

            # Calculate the distance to the object using the formula: distance = (focal_length * object_width) / width_in_pixels
            object_width = w * pixel_size
            distance = (focal_length * object_width) / w

            # Display the depth value
            depth_text = f'Depth: {distance:.2f}'
            cv2.putText(frame, depth_text, (x, y + h +1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = frame_count / elapsed_time
        start_time = end_time
        frame_count = 0

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Object Detection and Distance Estimation', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
