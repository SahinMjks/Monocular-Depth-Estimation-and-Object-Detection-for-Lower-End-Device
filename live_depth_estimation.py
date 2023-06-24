import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time

# Load the saved model
model_path = "~/transformer_model"
model = torch.load(model_path)
model.eval()

# Set up video capture
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    # (perform any necessary transformations on the frame before passing it to the model)
    # ...

    # Convert the frame to a tensor
    frame_tensor = transforms.ToTensor()(frame)

    # Run the inference
    with torch.no_grad():
        prediction = model(frame_tensor)

    # Perform any necessary post-processing on the prediction
    # ...

    # Display the depth estimation
    cv2.imshow('Depth Estimation', prediction)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    # Display FPS on the OpenCV window
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
