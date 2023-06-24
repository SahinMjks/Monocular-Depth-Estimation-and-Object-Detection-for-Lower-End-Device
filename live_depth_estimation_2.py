import cv2
import tensorflow as tf
import numpy as np
import time

# Load the saved model
model_path = 'C:/Users/ee20m/Documents/Model_1'
loaded_model = tf.keras.models.load_model(model_path)

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

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    frame_preprocessed = cv2.resize(frame_rgb, (256, 256))
    frame_preprocessed = frame_preprocessed / 255.0
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

    # Run the inference
    depth_map = loaded_model.predict(frame_preprocessed)

    # Extract the depth map from the output tensor
    depth_map = depth_map[0, :, :, 0]

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Rescale the depth map to 0-255
    depth_map = (depth_map * 255).astype(np.uint8)

    # Convert the depth map to grayscale
    depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # Calculate FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        start_time = end_time
        frame_count = 0

    # Display the depth map
    cv2.imshow('Depth Estimation', depth_map_gray)

    # Display the actual image
    cv2.imshow('Camera', frame)

    # Display FPS on the OpenCV window
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
