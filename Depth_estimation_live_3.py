import cv2
import tensorflow as tf
import numpy as np
import time

# Load the saved model
model_path = 'C:/Users/ee20m/Documents/Advance_model'
loaded_model = tf.saved_model.load(model_path)
input_tensor_info = loaded_model.signatures['serving_default'].inputs[0]

# Set up video capture
cap = cv2.VideoCapture(0)

# Define the desired input shape
input_shape = (256, 256, 3)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    frame = frame / 255.0

    # Expand dimensions to create a batch of size 1
    frame_batch = np.expand_dims(frame, axis=0)

    # Convert the frame batch to a tensor
    frame_tensor = tf.constant(frame_batch, dtype=tf.float32)

    # Run the inference
    output = loaded_model(frame_tensor)

    # Extract the depth map from the output tensor
    depth_map = output.numpy()[0, :, :, 0]

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Rescale the depth map to 0-255
    depth_map = (depth_map * 255).astype(np.uint8)

    # Convert the depth map to grayscale
    depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # Display the depth map
    cv2.imshow('Depth Estimation', depth_map_gray)

    frame_count += 1

    # Calculate FPS every 5 frames
    if frame_count % 5 == 0:
        end_time = time.time()
        fps = frame_count / (end_time - start_time)
        print(f'FPS: {fps:.2f}')

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
