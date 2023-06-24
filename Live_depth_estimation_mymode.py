import cv2
import tensorflow as tf
import numpy as np

# Load the saved model
model_path = 'C:/Users/ee20m/Downloads/transformer_model'
loaded_model = tf.saved_model.load(model_path)
infer = loaded_model.signatures['serving_default']

# Set up video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    frame = cv2.resize(frame, (320, 160))
    frame = frame / 255.0
    frame = np.expand_dims(frame, axis=0)

    # Run the inference
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
    output = infer(input_tensor)

    # Extract the depth map from the output tensor
    depth_map = output['output_1'].numpy()[0, :, :, 0]

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Rescale the depth map to 0-255
    depth_map = (depth_map * 255).astype(np.uint8)

    # Convert the depth map to grayscale
    depth_map_gray = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)

    # Display the depth map
    cv2.imshow('Depth Estimation', depth_map_gray)

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
