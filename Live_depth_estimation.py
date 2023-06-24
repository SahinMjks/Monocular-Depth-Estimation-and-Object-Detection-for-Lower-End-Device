import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load the MiDaS Small model
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
midas.eval()

# Set the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Set up video capture
cap = cv2.VideoCapture(0)

# Initialize variables for FPS calculation
start_time = time.time()
frame_count = 0
fps = 0

# Set up 3D plot for point cloud visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply the transformation
    input_tensor = transform(frame_rgb).unsqueeze(0)
    input_tensor = input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Run the inference
    with torch.no_grad():
        prediction = midas(input_tensor)

    # Convert the prediction tensor to a numpy array
    depth_map = prediction.squeeze().cpu().numpy()

    # Normalize the depth map for visualization
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # Generate point cloud from depth map
    height, width = depth_map.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = x.flatten()
    y = y.flatten()
    z = depth_map.flatten()
    mask = z > 0
    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Update the 3D plot with the new point cloud
    ax.clear()
    ax.scatter(x, y, z, c=z, cmap='jet')

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

    # Display the 3D point cloud
    plt.pause(0.001)
    plt.draw()

    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
