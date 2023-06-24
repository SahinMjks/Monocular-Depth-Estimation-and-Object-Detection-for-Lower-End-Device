import cv2

# Set up video capture
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Failed to open camera")
    exit()

# Get the focal length of the camera
focal_length = cap.get(cv2.CAP_PROP_FOCAL_LENGTH)

# Print the focal length
print("Focal Length:", focal_length)

# Release resources
cap.release()
