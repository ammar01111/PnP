import cv2

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)
i = 1
# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 's' to save a frame, 'q' to quit.")

while True:
    # Capture each frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Save the current frame as an image
        filename = f"captured_frame{i}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
        i+=1
    elif key == ord('q'):
        # Quit the program
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
