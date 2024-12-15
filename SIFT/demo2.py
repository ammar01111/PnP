import cv2
import numpy as np

# Global variable to store the point where the mouse clicks
click_point = None

# Function to handle mouse events and get the click position
def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"Clicked at: ({x}, {y})")

# Function to detect keypoints using SIFT and find the nearest keypoint
def find_nearest_keypoint(image, query_point):
    sift = cv2.SIFT_create()
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    query_point = np.array(query_point, dtype=np.float32)
    distances = []

    # Calculate Euclidean distance from the query point to each keypoint
    for keypoint in keypoints:
        keypoint_coords = np.array([keypoint.pt], dtype=np.float32)
        distance = np.linalg.norm(keypoint_coords - query_point)
        distances.append((distance, keypoint))
    
    # Sort the distances and get the keypoint with the smallest distance
    distances.sort(key=lambda x: x[0])
    nearest_keypoint = distances[0][1]
    
    return nearest_keypoint, nearest_keypoint.pt

# Load the image
image = cv2.imread(r'C:\Users\PC\Desktop\Pnp\PnP\surf\surfImages\image91.jpg')

# Set the window name
cv2.imshow("Click on the image", image)

# Set the mouse callback function to capture the click position
cv2.setMouseCallback("Click on the image", mouse_callback)

while True:
    # Wait for the user to click
    if click_point is not None:
        # Find the nearest keypoint to the clicked point
        nearest_keypoint, nearest_point = find_nearest_keypoint(image, click_point)

        # Draw the nearest keypoint on the image
        img_with_keypoint = image.copy()
        cv2.drawKeypoints(img_with_keypoint, [nearest_keypoint], img_with_keypoint, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # Show the image with the nearest keypoint
        cv2.imshow("Nearest Keypoint", img_with_keypoint)
        print(f"Nearest Keypoint Coordinates: {nearest_point}")

        # Reset click_point to None to wait for another click
        click_point = None

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
