import cv2
import numpy as np
import os

# Function to detect the four corners of a box (for simplicity, you can manually specify these coordinates)
def get_box_corners(img):
    # For simplicity, manually define the corners of the box in the first image (replace these with actual corner points)
    corners = np.array([(201, 204), (501, 173), (214, 410), (485, 350)], dtype=np.float32)
    return corners

# SIFT detector
sift = cv2.SIFT_create()

# Feature matcher (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Folder path where images are stored
image_folder = r'C:\Users\PC\Desktop\Pnp\PnP\surf\surfImages'

# Generate a list of image filenames
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])

# Check if there are at least two images
if len(image_files) < 2:
    print("Not enough images for matching!")
    exit()

# Initialize updated_box_corner1 to None
updated_box_corner1 = None


# Loop through image pairs
for i in range(len(image_files) - 1):
    # Load the current pair of images
    img1_path = os.path.join(image_folder, image_files[i])
    img2_path = os.path.join(image_folder, image_files[i + 1])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Check if images are loaded
    if img1 is None or img2 is None:
        print(f"Error loading images: {img1_path} or {img2_path}")
        continue

    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1_gray, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2_gray, None)

    # Match descriptors
    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key = lambda x: x.distance)

    # Get the four corners of the box in the first image
    box_corners_img1 = get_box_corners(img1)

    # If this is not the first image pair, update the box corners with the previous updated position
    if updated_box_corner1 is not None:
        box_corners_img1 = updated_box_corner1

    # Convert keypoints to their coordinates
    keypoints_1_coords = np.array([kp.pt for kp in keypoints_1], dtype=np.float32)
    keypoints_2_coords = np.array([kp.pt for kp in keypoints_2], dtype=np.float32)

    # Find the corresponding corners in the second image
    corners_img2 = []
    for corner in box_corners_img1:
        distances = np.linalg.norm(keypoints_2_coords - corner, axis=1)
        closest_match_index = np.argmin(distances)
        corners_img2.append(keypoints_2_coords[closest_match_index])

    # Shift x-coordinates of corners in the second image by half the width of the second image
    img2_width = img2.shape[1]  # Get width of the second image
    shifted_corners_img2 = []
    for corner in corners_img2:
        shifted_corner = corner.copy()
        shifted_corner[0] += 0  # Add half the width of the image to the x-coordinate
        shifted_corners_img2.append(shifted_corner)

    # Draw the matches on the images with the top 100 matches
    img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:500], None, flags=2)
    
    # Draw the box corners on the first image (img1)
    for i, corner in enumerate(box_corners_img1):
        cv2.circle(img_matches, tuple(corner.astype(int)), 10, (0, 255, 0), 2)  # Green for the box in image 1

    # Create separate copies for image1 and image2 to draw the corners
    img1_corners = img1.copy()
    img2_corners = img2.copy()

    # Draw corners on img1 (image 1)
    for corner in box_corners_img1:
        cv2.circle(img1_corners, tuple(corner.astype(int)), 10, (0, 255, 0), 2)  # Green for image 1

    # Draw corners on img2 (image 2)
    for corner in shifted_corners_img2:
        cv2.circle(img2_corners, tuple(corner.astype(int)), 10, (0, 0, 255), 2)  # Red for image 2

    # Display the results in separate windows
    cv2.imshow(f'Matching Corners: {image_files[i]} and {image_files[i + 1]}', img_matches)  # Matches in one window
    cv2.imshow(f'Corners in Image {image_files[i]}', img1_corners)  # Corners in the first image
    cv2.imshow(f'Corners in Image {image_files[i + 1]}', img2_corners)  # Corners in the second image

    # Update the box corners for the next image pair
    updated_box_corner1 = np.array(shifted_corners_img2, dtype=np.float32)

    # Wait for a key press to move to the next pair
    key = cv2.waitKey(0)

    # If 'q' is pressed, quit the loop
    if key == ord('q'):
        break

cv2.destroyAllWindows()
