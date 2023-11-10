import cv2
import numpy as np

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Function to extract features from an image
def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to match descriptors between two images
def match_descriptors(descriptors1, descriptors2):
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    # Sort them in the order of their distance (the lower the better)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Indexing: Extract features from all frames of the video and store them
def index_video_frames(video_frame_paths):
    indexed_frames = []
    for frame_path in video_frame_paths:
        keypoints, descriptors = extract_features(frame_path)
        indexed_frames.append((frame_path, keypoints, descriptors))
    return indexed_frames

# Search: Match features of test images against indexed frames
def search_video(test_image_paths, indexed_frames):
    results = []
    for test_image_path in test_image_paths:
        _, test_descriptors = extract_features(test_image_path)
        best_match = None
        best_match_score = None
        # Compare against each indexed frame
        for frame_path, _, frame_descriptors in indexed_frames:
            matches = match_descriptors(test_descriptors, frame_descriptors)
            score = len(matches)  # Simple scoring by number of matches
            if best_match is None or score > best_match_score:
                best_match = frame_path
                best_match_score = score
        results.append((test_image_path, best_match, best_match_score))
    return results

# Example usage
video_frame_paths = ['path/to/video/frame1.jpg', 'path/to/video/frame2.jpg', ...]  # Paths to your video frames
test_image_paths = ['path/to/test/image1.png', 'path/to/test/image2.jpeg', ...]  # Paths to your test images

# Index the video frames
indexed_frames = index_video_frames(video_frame_paths)

# Search for each test image
search_results = search_video(test_image_paths, indexed_frames)

# Print results
for test_image, best_match, score in search_results:
    print(f"Test image: {test_image} - Best match: {best_match} with score: {score}")
