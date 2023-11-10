import cv2
import numpy as np
import pandas as pd
import os

# Initialize SIFT detector
sift = cv2.SIFT_create()

def extract_features(image_path):
    # Processing both PNG and JPEG images identically
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_descriptors(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def index_video_frames(video_frame_paths):
    indexed_frames = []
    for frame_path in video_frame_paths:
        keypoints, descriptors = extract_features(frame_path)
        indexed_frames.append((frame_path, keypoints, descriptors))
    return indexed_frames

def search_video(test_image_paths, indexed_frames):
    # Adjusted to produce the output in the required format
    results = []
    for test_image_path in test_image_paths:
        _, test_descriptors = extract_features(test_image_path)
        best_match = None
        best_match_score = None
        best_match_frame = None
        # Compare against each indexed frame
        for frame_path, _, frame_descriptors in indexed_frames:
            matches = match_descriptors(test_descriptors, frame_descriptors)
            score = len(matches)  # Simple scoring by number of matches
            if best_match is None or score > best_match_score:
                best_match = frame_path
                best_match_score = score
                best_match_frame = os.path.basename(frame_path).split('.')[0]  # Assuming frame name format is videoName_frameNumber.jpg
        results.append((os.path.basename(test_image_path), best_match_frame, best_match_score))
    return results

def write_results_to_csv(results, output_file='predictions.csv'):
    df = pd.DataFrame(results, columns=['image', 'video_pred', 'minutage_pred'])
    df.to_csv(output_file, index=False)

# Example usage
directory_path = r'C:\Users\sawse\Downloads\INF8770_TP3_A2023\data\test\jpeg'  # Raw string for Windows path
video_frame_paths = glob.glob(os.path.join(directory_path, '*.jpeg'))
directory_path = r'C:\Users\sawse\Downloads\INF8770_TP3_A2023\data\test\png'  # Raw string for Windows path
test_image_paths = glob.glob(os.path.join(directory_path, '*.png'))

indexed_frames = index_video_frames(video_frame_paths)
search_results = search_video(test_image_paths, indexed_frames)

# Write the results to a CSV file compatible with evaluate.py
write_results_to_csv(search_results)
