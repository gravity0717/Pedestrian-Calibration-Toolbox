import numpy as np
import cv2
import json
import os 
import matplotlib.pyplot as plt 
from collections import defaultdict
from ultralytics import YOLO 
import warnings

class DataExtractor:
    def __init__(self, line_length=0.5):
        # YOLO
        self.yolo = YOLO("yolo11n-pose.pt")
        self.line_length = line_length 

    def homography(self, keypoints): 
        lsh, rsh, lh, rh = keypoints
        pts1 = np.array([[rsh[0], rsh[1]],
                         [lsh[0], lsh[1]],
                         [lh[0], lh[1]], 
                         [rh[0], rh[1]]]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]]).reshape(-1, 1, 2).astype(np.float32)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        head_dot = H @ np.array([2, 1, 1])
        bottom_dot = H @ np.array([2, 3, 1])
        head_dot = head_dot[:2] / head_dot[2]
        bottom_dot = bottom_dot[:2] / bottom_dot[2]

        return head_dot, bottom_dot
    
    def extract(self, video_path):
        '''
           video_path: Video that you want to extract data from.
           
           Description:
                - Extract 300 line segments and save head and bottom points into a JSON file.
                - "line_length" should be the length of a pedestrian's upper body.
        '''
        heads, bottoms = [], []

        # Check if the video file exists
        if not os.path.isfile(video_path):
            print("The video file doesn't exist.")
            return

        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Failed to open the video file.")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        t = 0
        head_list = []
        bottom_list = []
    
        while cap.isOpened():
            success, image = cap.read()
            if not success or image is None:
                print("Failed to read frame or end of video reached.")
                break

            # Ensure the original image size is preserved
            original_shape = image.shape
            print(f"Original image shape: {original_shape}")

            # YOLO tracking
            try:
                results = self.yolo.track(image.copy(), persist=True)
            except Exception as e:
                print(f"YOLO tracking failed: {e}")
                break

            t += 1

            # Process every 4th frame
            if t % 4 != 0:
                continue

            if len(head_list) > 300:
                break

            # Get keypoints
            try:
                target_id = 0
                keypoints = [
                    results[0].keypoints.xy[target_id][joint].cpu().numpy()
                    for joint in [5, 6, 11, 12]
                ]  # 5: Left Shoulder, 6: Right Shoulder, 11: Left Hip, 12: Right Hip
            except (IndexError, AttributeError) as e:
                print(f"Keypoint extraction failed: {e}")
                continue

            # Extract data with homography
            try:
                head, bottom = self.homography(keypoints)
            except Exception as e:
                print(f"Homography calculation failed: {e}")
                continue

            # Draw line only if within image bounds
            if (0 <= head[0] <= cam_w and 0 <= head[1] <= cam_h) and \
               (0 <= bottom[0] <= cam_w and 0 <= bottom[1] <= cam_h):
                head_list.append(head.tolist())
                bottom_list.append(bottom.tolist())
                cv2.line(image, (int(head[0]), int(head[1])), (int(bottom[0]), int(bottom[1])), color=(255, 0, 0), thickness=4)

            # Display the video
            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        heads.append(head_list)
        bottoms.append(bottom_list)
        
        # Save data to JSON
        output_path = "data/data.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        with open(output_path, 'w') as f:
            result = {
                'a': bottoms,
                'b': heads,
                'l': self.line_length,
                'cam_w': cam_w,
                'cam_h': cam_h,
            }
            json.dump(result, f, indent=4)

        print(f"Data saved to {output_path}")
