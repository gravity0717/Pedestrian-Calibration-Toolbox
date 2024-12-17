import numpy as np
import cv2
import json
import os 
from ultralytics import YOLO 

class DataExtractor:
    def __init__(self, line_length=0.5, set_whole_body=False):
        # YOLO
        self.yolo = YOLO("yolo11n-pose.pt")
        self.line_length = line_length 
        self.set_whole_body = set_whole_body #True: whole body, False: upper body 

    def homography(self, keypoints): 
        left_up, right_up, left_down, right_down = keypoints
        pts1 = np.array([[right_up[0], right_up[1]],
                         [left_up[0], left_up[1]],
                         [left_down[0], left_down[1]], 
                         [right_down[0], right_down[1]]]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]]).reshape(-1, 1, 2).astype(np.float32)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        head_dot = H @ np.array([2, 1, 1])
        bottom_dot = H @ np.array([2, 3, 1])
        head_dot = head_dot[:2] / head_dot[2]
        bottom_dot = bottom_dot[:2] / bottom_dot[2]

        return head_dot, bottom_dot
    
    def get_head_to_feet_joints(self):
        return [1, 2, 15, 16] # 1: Left Eye 2: Right Eye 15: Left Ankle 16: Right Ankle
    
    def get_shoulder_to_hip_joints(self):
        return [5, 6, 11, 12] # 5: Left Shoulder 6: Right Shoulder 11: Left Hip 12: Right Hip
    
    def extract(self, video_path):
        '''
           video_path: Video that you want to extract data from.
           
           Description:
                - Extract 300 line segments and save head and bottom points into a JSON file.
                - "line_length" should be the length of a pedestrian's upper body.
        '''

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
        heads = []
        bottoms = []
    
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
            if t % 3 != 0:
                continue

            if len(heads) > 1000:
                break

            # Get keypoints
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                target_id = int(len(track_ids)/2)
                
                if self.set_whole_body:
                    joints = self.get_head_to_feet_joints()
                else:
                    joints = self.get_shoulder_to_hip_joints()
                keypoints = [
                    results[0].keypoints.xy[target_id][joint].cpu().numpy()
                    for joint in joints
                ]  
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
                heads.append(head.tolist())
                bottoms.append(bottom.tolist())
                cv2.line(image, (int(head[0]), int(head[1])), (int(bottom[0]), int(bottom[1])), color=(255, 0, 0), thickness=4)

            # Display the video
            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
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

        print(f"============ Data saved to {output_path} with {len(heads)} line segments. =============")
