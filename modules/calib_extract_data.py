import numpy as np
import cv2
import json
import os 
from ultralytics import YOLO 

class DataExtractor:
    def __init__(self, line_length=0.7, set_whole_body=False, set_multi_person=True):
        # YOLO
        self.yolo = YOLO("yolo11n-pose.pt")
        self.line_length = line_length 
        self.set_whole_body = set_whole_body # True: whole body, False: upper body 
        self.set_multi_person = set_multi_person # True: detect multi person, False: detect single person 

    def homography(self, keypoints): 
        left_up, right_up, left_down, right_down = keypoints
        pts1 = np.array([[right_up[0], right_up[1]],
                         [left_up[0], left_up[1]],
                         [left_down[0], left_down[1]], 
                         [right_down[0], right_down[1]]]).reshape(-1, 1, 2).astype(np.float32)
        pts2 = np.array([[1, 1], [3, 1], [3, 3], [1, 3]]).reshape(-1, 1, 2).astype(np.float32)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        
        head_dot = H @ np.array([2, 1, 1])
        bott_dot = H @ np.array([2, 3, 1])
        head_dot = head_dot[:2] / head_dot[2]
        bott_dot = bott_dot[:2] / bott_dot[2]

        return head_dot, bott_dot
    
    def averaging(self, keypoints):
        left_up, right_up, left_down, right_down = keypoints

        head_dot = (left_up + right_up)/2 
        bott_dot = (left_down + right_down)/2
        return head_dot, bott_dot
    
    def get_head_to_feet_joints(self):
        return [1, 2, 15, 16] # 1: Left Eye 2: Right Eye 15: Left Ankle 16: Right Ankle
    
    def get_shoulder_to_hip_joints(self):
        return [5, 6, 11, 12] # 5: Left Shoulder 6: Right Shoulder 11: Left Hip 12: Right Hip
    
    def extract(self, video_path, n_lines=500):
        '''
           video_path: Video that you want to extract data from.
           
           Description:
                - Extract n line segments and save head and bottom points into a JSON file.
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

        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        t = 0
        head_final = []
        bott_final = []
    
        while cap.isOpened():
            success, image = cap.read()
            t += 1

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

            if len(head_final) > n_lines:
                break

            # Get keypoints
            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
                if not self.set_multi_person: 
                    track_ids = [int(len(track_ids)/2)]
                
                if self.set_whole_body:
                    joints = self.get_head_to_feet_joints()
                else:
                    joints = self.get_shoulder_to_hip_joints()
                
                torsos = []
                for track_id in range(len(track_ids)):
                    torso = [
                        results[0].keypoints.xy[track_id][joint].cpu().numpy()
                        for joint in joints
                    ]  
                    torsos.append(torso)
            except (IndexError, AttributeError) as e:
                print(f"Keypoint extraction failed: {e}")
                continue

            # Extract data with homography or averaging (TODO)
            try:
                heads, botts = [], []
                for torso in torsos:
                    head, bott = self.homography(torso)
                    heads.append(head)
                    botts.append(bott)
            except Exception as e:
                print(f"Homography calculation failed: {e}")
                continue

            # Draw line only if within image bounds

            for h, b in zip(heads, botts):
                if (0 <= h[0] <= cam_w and 0 <= h[1] <= cam_h) and \
                (0 <= b[0] <= cam_w and 0 <= b[1] <= cam_h):
                    cv2.line(image, (int(h[0]), int(h[1])), (int(b[0]), int(b[1])), color=(255, 0, 0), thickness=4)
                    head_final.append(h.tolist())
                    bott_final.append(b.tolist())
                else: continue

            # Display the video
            cv2.imshow("Video", image)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Save data to JSON
        seq = video_path[:-4]
        output_path = f"{seq}.json"
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
        with open(output_path, 'w') as f:
            result = {
                'a': bott_final,
                'b': head_final,
                'l': self.line_length,
                'cam_w': cam_w,
                'cam_h': cam_h,
            }
            json.dump(result, f, indent=4)
        print(f"============ Data saved to {output_path} with {len(head_final)} line segments. =============")
