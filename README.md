# Pedestrian-based Camera Calibration Toolbox

The **Pedestrian-based Camera Calibration Toolbox** is a Python-based tool designed to automatically calibrate cameras using pedestrian joint information. It leverages YOLO-based pose tracking and RANSAC-based calibration techniques to compute intrinsic and extrinsic camera parameters from input videos.

---

## Installation and Dependencies

To use this project, the following libraries need to be installed.

### **Required Libraries**
- `numpy`
- `opencv-python`
- `scipy`
- `matplotlib`
- `ultralytics` (YOLO11 model)
- `lap`

### **Installation**
Run the following command to install the necessary libraries:
```bash
pip install numpy opencv-python scipy matplotlib ultralytics lap
```

---

## Key Files and Descriptions

### **1. calib_extract_data.py**
- **Purpose:** Extracts 300 line segments from a given video, based on the pedestrian’s upper body length, and saves the extracted data in a JSON file.
- **Key Features:**
  - Tracks pedestrian joints using YOLO.
  - Computes line endpoints (head and bottom) using homography.
  - Saves the extracted data to `data/data.json`.
- **Usage Example:**
  ```python
  from calib_extract_data import DataExtractor

  data_extractor = DataExtractor(line_length=0.5)
  data_extractor.extract('data/input.avi')  # Replace 'input.avi' with your video
  ```

---

### **2. main.py**
- **Purpose:** Processes the extracted data to compute the camera's intrinsic and extrinsic parameters, and saves the calibration results.
- **Workflow:**
  1. **Data Extraction:** Use `calib_extract_data.py` to generate JSON data.
  2. **Data Loading:** Load the extracted JSON data for calibration.
  3. **Data Sampling:** Use `SpatialSampler` to refine the dataset.
  4. **Initial Calibration (MSAC):** Estimate camera parameters using the MSAC algorithm (a variant of RANSAC).
  5. **Optimization (Bundle Adjustment):** Refine the results for higher accuracy.
  6. **Save Results:** Save the camera’s rotation matrix, translation vector, intrinsic matrix, and distortion coefficients to text files.
- **How to Run:**
  ```bash
  python main.py
  ```

---

## Data File Structure

The output files generated during the execution are structured as follows:

### **1. data/data.json**
Contains extracted pedestrian data:
- `a`: Bottom (foot) positions.
- `b`: Head positions.
- `l`: Pedestrian upper body length.
- `cam_w`: Camera resolution (width).
- `cam_h`: Camera resolution (height).

### **2. data/calibration.txt**
Contains camera calibration results:
- `RotationMatrices`: Camera rotation matrix.
- `TranslationVectors`: Camera translation vector.
- `IntrinsicMatrix`: Camera intrinsic parameters (focal length, optical center).

### **3. data/dist_coef.txt**
Contains lens distortion coefficients:
- `Distortion`: Distortion coefficient values.

---

## Example Workflow

### **Extract Data from Video**
1. Run `calib_extract_data.py`:
   ```python
   from calib_extract_data import DataExtractor

   extractor = DataExtractor(line_length=0.5)
   extractor.extract('data/input.avi')
   ```

### **Perform Calibration**
2. Run `main.py`:
   ```bash
   python main.py
   ```
   After execution, the calibration results will be saved in `data/calibration.txt` and `data/dist_coef.txt`.

---

## Notes
- **YOLO Model:** The `yolo11n-pose.pt` model is required and can be downloaded from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- **Input Video:** Ensure your input video is available at `data/input.avi`.
- **Output Directory:** All results will be saved in the `data/` directory. Ensure the directory exists before running the scripts.

---

## References
- [OpenCV Documentation](https://docs.opencv.org/)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)