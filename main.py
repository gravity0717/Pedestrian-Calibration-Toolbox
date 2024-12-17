from modules.utils import load_panoptic_data
from modules.calib_camera_linear import get_cx_cy, get_R
from modules.calib_config import get_default_config
from modules.calib_extract_data import DataExtractor
from modules.calib_sampler import SpatialSampler
from modules.calib_camera_ransac import *
import cv2 

if __name__ == "__main__":

    # 1. Extract data
    data_extractor = DataExtractor(line_length=1, set_whole_body=False) 
    data_extractor.extract('data/input.avi')
    # data_extractor.extract('data/PETS09-S2L1.avi')

    # 2. Load data 
    a, b, cam_res, line_length = load_panoptic_data('data/data.json')
    a = a.reshape(-1, 2)
    b = b.reshape(-1, 2)

    # 3. Data selection 
    spatial_sampler = SpatialSampler(cam_res, (50, 50))
    filter_index = spatial_sampler.run_sample(a)
    a, b = a[filter_index], b[filter_index]

    # Draw line segments 
    # img_bgr = cv2.imread('assets/MOT15_PETS09S2L1.jpg')
    # for ai, bi in zip(a, b):
    #     pt_a = tuple(map(int, ai))
    #     pt_b = tuple(map(int, bi))
    #     cv2.line(img_bgr, pt_a, pt_b, (255,0,0),2,-1)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("assets/line_segments.png", img_rgb)

    # 4. MSAC
    config = get_default_config()
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, config=config)
    inlier_mask = calib_ransac[-2]

    # 5. Bundle adjustment 
    cam_focal = calib_ransac[0]
    calib_bundle = config["final_BA_fun"](a[inlier_mask], b[inlier_mask], img_size=cam_res, f = cam_focal, config = config, line_length = line_length)
    f, dist_coeff, theta, phi, h, xy = calib_bundle

    # Save result 
    cx, cy = get_cx_cy(cam_res)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    R = get_R(theta, phi)
    T = -R @ np.array([0, 0, h])

    with open('data/calibration.txt', 'w') as f:
        f.write("RotationMatrices\n")
        for i in range(3):
            for j in range(3):
                f.write(str(R[i,j])+" ")
            f.write("\n")
        f.write("\nTranslationVectors\n")
        for i in range(3):
            f.write(str(int(T[i])*1000)+" ")
        f.write("\n\nIntrinsicMatrix\n")
        for i in range(3):
            for j in range(3):
                f.write(str(K[i,j])+" ")
            f.write("\n")

    with open('data/dist_coef.txt', 'w') as f:
        f.write("Distortion\n")
        f.write(str(dist_coeff[0])+" ")
        for i in range(3):
            f.write(str(0.0)+" ")