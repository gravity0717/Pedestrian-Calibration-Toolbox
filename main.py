from modules.utils import load_panoptic_data
from modules.calib_camera_linear import get_cx_cy, get_R
from modules.calib_config import get_default_config
from modules.calib_extract_data import DataExtractor
from modules.calib_sampler import SpatialSampler
from modules.calib_camera_ransac import *
import cv2 

if __name__ == "__main__":
    for seq in ["MOT17_02","MOT17_04","MOT17_05","MOT17_09","MOT17_10","MOT17_11","MOT17_13"]:

        # 1. Extract data
        data_extractor = DataExtractor(line_length=0.5, set_whole_body=False) 
        data_extractor.extract(f'data/{seq}.avi')

        # 2. Load data 
        a, b, cam_res, line_length = load_panoptic_data(f'data/data_{seq}.json')
        a = a.reshape(-1, 2)
        b = b.reshape(-1, 2)

        # 3. Data selection 
        spatial_sampler = SpatialSampler(cam_res, (50, 50))
        filter_index = spatial_sampler.run_sample(a)
        a, b = a[filter_index], b[filter_index]

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

        with open(f'data/calibration_{seq}.txt', 'w') as f:
            f.write("RotationMatrices\n")
            for i in range(3):
                for j in range(3):
                    f.write(str(R[i,j])+" ")
                f.write("\n")
            f.write("\nTranslationVectors\n")
            for i in range(3):
                f.write(str(float(T[i])*1000)+" ")
            f.write("\n\nIntrinsicMatrix\n")
            for i in range(3):
                for j in range(3):
                    f.write(str(K[i,j])+" ")
                f.write("\n")

        with open(f'data/dist_coef_{seq}.txt', 'w') as f:
            f.write("Distortion\n")
            f.write(str(dist_coeff[0])+" ")
            for i in range(3):
                f.write(str(0.0)+" ")