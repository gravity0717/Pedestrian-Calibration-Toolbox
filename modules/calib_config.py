from modules.calib_camera_linear import * 
from modules.calib_camera_nonlinear import *

def get_default_config():
    config = {
        'ransac_options': {'n_sample': 4, 
                            'min_iter': 20, 
                            'max_iter': 500, 
                            'threshold': 30, 
                            'heuristic': True, 
                            'regularization': 1, 
                            'msac': True},
        "ls_options": {'max_nfev': 50},
        "project_fun": project_k1, # project_k1 or project_k1_k2
        "bundle_fun": calib_camera_ba_k1, # calib_camera_ba_k1 or calib_camera_ba_k1_k2
        "final_BA_fun": calib_camera_ba_k1_f_static# calib_camera_ba_k1_f_static or calib_camera_ba_k1_k2_f_static
    }
    return config

