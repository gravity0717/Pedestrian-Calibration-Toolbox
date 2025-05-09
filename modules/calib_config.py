from modules.calib_camera_linear import * 
from modules.calib_camera_nonlinear import *

def get_default_config():
    config = {
        'line_length': 0.5, # half-body 
        'n_lines':300, 
        'ransac_options': {'n_sample': 5, 
                            'min_iter': 20, 
                            'max_iter': 100, 
                            'threshold': 20, 
                            'heuristic': True, 
                            'regularization': 1, 
                            'msac': True},
        "ls_options": {'max_nfev': 100},
        "project_fun": project_k1, # project_k1 or project_k1_k2
        "bundle_fun": calib_camera_ba_k1, # calib_camera_ba_k1 or calib_camera_ba_k1_k2
        "final_BA_fun": calib_camera_ba_k1# calib_camera_ba_k1_f_static or calib_camera_ba_k1_k2_f_static
    }
    return config

def get_whole_body_config():
    config = get_default_config()
    line_config = {'line_length': 1.8, 'n_lines':100}
    config.update(line_config)
    return config

