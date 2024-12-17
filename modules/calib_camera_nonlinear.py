import numpy as np
from scipy.optimize import least_squares
from modules.calib_camera_linear import calib_camera_nlines, centerize_2d_pts, get_cx_cy, get_R
import matplotlib.pyplot as plt

def calib_camera_ba(a, b, img_size, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 0, heuristic=False):
    pass  

def calib_camera_ba_k1(a, b, img_size, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 1, heuristic=True):
    
    '''Calibrate a camera using bundle adjustment'''
    # Heuristic
    if heuristic:
        init_xy = np.zeros((len(a), 2))
        init_xy[:,-1] = 10
        ls_init = [np.mean(img_size), 0, np.pi/2, 0, 1, init_xy]
    
    # Linear 
    if ls_init is None or not heuristic:
        ls_init = calib_camera_nlines(a, b, img_size, line_length)
        if np.isnan([ls_init[0]]):
            init_xy = np.zeros((len(a), 2))
            init_xy[:,-1] = 10
            ls_init = [np.mean(img_size), 0 , np.pi/2, 0, 1, init_xy]
            
    # Prepare `a` and `b` by following Nakano's convention
    assert len(a) == len(b) and len(a) >= 2
    if not centerized:
        a = centerize_2d_pts(a, img_size)[:,:2]
        b = centerize_2d_pts(b, img_size)[:,:2]
    center = get_cx_cy(img_size)
    
    # Optimize the initial calibration result
    # Note) The sequence of `unknown`: f, dist_coef, theta, phi, h, (x_0, y_0, x_1, y_1, ...)
    
    unknown = np.hstack((ls_init[0], ls_init[1], ls_init[2:-1], ls_init[-1].ravel()))
    result = least_squares(reproject_error_ba_k1, unknown, args=(a, b, config, line_length, regularization), **ls_options)
    f, k1, theta, phi, h, *xy = result['x']
    return f, np.array([k1]), theta, phi, h, np.array(xy).reshape(-1, 2)

def calib_camera_ba_k1_f_static(a, b, img_size, f, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 1, heuristic=True):
        '''Calibrate a camera using bundle adjustment'''
        DEBUG = True if len(a) > 7 else False
        if DEBUG:
            print("*** DEBUG MODE ***")
            print(ls_init)
            
        # Heuristic
        if heuristic:
            init_xy = np.zeros((len(a), 2))
            init_xy[:,-1] = 10
            ls_init = [0, np.pi/2, 0, 1, init_xy]
        
        # Linear 
        if ls_init is None or not heuristic:
            ls_init = calib_camera_nlines(a, b, img_size, line_length)
            if np.isnan([ls_init[0]]):
                init_xy = np.zeros((len(a), 2))
                init_xy[:,-1] = 10
                ls_init = [0, np.pi/2, 0, 1, init_xy]
        if DEBUG:
            print(ls_init is None)
            print(heuristic)

        # Prepare `a` and `b` by following Nakano's convention
        assert len(a) == len(b) and len(a) >= 2
        if not centerized:
            a = centerize_2d_pts(a, img_size)[:,:2]
            b = centerize_2d_pts(b, img_size)[:,:2]
        center = get_cx_cy(img_size)
        
        # Optimize the initial calibration result
        # Note) The sequence of `unknown`: Wdist_coef, theta, phi, h, (x_0, y_0, x_1, y_1, ...)
        
        unknown = np.hstack((ls_init[0], ls_init[1], ls_init[2:-1], ls_init[-1].ravel()))
        if(DEBUG):
            print(f'Initial guess: {unknown}') 
        result = least_squares(reproject_error_ba_k1_f_static, unknown, args=(a, b, f, config, line_length, regularization), **ls_options)
        k1, theta, phi, h, *xy = result['x']
        return f, np.array([k1]), theta, phi, h, np.array(xy).reshape(-1, 2)
   
def reproject_error_ba_k1(unknown, a, b, config, line_length, regularization = 0):
    '''Calculate reprojection errors of `a` and `b` with `unknown`'''
    # 입력 데이터 타입 변환
    a = a.astype('float64')
    b = b.astype('float64')
    f, k1, theta, phi, h, *xy = unknown
    xy = np.array(xy).reshape(-1, 2).astype('float64')  # xy 타입 변환
    A = np.hstack((xy, np.zeros((len(xy), 1)))).astype('float64')  # 타입 변환
    B = np.hstack((xy, np.ones((len(xy), 1)) * line_length)).astype('float64')  # 타입 변환
    AB = np.vstack((A, B)).astype('float64')  # 타입 변환
    ab = np.vstack((a, b)).astype('float64')  # 타입 변환
    ab_proj = config['project_fun'](AB, f, k1, theta, phi, h).astype('float64')  # project 결과 타입 변환
    err = (ab_proj - ab).astype('float64')  # 계산 결과 타입 변환
    
    return np.hstack([err.ravel(), regularization*abs(k1)]).astype('float64')

def reproject_error_ba_k1_f_static(unknown, a, b, f, config, line_length, regularization = 0):
    
    '''Calculate reprojection errors of `a` and `b` with `unknown`'''
    # 입력 데이터 타입 변환
    a = a.astype('float64')
    b = b.astype('float64')
    k1, theta, phi, h, *xy = unknown
    xy = np.array(xy).reshape(-1, 2).astype('float64')  # xy 타입 변환
    A = np.hstack((xy, np.zeros((len(xy), 1)))).astype('float64')  # 타입 변환
    B = np.hstack((xy, np.ones((len(xy), 1)) * line_length)).astype('float64')  # 타입 변환
    AB = np.vstack((A, B)).astype('float64')  # 타입 변환
    ab = np.vstack((a, b)).astype('float64')  # 타입 변환
    ab_proj = config['project_fun'](AB, f, k1, theta, phi, h).astype('float64')  # project 결과 타입 변환
    err = (ab_proj - ab).astype('float64')  # 계산 결과 타입 변환
    
    return np.hstack([err.ravel(), regularization*abs(k1) ]).astype('float64')

def project(Xw, f, k1, k2, theta, phi, h):
    # Only used in graph_model_generator_nlines.py...
    return project_k1_k2(Xw, f, k1, k2, theta, phi, h)

def project_k1(Xw, f, k1 ,theta ,phi, h):
    '''Project 3D points on the 2D image plane'''
    R = get_R(theta, phi).astype('float64') 
    t = (-h * R[:, -1]).astype('float64')    
    Xc = (Xw @ R.T + t).astype('float64')    
    xn = (Xc[:, :2] / Xc[:, -1].reshape(-1, 1)).astype('float64')  
    r2 = ((xn[:, 0]**2 + xn[:, 1]**2).reshape(-1, 1)).astype('float64')
    xd = ((1 + k1*r2) * xn).astype('float64')
    return (f * xd).astype('float64')

def project_k1_k2(Xw, f, k1, k2, theta, phi, h):
    '''Project 3D points on the 2D image plane'''
    R = get_R(theta, phi).astype('float64') 
    t = (-h * R[:, -1]).astype('float64')    
    Xc = (Xw @ R.T + t).astype('float64')    
    xn = (Xc[:, :2] / Xc[:, -1].reshape(-1, 1)).astype('float64')  
    r2 = ((xn[:, 0]**2 + xn[:, 1]**2).reshape(-1, 1)).astype('float64')
    xd = ((1 + k1*r2 + k2*r2*r2) * xn).astype('float64')
    return (f * xd).astype('float64')

