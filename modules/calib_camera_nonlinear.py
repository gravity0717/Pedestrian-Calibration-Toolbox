import numpy as np
from scipy.optimize import least_squares
from calib_camera_linear import calib_camera_nlines, centerize_2d_pts, get_cx_cy, get_R
import matplotlib.pyplot as plt

def calib_camera_ba(a, b, img_size, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 0, heuristic=False):
    pass  

def calib_camera_ba_k1(a, b, img_size, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 0, heuristic=True):
    
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

def calib_camera_ba_k1_f_static(a, b, img_size, f, config, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 0, heuristic=False):
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
        # Note) The sequence of `unknown`: f, dist_coef, theta, phi, h, (x_0, y_0, x_1, y_1, ...)
        
        unknown = np.hstack((ls_init[0], ls_init[1], ls_init[2:-1], ls_init[-1].ravel()))
        if(DEBUG):
            print(f'Initial guess: {unknown}') 
        result = least_squares(reproject_error_ba_k1_f_static, unknown, args=(a, b, f, config, line_length, regularization), **ls_options)
        k1, theta, phi, h, *xy = result['x']
        return f, np.array([k1]), theta, phi, h, np.array(xy).reshape(-1, 2)
    
def calib_camera_ba_k1_k2(a, b, img_size, line_length=1, centerized=False, ls_init=[], ls_options={}, regularization = 0, heuristic=False):
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
   
        # Prepare `a` and `b` by following Nakano's convention
        assert len(a) == len(b) and len(a) >= 2
        if not centerized:
            a = centerize_2d_pts(a, img_size)[:,:2]
            b = centerize_2d_pts(b, img_size)[:,:2]
        center = get_cx_cy(img_size)
        
        # Optimize the initial calibration result
        # Note) The sequence of `unknown`: f, dist_coef, theta, phi, h, (x_0, y_0, x_1, y_1, ...)
        
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
    
    # ConstraintK 
    const = ConstaintK(f)
    const.get_cd(a, b, (1920, 1080), line_length, centerized=True)
    c = sum(const.get_constraint()) if config["constraint"] else 0
    return np.hstack([err.ravel(), regularization*abs(k1), c]).astype('float64')

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

def test_project():
    '''Run tests for `project()`'''
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    cam_theta = np.deg2rad(90+30)
    cam_phi = np.deg2rad(0)
    cam_height= 3
    line_n = 4
    line_length = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort, cam_theta, cam_phi, cam_height)
    print(f'* OpenCV:\n{centerize_2d_pts(ab, cam_res)[:,:2]}')

    # Project points
    ab_proj = project(AB, cam_focal, cam_distort[0], cam_distort[1], cam_theta, cam_phi, cam_height)
    print(f'* project():\n{ab_proj}')

def test_nonlinear_method():
    '''Run tests for `calib_camera_ba()`'''
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 5
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, _ = make_gaussian_outlier(ab, 100, 0.3) # Make 30% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]
    plt.scatter(AB[...,0],AB[...,1],c='g', label = 'GT')
    # Calibrate the camera
    calib_linear = calib_camera_nlines(a, b, cam_res, line_length)
    print(f'\n### n-point method (n={line_n})')
    print(f'* focal : {calib_linear[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* theta : {np.rad2deg(calib_linear[2]):.0f} [deg]')
    print(f'* phi   : {np.rad2deg(calib_linear[3]):.0f} [deg]')
    print(f'* height: {calib_linear[4]:.3f} [m]')
    xy = calib_linear[-1]
    plt.scatter(xy[...,0], xy[...,1], c = 'b', label="linear")
    # calib_nonlin = calib_camera_ba(a, b, cam_res, line_length, ls_init=calib_linear)
    # Note) The following `loss` and `f_scale` options enable a robust kernel and its bandwidth.
    calib_nonlin = calib_camera_ba(a, b, cam_res, line_length, ls_init=calib_linear, heuristic=True)
    print(f'\n### BA method (n={line_n})')
    print(f'* focal  : {calib_nonlin[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_nonlin[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_nonlin[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_nonlin[3]):.0f} [deg]')
    print(f'* height : {calib_nonlin[4]:.3f} [m]')
    xy = calib_nonlin[-1]
    plt.scatter(xy[...,0], xy[...,1], c = 'r', label="BA")

def test_ba():
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier, draw_2d_lines
    """ Test if the bundle adjustment works being effected by the inliers/outliers"""
    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, 0]
    # cam_distort = [0,0]
    line_n = 20
    line_length = 1
    noise_std = 1
    p = 0.5
    
    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, outlier_idx = make_gaussian_outlier(ab, 100, p) # Make p% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]

    # Draw 2D outliers and inliers with the color match outlier: red, inlier: blue
    draw_2d_lines(a,b, linewidth=2, img_size=cam_res, img_boundary=1, oultier_idx = None)
    plt.show()
    
    # Make ab_outliers and ab_inliers
    a_new = a.copy()    
    b_new = b.copy()    
    a_out = a_new[outlier_idx]
    b_out = b_new[outlier_idx]
    
    a_new = a.copy()
    b_new = b.copy()
    a_in = np.delete(a_new, outlier_idx, axis=0)
    b_in = np.delete(b_new, outlier_idx, axis=0)
    
    # BA with outliers
    calib = calib_camera_ba(a_out, b_out, cam_res, line_length, ls_options={"max_nfev":100},ls_init=[],lambda1=0, lambda2=0, heuristic=True)
    print(f'\n### BA method wiht Outliers ')
    print(f'* focal  : {calib[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib[3]):.0f} [deg]')
    print(f'* height : {calib[4]:.3f} [m]')
    
    # BA with inliers 
    calib = calib_camera_ba(a_in, b_in, cam_res, line_length, ls_options={"max_nfev":100},ls_init=[],lambda1=0, lambda2=0, heuristic=True)
    print(f'\n### BA method with Inliers')
    print(f'* focal  : {calib[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib[3]):.0f} [deg]')
    print(f'* height : {calib[4]:.3f} [m]')
    

    
if __name__ == '__main__':
    # test_project()
    test_nonlinear_method()
    # plt.legend()
    # plt.show()
    # test_ba()