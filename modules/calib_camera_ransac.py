import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from calib_camera_linear    import calib_camera_nlines, centerize_2d_pts, get_cx_cy, get_R

def reproject_error_triangulate_k1(unknown, a, b, config, line_length, f, k1, theta, phi, h):
    '''Calculate reprojection errors of `a` and `b` with `unknown`'''
    xy = np.array(unknown).reshape(-1, 2)
    A = np.hstack((xy, np.zeros((len(xy), 1))))              # Make (x, y) to (x, y, 0)
    B = np.hstack((xy, np.ones((len(xy), 1)) * line_length)) # Make (x, y) to (x, y, line_length)
    AB = np.vstack((A, B))
    ab = np.vstack((a, b))
    ab_proj = config["project_fun"](AB, f, k1, theta, phi, h)
    err = ab_proj - ab
    return err.ravel()

def triangulate(a, b, config, line_length, f, k1, theta, phi, h, ls_options={}):
    '''Estimate 3D points using nonlinear optimization'''
    # Optimize 3D points (named as `unknown`)
    # Note) The sequence of `unknown`: x_0, y_0, x_1, y_1, ...
    unknown = np.zeros((len(a) * 2))
    unknown[1::2] = 10 # Assign all `y_i` as 10
    result = least_squares(reproject_error_triangulate_k1, unknown, args=(a, b, config, line_length, f, k1, theta, phi, h), **ls_options)     
    xy = result['x'].reshape(-1, 2)
    A = np.hstack((xy, np.zeros((len(xy), 1))))              # Make (x, y) to (x, y, 0)
    B = np.hstack((xy, np.ones((len(xy), 1)) * line_length)) # Make (x, y) to (x, y, line_length)
    return A, B

def calib_camera_ransac(a, b, img_size, line_length=1, centerized=False, viz_loss=True ,config={}):
    '''Calibrate a camera using n-point RANSAC'''
    # Note) `max_nfev` in `ls_options` can control the number of iterations in least squares.
    # Prepare `a` and `b` by following Nakano's convention
    assert len(a) == len(b) and len(a) >= 2
    if not centerized:
        a = centerize_2d_pts(a, img_size)[:,:2]
        b = centerize_2d_pts(b, img_size)[:,:2]
    n = len(a)

    # config 
    ransac_options = config['ransac_options']
    ls_options = config['ls_options']
    
    ransac = {'n_sample' : 5, 'max_iter' : 1000, 'min_iter': 100, 'success_rate': 0.99, 'threshold': 1, 'verbose': True} # The default options of RANSAC
    # The minimum number of iterations
    ransac["min_iter"] =  int(np.log(1-ransac['success_rate']) / np.log(1 - 0.5**ransac['n_sample'])) 
    ransac.update(ransac_options)
    ransac['threshold2'] = ransac['threshold'] * ransac['threshold']

    # Find inliers using n-point RANSAC
    best_loss = n
    iter, max_iter = 0, ransac['max_iter']
    while iter <= max_iter:
        iter += 1
        # 1. Generate an hypothesis
        sample = random.sample(range(n), ransac['n_sample'])
        calib = config["bundle_fun"](a[sample], b[sample], img_size, config, line_length, 
                                centerized=True, ls_init=[], ls_options=ls_options, heuristic=ransac_options["heuristic"], 
                                regularization = ransac_options["regularization"])
        calib = np.hstack(calib[:-1])

        # 2. Evaluate the hypothesis
        A, B = triangulate(a, b, config, line_length, *calib)

        delta = np.hstack((config["project_fun"](A, *calib) - a, config["project_fun"](B, *calib) - b))
        error2 = np.sum(delta*delta, axis=1) # The squared reprojection error
                
        inlier_mask = error2 < ransac['threshold2']
        if ransac_options["msac"]:
            loss = sum(error2[inlier_mask]) / ransac['threshold2'] + (n - sum(inlier_mask)) # MSAC's robust kernel
        else:
            loss =  n - sum(inlier_mask) # RANSAC's original loss

        if ransac['verbose']:
            print(f'- RANSAC log) iter={iter:4d} | loss={loss:.1f}, focal={calib[0]:4.1f}, k1={calib[1]:.4f} | best_loss={best_loss:.2f}')
        if loss < best_loss:
            best_loss = loss
            best_calib = (calib.copy(), A.copy())
            best_inlier_mask = inlier_mask.copy()
            if viz_loss:
                best_losses = error2[inlier_mask].copy()
            inlier_ratio = sum(inlier_mask) / len(a)
            if inlier_ratio > 0:
                max_iter = int(np.log(1-ransac['success_rate']) / np.log(1 - inlier_ratio**ransac['n_sample'])) # Adaptive termination
                max_iter = min(max(max_iter, ransac['min_iter']), ransac['max_iter'])
            if ransac['verbose']:
                print(f'- RANSAC log) iter={iter:4d} | ***The best is updated (inlier_ratio: {inlier_ratio}, max_iter: {max_iter}).***')
    
    f, k1, theta, phi, h = best_calib[0]
    xy = best_calib[1][:,:2]
    xy = xy[best_inlier_mask]
    if viz_loss:
        return f, np.array([k1]), theta, phi, h, np.array(xy), best_inlier_mask, best_losses
    return f, np.array([k1]), theta, phi, h, np.array(xy), best_inlier_mask

def test_triangulate():
    '''Run tests for `triangulate()`'''
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 4
    line_length = 1

    # Generate synthetic data without noise
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab, _ = make_gaussian_outlier(ab, 100, 0.3)  # Make 30% of points as outliers

    a = ab[::2,:]
    b = ab[1::2,:]

    # Calibrate the camera
    calib_result = calib_camera_ba(a, b, cam_res, line_length, ls_options={'max_nfev': 100}, lambda1=10, lambda2=10)
    print(f'\n### N-point method (n={line_n})')
    print(f'* focal : {calib_result[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* theta : {np.rad2deg(calib_result[2]):.0f} [deg]')
    print(f'* phi   : {np.rad2deg(calib_result[3]):.0f} [deg]')
    print(f'* height: {calib_result[4]:.3f} [m]')

    # Triangulate points
    a = centerize_2d_pts(a, cam_res)[:,:2]
    b = centerize_2d_pts(b, cam_res)[:,:2]
    A, B = triangulate(a, b, line_length, *np.hstack(calib_result[:-1]), plot_residual=True)
    print(f'\n### True 3D lines (n={line_n})')
    print(AB[::2])
    print(AB[1::2])
    print(f'\n### Estimated 3D lines (n={line_n})')
    print(A)
    print(B)

def test_ransac_method():
    '''Run tests for this module'''
    import warnings
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 30
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, _ = make_gaussian_outlier(ab, 100, 0.1)  # Make 30% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]

    # Calibrate the camera
    calib_linear = calib_camera_nlines(a, b, cam_res, line_length)
    print(f'\n### n-point method (n={line_n})')
    print(f'* focal : {calib_linear[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* theta : {np.rad2deg(calib_linear[2]):.0f} [deg]')
    print(f'* phi   : {np.rad2deg(calib_linear[3]):.0f} [deg]')
    print(f'* height: {calib_linear[4]:.3f} [m]')

    calib_nonlin = calib_camera_ba(a, b, cam_res, line_length, ls_init=calib_linear, ls_options={'max_nfev': 1000})
    print(f'\n### BA method (n={line_n})')
    print(f'* focal  : {calib_nonlin[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_nonlin[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_nonlin[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_nonlin[3]):.0f} [deg]')
    print(f'* height : {calib_nonlin[4]:.3f} [m]')

    ransac_options = {'n_sample': 4, 'max_iter': 500, 'threshold': 8}
    with warnings.catch_warnings(): # Ignore warnings
        warnings.simplefilter('ignore', category=UserWarning)
        warnings.simplefilter('ignore', category=RuntimeWarning)
        calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, ransac_options=ransac_options)
    inlier_mask = calib_ransac[-2]

    print(f'\n### {ransac_options["n_sample"]}-point RANSAC method (n={line_n})')
    print(f'* focal  : {calib_ransac[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac[3]):.0f} [deg]')
    print(f'* height : {calib_ransac[4]:.3f} [m]')
    print(f'* inliers: {sum(inlier_mask)} / {len(a)}')

    calib_ransac_ba = calib_camera_ba(a[inlier_mask], b[inlier_mask], cam_res, line_length, ls_options={'max_nfev': 1000})
    print(f'\n### RANSAC + BA method (n={line_n})')
    print(f'* focal  : {calib_ransac_ba[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac_ba[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac_ba[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac_ba[3]):.0f} [deg]')
    print(f'* height : {calib_ransac_ba[4]:.3f} [m]')

    # Plot line data
    plt.xlim(0,cam_res[0])
    plt.ylim(cam_res[1],0)
    for i, (ai,bi) in enumerate(zip(a,b)):
        plt.plot([ai[0],bi[0]],[ai[1],bi[1]], color = 'k', linewidth=1.5)

    # Plot inliers
    for i, (ai,bi) in enumerate(zip(a[inlier_mask],b[inlier_mask])):
        plt.plot([ai[0],bi[0]],[ai[1],bi[1]], color = 'r', linewidth=1.5)

def test_ransac_method2():
    '''Run tests for this module'''
    import warnings
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 30
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, _ = make_gaussian_outlier(ab, 100, 0.5)  # Make 30% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]


    # RANSAC(HN) + BA(LN)
    ransac_options = {'n_sample': 3, 'min_iter': 20, 'max_iter': 500, 'threshold': 100, 'verbose': True, 'msac': True, 'lambda1': 1, 'lambda2': 1, 'heuristic': True}
    ls_options = {'max_nfev': 100}
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, ls_options=ls_options, ransac_options=ransac_options)
    
    print(f'\n### RANSAC(HN) + BA(LN) method (n={line_n})')
    print(f'* focal  : {calib_ransac[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac[3]):.0f} [deg]')
    print(f'* height : {calib_ransac[4]:.3f} [m]')
    print(f'* inliers: {sum(calib_ransac[-1])} / {len(a)}')  
    
def test_ransac_time():
    '''Run tests for this module'''
    import warnings
    from synthetic_lines import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 10
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, _ = make_gaussian_outlier(ab, 100, 0.3)  # Make 30% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]

    # RANSAC(HN) + BA(LN)
    ransac_options = {'n_sample': 3, 'min_iter': 20, 'max_iter': 500, 'threshold': 100, 'verbose': True, 'msac': True, 'lambda1': 1, 'lambda2': 1, 'heuristic': True}
    ls_options = {'max_nfev': 100}
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, ls_options=ls_options, ransac_options=ransac_options)

def test_ransac_method_with_constraint():
    '''Run tests for this module'''
    import warnings
    from lines_synthetic import gen_3d_lines_cone, project_3d_pts, add_gaussian_noise, make_gaussian_outlier

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    cam_distort = [-0.2, -0.1]
    line_n = 30
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    ab = project_3d_pts(AB, cam_focal, cam_res, cam_distort)
    ab = add_gaussian_noise(ab, noise_std)
    ab, _ = make_gaussian_outlier(ab, 100, 0.5)  # Make 30% of points as outliers
    a = ab[::2,:]
    b = ab[1::2,:]
    from calib_config import get_default_config
    config = get_default_config()
    config["constraint"] = True
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, config=config)
    
    print(f'\n### MSAC with constraint (n={line_n})')
    print(f'* focal  : {calib_ransac[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac[3]):.0f} [deg]')
    print(f'* height : {calib_ransac[4]:.3f} [m]')
    print(f'* inliers: {sum(calib_ransac[-1])} / {len(a)}')  
    
    config["constraint"] = True
    config["const_fctr"] = 100
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, config=config)
    
    print(f'\n### MSAC with 10 * constraint (n={line_n})')
    print(f'* focal  : {calib_ransac[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac[3]):.0f} [deg]')
    print(f'* height : {calib_ransac[4]:.3f} [m]')
    print(f'* inliers: {sum(calib_ransac[-1])} / {len(a)}')  
    
    
    config["constraint"] = False
    calib_ransac = calib_camera_ransac(a, b, cam_res, line_length, config=config)
    
    print(f'\n### MSAC without constraint (n={line_n})')
    print(f'* focal  : {calib_ransac[0]:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* (k1,k2): {calib_ransac[1]} (Truth: {cam_distort})')
    print(f'* theta  : {np.rad2deg(calib_ransac[2]):.0f} [deg]')
    print(f'* phi    : {np.rad2deg(calib_ransac[3]):.0f} [deg]')
    print(f'* height : {calib_ransac[4]:.3f} [m]')
    print(f'* inliers: {sum(calib_ransac[-1])} / {len(a)}')  
    
if __name__ == '__main__':
    # test_triangulate()
    # test_ransac_method()
    # test_ransac_method2()
    test_ransac_method_with_constraint()