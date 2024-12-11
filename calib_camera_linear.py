import numpy as np
import warnings
from itertools import combinations
import scipy.optimize

def get_cx_cy(img_size):
    '''Return (cx, cy) from the image resolution'''
    img_w, img_h = img_size
    return np.array([(img_w - 1) / 2, (img_h - 1) / 2])

def get_R(theta, phi):
    '''Return a 3D rotation matrix by following Nakano's convention'''
    return np.array([[np.cos(phi), -np.cos(theta)*np.sin(phi),  np.sin(theta)*np.sin(phi)],
                     [np.sin(phi),  np.cos(theta)*np.cos(phi), -np.sin(theta)*np.cos(phi)],
                     [          0,  np.sin(theta)            ,  np.cos(theta)]]) # From Equation (3)

def centerize_2d_pts(pts, img_size):
    '''Centerize 2D points by following Nakano's convention'''
    center = get_cx_cy(img_size)
    pts = pts - center
    pts = np.hstack((pts, np.ones((len(pts), 1)))) # To homogeneous notation
    return pts

def calib_camera_2lines(a, b, img_size, line_length=1, centerized=False):
    '''Calibrate a camera using Nakano's 2-point linear method'''
    # Prepare `a` and `b` by following Nakano's convention
    assert len(a) == len(b) and len(a) >= 2
    if not centerized:
        a = centerize_2d_pts(a[:2,:], img_size)
        b = centerize_2d_pts(b[:2,:], img_size)
    elif len(a[0]) == 2:
        a = np.hstack((a, np.ones((len(a), 1)))) # To homogeneous notation
        b = np.hstack((b, np.ones((len(b), 1)))) # To homogeneous notation

    # Solve `M @ v = 0 such that v > 0` in Equation (11) and (12)
    M = np.vstack((-a[0], a[1], b[0], -b[1])).T # M: (3 x 4)
    _, _, Vh = np.linalg.svd(M.T @ M)
    v = Vh[-1]
    if (v <= 0).all():
        v *= -1
    if (v < 0).any():
        warnings.warn('At least one element of `v` is negative.')

    # Calculate `f` using Equation (15)
    lm, mu = v[:2, np.newaxis], v[2:, np.newaxis]
    c = mu[0] * b[0] - lm[0] * a[0]
    d = lm[1] * a[1] - lm[0] * a[0]
    f2 = -(c[0]*d[0] + c[1]*d[1]) / (c[2]*d[2])
    if f2 < 0:
        warnings.warn('`f^2` is negative.')
        f2 *= -1
    f = np.sqrt(f2)

    # Calculate `l`, `r3`, `theta`, and `phi` using Equation (16) and (17)
    K_inv = np.diag([1/f, 1/f, 1])
    l = np.linalg.norm(K_inv @ c)
    r3 = K_inv @ c / l
    theta = np.arccos(r3[2])
    phi = np.arctan2(r3[0]/np.sin(theta), -r3[1]/np.sin(theta))

    # Correct scale with the known `line_length`
    scale_factor = line_length / l
    lm *= scale_factor
    mu *= scale_factor

    # Calculate `x_i`, `y_i`, and `h` using Equation (19)
    R = get_R(theta, phi)
    Q = np.diag([1, 1, -1])
    xyh = Q @ R.T @ K_inv @ (lm * a).T
    h = np.mean(xyh[-1])
    return f, np.zeros(1), theta, phi, h, xyh[:2].T

def calib_camera_nlines(a, b, img_size, line_length=1, centerized=False):
    '''Calibrate a camera using Nakano's n-point linear method'''
    # Prepare `a` and `b` by following Nakano's convention
    
    try:assert len(a) == len(b) and len(a) >= 2
    except:breakpoint()
    if not centerized:
        a = centerize_2d_pts(a, img_size)
        b = centerize_2d_pts(b, img_size)
    elif len(a[0]) == 2:
        a = np.hstack((a, np.ones((len(a), 1)))) # To homogeneous notation
        b = np.hstack((b, np.ones((len(b), 1)))) # To homogeneous notation

    # Solve `M @ v = 0 such that v > 0` in Equation (21) and (22)
    n = len(a)
    M = np.zeros((3*(n-1), 2*n), dtype=a.dtype)
    for i in range(1, n):
        s, e = (3*(i-1), 3*i)
        M[s:e, 0] = -a[0]
        M[s:e, i] = a[i]
        M[s:e, n] = b[0]
        M[s:e, n+i] = -b[i]
    _, _, Vh = np.linalg.svd(M.T @ M)
    v = Vh[-1]
    if (v <= 0).all():
        v *= -1
    if (v < 0).any():
        warnings.warn('At least one element of `v` is negative.')

    # Calculate `f` using Equation (24)
    lm, mu = v[:n, np.newaxis], v[n:, np.newaxis]
    c = mu * b - lm * a
    d = lm * a - lm[0] * a[0]
    f_num = sum([(ci[0]*di[0] + ci[1]*di[1]) * ci[2]*di[2] for ci, di in zip(c[1:], d[1:])])
    f_den = sum([(ci[2]*di[2])**2 for ci, di in zip(c[1:], d[1:])])
    if f_num > 0:
        warnings.warn('`f_numerator` is positive.')
    if f_den <= 0:
        warnings.warn('`f_denominator` is negative or equal to zero.')
    f = np.sqrt(-f_num / f_den)

    # Calculate `l`, `r3`, `theta`, and `phi` using Equation (27)
    K_inv = np.diag([1/f, 1/f, 1])
    avg_K_inv_c = np.mean(c @ K_inv.T, axis=0)
    l = np.linalg.norm(avg_K_inv_c)
    r3 = avg_K_inv_c / l
    theta = np.arccos(r3[2])
    phi = np.arctan2(r3[0]/np.sin(theta), -r3[1]/np.sin(theta))

    # Correct scale with the known `line_length`
    scale_factor = line_length / l
    lm *= scale_factor
    mu *= scale_factor
    # Calculate `x`, `y`, and `h` using Equation (19)
    R = get_R(theta, phi)
    Q = np.diag([1, 1, -1])
    p = Q @ R.T @ K_inv @ (lm * a).T
    q = Q @ R.T @ K_inv @ (mu * b).T
    x = (p[0,:] + q[0,:]) / 2
    y = (p[1,:] + q[1,:]) / 2
    h = sum(p[2,:] + q[2,:] + line_length) / 2 / n
    return f, np.zeros(1), theta, phi, h, np.vstack((x, y)).T

def calib_camera_nlines_single_view(a, b, img_size, centerized=False, mode = "iso"):
    '''Calibrate a camera inspired by "Single View Physical Distance Estimation using Human Pose". '''
    """
    Calculate focal length along the discription written in the paper "Single View Physical Distance Estimation using Human Pose (ICCV 2021)"
    
    Args:
        a (np.array): 2D foot keypoints of data. (u,v)
        b (np.array): 2D head keypoints of data. (u,v)
        centerized 
        mode (str, optional): ani(Aniotropic means fx != fy) and iso(isotropic fx =fy ). 
        Defaults to "Ani".
    Returns:
        Focal length
    """
    try:assert len(a) == len(b) and len(a) >= 2
    except:breakpoint()
    if not centerized:
        a = centerize_2d_pts(a, img_size)
        b = centerize_2d_pts(b, img_size)
    elif len(a[0]) == 2:
        a = np.hstack((a, np.ones((len(a), 1)))) # To homogeneous notation
        b = np.hstack((b, np.ones((len(b), 1)))) # To homogeneous notation

    # Av = 0
    A = []
    for _b,_a in zip(b, a):
        e = np.cross(_b,_a)
        A.append(e)
    A =np.array(A) 
    _,_,Vt = np.linalg.svd(A)
    v = Vt[:][-1]

       # Solve `M @ v = 0 such that v > 0` in Equation (11) and (12)
    M = np.vstack((-a[0], a[1], b[0], -b[1])).T # M: (3 x 4)
    _, _, Vh = np.linalg.svd(M.T @ M)
    w = Vh[-1]
    if (w <= 0).all():
        w *= -1
    if (w < 0).any():
        warnings.warn('At least one element of `w` is negatiwe.')

    # Calculate `f` using Equation (15)
    lm_t, lm_b = w[:n, np.newaxis], w[n:, np.newaxis]
    
    # Get deltas
    comb_100_2 = combinations(np.arange(n), 2)
    deltas = np.array([lm_b[i] * a[i] -lm_b[j] * a[j] for i,j in list(comb_100_2)])

    # Focal lengths 
    if  mode == 'iso':
        numerator = []
        denominator = []

        for delta in deltas:
            n = -np.dot(v[:2], delta[:2])
            d = v[2] * delta[2]
            numerator.append(n)
            denominator.append(d)

        # Using logsumexp deal with Overflow 
        n_sum = sum(numerator)
        d_sum = sum(denominator)
        f_square = n_sum/d_sum
        f = np.sqrt(f_square)
        return f
    else:
        B = []
        y = []
        for delta in deltas:
            # Component-wise
            B.append(v[:2] * delta[:2])
            y.append(v[2] * delta[2])
        B = np.array(B)
        y = np.array(y)

        # X is [1/f_x^2, 1/f_y^2]
        x,_ = scipy.optimize.nnls(B,y)
        x = x.astype(np.float64)

        # Get fx , fy 
        fx = np.sqrt(1/x[0])
        fy = np.sqrt(1/x[1])
        return fx, fy 

def test_linear_methods(draw_lines=True):
    '''Run tests for this module'''
    import matplotlib.pyplot as plt
    from lines_synthetic import gen_3d_lines_cone, gen__lines_line, project_3d_pts, add_gaussian_noise, draw_2d_lines

    # Configure a camera
    cam_focal = 800
    cam_res = (1920, 1080)
    line_n = 4
    line_length = 1
    noise_std = 1

    # Generate synthetic data
    AB = gen_3d_lines_cone(line_n, line_length)
    print(f'\n### Generated lines')
    print(f'* (x, y):\n{AB[::2,:2]}')

    ab = project_3d_pts(AB, cam_focal, cam_res)
    ab = add_gaussian_noise(ab, noise_std)
    a = ab[::2,:]
    b = ab[1::2,:]
    
    # Calibrate the camera using Nakano's 2-point method
    calib_focal, _, calib_theta, calib_phi, calib_h, calib_xy = calib_camera_2lines(a, b, cam_res, line_length)
    print(f'\n### 2-point method')
    print(f'* focal : {calib_focal:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* theta : {np.rad2deg(calib_theta):.0f} [deg]')
    print(f'* phi   : {np.rad2deg(calib_phi):.0f} [deg]')
    print(f'* height: {calib_h:.3f} [m]')
    print(f'* (x, y):\n{calib_xy}')

    # Calibrate the camera using Nakano's n-point method
    calib_focal, _, calib_theta, calib_phi, calib_h, calib_xy = calib_camera_nlines(a, b, cam_res, line_length)
    print(f'\n### n-point method (n={line_n})')
    print(f'* focal : {calib_focal:.1f} [pixel] (Truth: {cam_focal:.1f})')
    print(f'* theta : {np.rad2deg(calib_theta):.0f} [deg]')
    print(f'* phi   : {np.rad2deg(calib_phi):.0f} [deg]')
    print(f'* height: {calib_h:.3f} [m]')
    print(f'* (x, y):\n{calib_xy}')

    # Calibrat the camera using Fei's 2-point method 
    calib_focal = calib_camera_nlines_single_view(a,b, cam_res)
    if draw_lines:
        # Draw 2D lines
        draw_2d_lines(a, b, img_size=cam_res)
        plt.show()

if __name__ == '__main__':
    test_linear_methods()
    