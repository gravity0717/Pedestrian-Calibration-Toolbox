import random
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import json 
from tqdm import tqdm
from scipy.optimize import minimize 
from .calib_camera_linear import get_R
from collections import defaultdict
from copy import deepcopy

# Load data 
def load_panoptic_data(path):
    """
    Load data a,b acquired panoptic 
    """
    with open(path, 'r') as f:
        from_file = json.load(f)
        
        # Camera resolution
        cam_w = from_file['cam_w']
        cam_h = from_file['cam_h']
        cam_res = (cam_w, cam_h)
        
        # Line Data 
        a_json = from_file['a']
        b_json = from_file['b']
        a = np.asarray(a_json)
        b = np.asarray(b_json)
        l = from_file['l']
        
    return a, b, cam_res, l
