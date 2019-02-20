from __future__ import print_function

import sys
sys.path.append("..")   # Adds higher directory to python modules path.

import numpy as np
import time
from RobustPhotometricStereo import rpsutil
from scps import SCPS
import cv2


# Choose a method
METHOD = SCPS.LINEAR    # Linear solution method
#METHOD = SCPS.ALTERNATE    # Alternating minimization method

DATA_FOLDERNAME = '../RobustPhotometricStereo/data/bunny/bunny_lambert/'
LIGHT_FILENAME = '../RobustPhotometricStereo/data/bunny/lights.npy'
MASK_FILENAME = '../RobustPhotometricStereo/data/bunny/mask.png'
GT_NORMAL_FILENAME = '../RobustPhotometricStereo/data/bunny/gt_normal.npy'

scps = SCPS()
scps.load_mask(filename=MASK_FILENAME)    # Load mask image
scps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
scps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations
start = time.time()
scps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Semi-Calibrated Photometric Stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
print(scps.E)
#scps.save_normalmap(filename="./est_normal")    # Save the estimated normal map
#rpsutil.disp_normalmap(normal=scps.N.T, height=scps.height, width=scps.width)
