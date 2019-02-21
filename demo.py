from __future__ import print_function

import sys
sys.path.append("..")   # Adds higher directory to python modules path.

import numpy as np
import time
import psutil
from scps import SCPS
import cv2


# Choose a method
METHOD = SCPS.LINEAR    # Linear solution method
#METHOD = SCPS.FACTORIZATION    # Factorization based method
#METHOD = SCPS.ALTERNATE    # Alternating minimization method

DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'
LIGHT_FILENAME = './data/bunny/lights.npy'
MASK_FILENAME = './data/bunny/mask.png'
GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'

scps = SCPS()
scps.load_mask(filename=MASK_FILENAME)    # Load mask image
scps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
scps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations

# Give fluctuation to light intensities (equivalent to rescaling scps.M's row vectors)
for i in range(scps.M.shape[0]):
    scps.M[i, :] = scps.M[i, :] * (i % 2 + 1.0)

start = time.time()
scps.solve(METHOD)    # Compute
elapsed_time = time.time() - start
print("Semi-Calibrated Photometric Stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
#scps.save_normalmap(filename="./est_normal")    # Save the estimated normal map
print("Light intensities (ratio)", np.diag(scps.E)/max(scps.E.ravel()))

# Evaluate the estimate
N_gt = psutil.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
N_gt = np.reshape(N_gt, (scps.height*scps.width, 3))    # reshape as a normal array (p \times 3)
angular_err = psutil.evaluate_angular_error(N_gt, scps.N.T, scps.background_ind)    # compute angular error
print("Mean angular error [deg]: ", np.mean(angular_err[:]))
psutil.disp_normalmap(normal=scps.N.T, height=scps.height, width=scps.width)
print("done.")
