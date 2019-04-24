from __future__ import print_function

import numpy as np
import time
import psutil
from scps import SCPS
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt


def conventional_ps(M, L, background_ind=None):
    """
    Lambertian Photometric stereo based on least-squares
    Woodham 1980
    :return: surface normal : numpy array of surface normal (p \times 3)
    """
    N = np.linalg.lstsq(L, M, rcond=None)[0]
    N = normalize(N, axis=0)
    if background_ind is not None:
        for i in range(N.shape[0]):
            N[i, background_ind] = 0
    return N


def plot_light_intensities(ground_truth=None, estimated=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ind = np.arange(0, len(ground_truth), 1)
    width = 0.35
    ax.bar(ind - width/2, ground_truth, width, color='SkyBlue', label='Ground truth')
    ax.bar(ind + width/2, estimated, width, color='IndianRed', label='Estimated')
    ax.set_ylabel('Light intensity (ratio)')
    ax.set_title('Light intensities')
    ax.legend()
    plt.show()


if __name__=='__main__':
    # Choose a method
    #METHOD = SCPS.LINEAR    # Linear solution method
    #METHOD = SCPS.FACTORIZATION    # Factorization based method
    METHOD = SCPS.ALTERNATE    # Alternating minimization method

    # Choose a dataset
    #DATA_FOLDERNAME = './data/bunny/bunny_lambert/'    # Lambertian diffuse with cast shadow
    DATA_FOLDERNAME = './data/bunny/bunny_lambert_noshadow/'    # Lambertian diffuse without cast shadow

    LIGHT_FILENAME = './data/bunny/lights.npy'
    MASK_FILENAME = './data/bunny/mask.png'
    GT_NORMAL_FILENAME = './data/bunny/gt_normal.npy'

    np.random.seed(1)
    # Semi-calibrated photometric stereo
    scps = SCPS()
    scps.load_mask(filename=MASK_FILENAME)    # Load mask image
    scps.load_lightnpy(filename=LIGHT_FILENAME)    # Load light matrix
    scps.load_npyimages(foldername=DATA_FOLDERNAME)    # Load observations
    ########################################################################################
    # Give fluctuation to light intensities (equivalent to rescaling scps.M's row vectors)
    E_gt = np.random.rand(scps.M.shape[0])
    E_gt = E_gt / max(E_gt.ravel())
    scps.M = np.diag(E_gt) @ scps.M

    # Conventional photometric stereo with assuming that the light intensities are uniform (or calibrated)
    N = conventional_ps(scps.M, scps.L, scps.background_ind)
    start = time.time()
    scps.solve(METHOD)    # Compute
    elapsed_time = time.time() - start
    print("Semi-Calibrated Photometric Stereo: elapsed_time:{0}".format(elapsed_time) + "[sec]")
    #scps.save_normalmap(filename="./est_normal")    # Save the estimated normal map
    #print("Light intensities (ratio)", np.diag(scps.E)/max(scps.E.ravel()))

    # Evaluate the estimate
    N_gt = psutil.load_normalmap_from_npy(filename=GT_NORMAL_FILENAME)    # read out the ground truth surface normal
    N_gt = np.reshape(N_gt, (scps.height*scps.width, 3))    # reshape as a normal array (p \times 3)
    angular_err = psutil.evaluate_angular_error(N_gt, scps.N.T, scps.background_ind)  # compute angular error
    angular_err_conventional = psutil.evaluate_angular_error(N_gt, N.T, scps.background_ind)  # compute angular error
    print('Mean Angular Error (SCPS):', np.mean(angular_err))
    print('Mean Angular Error (Conventional):', np.mean(angular_err_conventional))
    plot_light_intensities(E_gt, np.diag(scps.E)/max(scps.E.ravel()))
    psutil.disp_normalmap(normal=scps.N.T, height=scps.height, width=scps.width, name='SCPS surface normal estimates')
    #psutil.disp_normalmap(normal=N.T, height=scps.height, width=scps.width, name='Conventional')


