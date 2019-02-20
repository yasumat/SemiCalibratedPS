#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Semi-Calibrated Photometric Stereo in Python
"""
__author__ = "Yasuyuki Matsushita <yasumat@ist.osaka-u.ac.jp>"
__version__ = "0.1.0"
__date__ = "19 Feb 2019"

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from RobustPhotometricStereo import rpsutil


class SCPS(object):
    """
    Semi-calibrated photometric stereo class
    Given M and L, solve for N and E with the following objective function:
        min || M - ELN ||_F^2
    """
    # Choice of solution methods
    LINEAR = 0    # Linear solution method
    TMP = 1    # tmp
    ALTERNATE = 2    # Alternating minimization method

    def __init__(self):
        self.M = None   # measurement matrix in numpy array
        self.L = None   # light direction matrix in numpy array
        self.E = None   # diagonal light intensity matrix in numpy array
        self.N = None   # surface normal matrix in numpy array
        self.height = None  # image height
        self.width = None   # image width
        self.foreground_ind = None    # mask (indices of active pixel locations (rows of M))
        self.background_ind = None    # mask (indices of inactive pixel locations (rows of M))

    def load_lighttxt(self, filename=None):
        """
        Load light file specified by filename.
        The format of lights.txt should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.txt
        """
        self.L = rpsutil.load_lighttxt(filename)
        self.L = self.L.T

    def load_lightnpy(self, filename=None):
        """
        Load light numpy array file specified by filename.
        The format of lights.npy should be
            light1_x light1_y light1_z
            light2_x light2_y light2_z
            ...
            lightf_x lightf_y lightf_z

        :param filename: filename of lights.npy
        """
        self.L = rpsutil.load_lightnpy(filename)
        self.L = self.L.T

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = rpsutil.load_images(foldername, ext)
        self.M = self.M.T

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = rpsutil.load_npyimages(foldername)
        self.M = self.M.T

    def load_mask(self, filename=None):
        """
        Load mask image and set the mask indices
        In the mask image, pixels with zero intensity will be ignored.
        :param filename: filename of the mask image
        :return: None
        """
        if filename is None:
            raise ValueError("filename is None")
        mask = rpsutil.load_image(filename=filename)
        mask = mask.T
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        rpsutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        rpsutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self, method=LINEAR):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[0] != self.L.shape[0]:
            raise ValueError("Inconsistent dimensionality between M and L")

        if method == SCPS.LINEAR:
            self._solve_linear()
        elif method == SCPS.ALTERNATE:
            self._solve_alternate()
        else:
            raise ValueError("Undefined solver")

    def _solve_linear(self):
        f, p = self.M.shape
        Ip = sp.identity(p)
        Dl = sp.kron(-Ip, self.L)
        Drt = sp.lil_matrix((f, p*f))
        for i in range(p):
            Drt.setdiag(self.M[:, i], k=i*f)
        D = sp.hstack([Dl, Drt.T])
        u, s, vt = sp.linalg.svds(D, k=2)
        print(s)
        print(u)
        print(p, f)
        return

    def _solve_alternate(self):
        """
        Derive solution by alternating minimization
        """
        max_iter = 100
        tol = 1.0e-6
        f = self.L.shape[0]
        self.E = np.identity(f)
        E_old = np.zeros((f, f))
        for iter in range(max_iter):
            # Step 1 : Solve for N
            self.N = np.linalg.lstsq(self.E @ self.L, self.M, rcond=None)[0]
            # Step 2 : Solve for E
            LN = self.L @ self.N
            for i in range(self.E.shape[0]):
                self.E[i, i] = (LN[i, :] @ self.M[i, :]) / (LN[i, :] @ LN[i, :])
            # normalize E
            self.E /= np.linalg.norm(self.E)
            if np.linalg.norm(self.E - E_old) < tol:
                break
            else:
                E_old = self.E
        # normalize N
        self.N = normalize(self.N, axis=0)
        return

