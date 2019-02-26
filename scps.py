#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Semi-Calibrated Photometric Stereo in Python

Please refer to the following papers for algorithmic details.

    @inproceedings{SCPS2018,
        title   = {Semi-Calibrated Photometric Stereo},
        author  = {DongHyeon Cho, Yasuyuki Matsushita, Yu-Wing Tai, and In So Kweon},
        journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        year    = {2018}
    }

    @inproceedings{SCPS2016,
        title     = {Photometric Stereo Under Non-uniform Light Intensities and Exposures},
        author    = {Donghyeon Cho, Yasuyuki Matsushita, Yu-Wing Tai, and In So Kweon},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year      = {2016},
        volume    = {II},
        pages     = {170--186}
    }
"""
__author__ = "Yasuyuki Matsushita <yasumat@ist.osaka-u.ac.jp>"
__version__ = "0.1.0"
__date__ = "19 Feb 2019"

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import psutil


class SCPS(object):
    """
    Semi-calibrated photometric stereo class
    Given M and L, solve for N and E with the following objective function:
        min || M - ELN ||_F^2
    """
    # Choice of solution methods
    LINEAR = 0    # Linear solution method
    FACTORIZATION = 1    # Factorization based method
    ALTERNATE = 2    # Alternating minimization method

    SN_DIM = 3    # Surface normal dimension (because we live in the 3D space)

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
        self.L = psutil.load_lighttxt(filename).T

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
        self.L = psutil.load_lightnpy(filename).T

    def load_images(self, foldername=None, ext=None):
        """
        Load images in the folder specified by the "foldername" that have extension "ext"
        :param foldername: foldername
        :param ext: file extension
        """
        self.M, self.height, self.width = psutil.load_images(foldername, ext)
        self.M = self.M.T

    def load_npyimages(self, foldername=None):
        """
        Load images in the folder specified by the "foldername" in the numpy format
        :param foldername: foldername
        """
        self.M, self.height, self.width = psutil.load_npyimages(foldername)
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
        mask = psutil.load_image(filename=filename)
        mask = mask.reshape((-1, 1))
        self.foreground_ind = np.where(mask != 0)[0]
        self.background_ind = np.where(mask == 0)[0]

    def disp_normalmap(self, delay=0):
        """
        Visualize normal map
        :return: None
        """
        psutil.disp_normalmap(normal=self.N, height=self.height, width=self.width, delay=delay)

    def save_normalmap(self, filename=None):
        """
        Saves normal map as numpy array format (npy)
        :param filename: filename of a normal map
        :return: None
        """
        psutil.save_normalmap_as_npy(filename=filename, normal=self.N, height=self.height, width=self.width)

    def solve(self, method=LINEAR):
        if self.M is None:
            raise ValueError("Measurement M is None")
        if self.L is None:
            raise ValueError("Light L is None")
        if self.M.shape[0] != self.L.shape[0]:
            raise ValueError("Inconsistent dimensionality between M and L")

        if method == SCPS.LINEAR:
            self._solve_linear()
        elif method == SCPS.FACTORIZATION:
            self._solve_factorization()
        elif method == SCPS.ALTERNATE:
            self._solve_alternate()
        else:
            raise ValueError("Undefined solver")

    def _solve_linear(self):
        """
        Semi-calibrated photometric stereo
        solution method based on null space (linear)
        """
        self.N = np.zeros((self.SN_DIM, self.M.shape[1]))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        M = self.M[:, indices]
        # Look at pixels that are illuminated under ALL the illuminations
        illum_ind = np.where(np.min(M, axis=0) > 0.0)[0]
        f, p = M.shape
        Dl = sp.kron(-sp.identity(p), self.L)
        Drt = sp.lil_matrix((f, p*f))
        for i in range(len(illum_ind)):
            Drt.setdiag(M[:, illum_ind[i]], k=i*f)
        D = sp.hstack([Dl, Drt.T])
        u, s, vt = sp.linalg.svds(D, k=1, which='SM')    # Compute 1D (primary) null space of D
        null_space = vt.T.ravel()
        self.E = np.diag(1.0 / null_space[self.SN_DIM * p:])
        if np.mean(self.E) < 0.0:    # flip if light intensities are negative
            self.E *= -1.0
        self.N = np.linalg.lstsq(self.E @ self.L, self.M, rcond=None)[0]
        # The above operation is almost equivalent to obtaining the solution from the null space
        # self.N[:, indices] = np.reshape(null_space[:self.SN_DIM*p], (p, self.SN_DIM)).T
        # However, the null space method may be contaminated by shadows.
        self.N[:, indices] = normalize(self.N[:, indices], axis=0)
        return

    def _solve_factorization(self):
        """
        Semi-calibrated photometric stereo
        solution method based on factorization
        """
        self.N = np.zeros((self.SN_DIM, self.M.shape[1]))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        M = self.M[:, indices]
        # Look at pixels that are illuminated under ALL the illuminations
        illum_ind = np.where(np.min(M, axis=0) > 0.0)[0]
        # Step 1 factorize (uncalibrated photometric stereo step)
        f = M.shape[0]
        u, s, vt = np.linalg.svd(M[:, illum_ind], full_matrices=False)
        u = u[:, :self.SN_DIM]
        s = s[:self.SN_DIM]
        S_hat = u @ np.diag(np.sqrt(s))
        # Step 2 solve for ambiguity H
        A = np.zeros((2 * f, self.SN_DIM * self.SN_DIM))
        for i in range(f):
            s = S_hat[i, :]
            A[2 * i, :] = np.hstack([np.zeros(self.SN_DIM), -self.L[i, 2] * s, self.L[i, 1] * s])
            A[2 * i + 1, :] = np.hstack([self.L[i, 2] * s, np.zeros(self.SN_DIM), -self.L[i, 0] * s])
        u, s, vt = np.linalg.svd(A, full_matrices=False)
        H = np.reshape(vt[-1, :], (self.SN_DIM, self.SN_DIM)).T
        S_hat = S_hat @ H
        self.E = np.identity(f)
        for i in range(f):
            self.E[i, i] = np.linalg.norm(S_hat[i, :])
        self.N = np.linalg.lstsq(self.E @ self.L, self.M, rcond=None)[0]
        self.N[:, indices] = normalize(self.N[:, indices], axis=0)
        return

    def _solve_alternate(self):
        """
        Semi-calibrated photometric stereo
        solution method based on alternating minimization
        """
        max_iter = 1000   # can be changed
        tol = 1.0e-8    # can be changed
        self.N = np.zeros((self.SN_DIM, self.M.shape[1]))
        if self.foreground_ind is None:
            indices = range(self.M.shape[0])
        else:
            indices = self.foreground_ind
        M = self.M[:, indices]
        # Look at pixels that are illuminated under ALL the illuminations
        illum_ind = np.where(np.min(M, axis=0) > 0.0)[0]
        f = M.shape[0]
        self.E = np.ones(f)
        N_old = np.zeros((self.SN_DIM, M.shape[1]))
        for iter in range(max_iter):
            # Step 1 : Solve for N
            N = np.linalg.lstsq(np.diag(self.E) @ self.L, M, rcond=None)[0]
            # Step 2 : Solve for E
            LN = self.L @ N[:, illum_ind]
            for i in range(f):
                self.E[i] = (LN[i, :] @ M[i, illum_ind]) / (LN[i, :] @ LN[i, :])
            # normalize E
            self.E /= np.linalg.norm(self.E)
            if np.linalg.norm(N - N_old) < tol:
                break
            else:
                N_old = N
        self.N[:, indices] = normalize(N, axis=0)    # normalize N
        self.E = np.diag(self.E)     # convert to a diagonal matrix
        return


