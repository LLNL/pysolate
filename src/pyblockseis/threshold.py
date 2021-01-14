# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Ana Aguiar Moya (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea4@llnl.gov)
#
# Non-linear thresholding operations using CWT, functions adapted from
# the BCseis ver1.1 MATLAB package by Charles A. Langston and S. Mostafa Mousavi.

import numpy as np
from scipy.interpolate import interp1d


def ecdf(x):
    """
    Empirical cumulative probability distribution
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    
    return (ys, xs)


def noise_model(Wx, delta, noise_starttime, noise_endtime, nsigma_method, nlbound):
    """
    Calculates noise model and threshold function.
    
    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param delta: sampling interval.
    :type delta: float
    :param noise_starttime: noise start time.
    :type noise_starttime: float
    :param noise_endtime: noise end time.
    :type noise_endtime: float
    :param nsigma_method: block thresholding method.
    :type nsigma_method: str
    :param nlbound: noise level lower bound in percent.
    :type nlbound: float
    :param nsigma_gauss: umber of std for block threshold using Gaussian statistic.
    :type nsigma_gauss: float
    :return M: mean of noise model.
    :rtype M: :class:`numpy.ndarray`
    :return S: standard deviation of noise model.
    :rtype S: :class:`numpy.ndarray`
    :return P: threshold of the noise signal.
    :rtype P: :class:`numpy.ndarray`
    """
    # Get the time window
    #newbeg = int(np.round(noise_starttime/delta) +1)
    #newend = int(np.round(noise_endtime/delta) +1)
    newbeg = int(np.round(noise_starttime/delta)) # python indexing starts with zero
    newend = int(np.round(noise_endtime/delta)+1)

    # calculate the mean and standard deviations at each scale in the time-scale window
    M = np.mean(np.abs(Wx[:, newbeg:newend]), axis=1)
    S = np.std(np.abs(Wx[:, newbeg:newend]), axis=1)
    if nsigma_method == "ECDF":
        # Estimate empirical CDF for each scale,
        # calculate noise threshold at the desired confidence level.
        nrow, ncol = Wx.shape
        conf = 1.0 - nlbound*0.01
        n_noise = newend-newbeg+1
        P = np.zeros(nrow)
        for k in range(nrow):
            W = []
            W[0:n_noise] = np.abs(Wx[k,newbeg:newend])
            f,x = ecdf(W)
            fnew = interp1d(f, x, kind='linear')
            P[k] = fnew(conf)
    else:
        if nsigma_method == "donoho":
            # compute Donoho's Threshold Criterion
            nsigma = np.sqrt(2*np.log10(newend-newbeg +1))
        else:
            nsigma = float(nsigma_method)
        # assuming Gaussian statistics
        P = M + nsigma * S
            
    return (M, S, P, newbeg, newend)


def SNR_detect(Wx, M, newbeg, newend, snr_lowerbound):
    """
    Apply the SNR detection method

    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param M: mean of noise model.
    :type M: :class:`numpy.ndarray`
    :param snr_lowerbound: noise level lower bound in percent.
    :type snr_lowerbound: float
    :return M_new: mean of noise model.
    :rtype M_new: standard deviation of noise model.
    :return S: :class:`numpy.ndarray`
    :rtype S: :class:`numpy.ndarray`
        
    """
    nlbound = snr_lowerbound*0.01
    M_max = np.max(np.abs(M))
    Wx = (Wx.T / (M+nlbound*M_max)).T
    
    # Recalculate the noise model for possible further use
    M_new = np.mean(np.abs(Wx[:,newbeg:newend]),axis=1)
    S = np.std(np.abs(Wx[:,newbeg:newend]),axis=1)
    
    return (M_new, S)


def noise_thresholding(Wx, noise_threshold, P):
    """
    Apply hard/soft thresholding to the noise (removing noise).

    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param noise_threshold: "soft" or "hard" noise thresholding.
    :type noise_threshold: str
    :param P: threshold of the noise signal.
    :type P: :class:`numpy.ndarray`
    :return Wx_new: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx_new: :class:`numpy.ndarray`

    """
    W_test = np.abs(Wx)
    if noise_threshold == "hard":        
        Wx_new = (Wx.T * (P<W_test.T)).T
    elif noise_threshold == "soft":
        Wx_new = (np.sign(Wx).T * (W_test.T-P) * (P<W_test.T)).T
        
    return Wx_new


def signal_thresholding(Wx, signal_threshold, P):
    """
    Apply hard/soft thresholding to the signal (removing signal).

    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param signal_threshold: "soft" or "hard" noise thresholding.
    :type signal_threshold: str
    :param P: threshold of the noise signal.
    :type P: :class:`numpy.ndarray`
    :return Wx_new: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx_new: :class:`numpy.ndarray`
    """
    W_test = np.abs(Wx)
    if signal_threshold == "hard":
        Wx_new = (Wx.T * (P>W_test.T)).T
    elif signal_threshold == "soft":
        Wx_new = (np.sign(Wx.T)* P).T * (P <= W_test.T).T + (Wx.T*(P > W_test.T)).T

    return Wx_new