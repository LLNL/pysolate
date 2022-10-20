# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# authors:
#        Ana Aguiar (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea@llnl.gov)
"""
Non-linear thresholding operations using the continuous wavelet transform
"""
# Most functions are adapted from the BCseis ver1.1 MATLAB package
# by Charles A. Langston and S. Mostafa Mousavi.


import numpy as np
from scipy.interpolate import interp1d


def ecdf(x):
    """
    Empirical cumulative probability distribution

    Function returns the empirical cumulative distribution function of
    the input array.

    :param x: a sample
    :type x: :class:`numpy.ndarray`
    :return: a 2-D array where the first and second axes correspond to
        the empirical cumulative distribution function evaluated
        at the sorted sample points.
    :rtype: :class:`numpy.ndarray`
    """
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    
    return np.array([ys,xs])


def SNR_detect(Wx, M, newbeg, newend, snr_lowerbound):
    """
    Apply SNR detection to CWT

    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param M: mean of noise model.
    :type M: :class:`numpy.ndarray`
    :param snr_lowerbound: noise level lower bound in percent.
    :type snr_lowerbound: float
    :return: the updated mean and standard deviation of the noise model.
    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    nlbound = snr_lowerbound * 0.01
    M_max = np.max(np.abs(M))
    Wx = (Wx.T / (M + nlbound * M_max)).T

    # Recalculate the noise model for possible further use
    M = np.mean(np.abs(Wx[:, newbeg:newend]), axis=1)  # point to re-calculated mean
    S = np.std(np.abs(Wx[:, newbeg:newend]), axis=1)

    return (M, S)


def noise_model(Wx, delta, noise_starttime, noise_endtime, nsigma_method, nlbound, detection=False):
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
    :param detection: If ``True`` it will be applied before hard thresholding.
        Default is ``False``.
    :type detection: bool
    :return: the mean, standard deviation of the noise model, and
        threshold of the noise signal.
    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    # Get the time window
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
            
    # SNR detection
    if detection:
        M, S = SNR_detect(Wx, M, newbeg, newend, nlbound)

    #return (M, S, P, newbeg, newend)
    return (M, S, P)


def noise_thresholding(Wx, noise_threshold, P):
    """
    Apply hard/soft thresholding to the noise (removing noise).

    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param noise_threshold: "soft" or "hard" noise thresholding.
    :type noise_threshold: str
    :param P: threshold of the noise signal.
    :type P: :class:`numpy.ndarray`
    :return Wx_new: the new wavelet transform.
    :rtype Wx_new: :class:`numpy.ndarray`

    """
    W_test = np.abs(Wx)
    if noise_threshold == "hard":        
        Wx_new = (Wx.T * (P<W_test.T)).T
    elif noise_threshold == "soft":
        Wx_new = (Wx/W_test) * ((W_test.T-P) * (P<W_test.T)).T
        
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
    :return Wx_new: the new wavelet transform.
    :rtype Wx_new: :class:`numpy.ndarray`
    """
    W_test = np.abs(Wx)
    if signal_threshold == "hard":
        Wx_new = (Wx.T * (P>W_test.T)).T
    elif signal_threshold == "soft":
        Wx_new = ((Wx/W_test).T * P).T * (P <= W_test.T).T + (Wx.T*(P > W_test.T)).T

    return Wx_new
