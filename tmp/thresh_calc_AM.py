# thres_calc.py
# 
# Code modified from the BCseis MatLab version from Langston et al. 
# at U. ofMenphis http://www.ceri.memphis.edu/people/clangstn/index.html
#
# ACAM 12/10/2020
#
# Computes filtering and thresholds for noise or data
#

import numpy as np
#import math as m
from scipy.interpolate import interp1d

def waveletSize(Wx):
    [na, n] = np.shape(Wx)
    return na, n

def blockbandpass(Wx, aS, scale_min, scale_max, block_threshold):
    
    """
    Applies block bandpass is wanted
    
    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param scale_min: minimum time scale for bandpass blocking. Default is ``0``.
    : type scale_min: float
    :param scale_max: maximum time scale for bandpass blocking. Default is ``200``.
    : type scale_max: float
    :param block_threshhold: percent amplitude adjustment to the wavelet coefficients within
        ``scale_min`` and ``scale_max``. For example a threshold of 5% means the wavelet cofficients
        in the band will be multipled by 0.05. Default is ``0``.
    :type block_threshold: float
    
    :return Wx_new: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx_new: :class:`numpy.ndarray`

    """    
    #[na, n] = np.shape(Wx)    
    [na, n] = waveletSize(Wx)
    thresh=block_threshold*0.01
    
    #  save old CWT
    # Wx_old = Wx
    # as_old = aS    
    a=np.ones((1,na))
    a=a*[ (aS <= scale_min) | (aS >= scale_max)]
    #a=a*aS[ (aS <= scale_min) | (aS >= scale_max)]
    # a=a.*(as_old <= scale_min | as_old >= scale_max); ##NEED TO CHANGE TO PYTHON
    
    for k in range(na):
        if a[0,k] == 0:
            a[0,k] = thresh
            
    Wx_new = Wx*np.transpose(a)
    
    return Wx_new

def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return [ys, xs]

def noiseModel(Wx, delta, noise_model, noise_starttime, noise_endtime, nsigma_method, nlbound):
        
    """
    Calculate noise model and threshold function, if needed
    
    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param noise_model: flag to compute the noise model, default is ``True``.
    : type noise_model: bool
    :param noise_starttime: noise start time, default is ``0``.
    : type noise_starttime: float
    :param noise_endtime: noise end time, default is ``60``.
    : type noise_endtime: float

    :param nsigma_method: method to determine the number of standard deviations for block thresholding.
        ``"donoho"`` for Donoho's Threshold criterion, ``"ECDF"`` for empirical cumulative probability
        distribution method. The default method ``"ECDF"`` is recommended.
    :param nlbound: noise level bund percent.
    : type nlbound: float
    
    :return M:
    : rtype M:
    :return S:
    : rtype S:
    :return P:
     :rtype P:
    
    """

    # Get the time window
    newbeg = int(np.round(noise_starttime/delta) +1)
    newend = int(np.round(noise_endtime/delta) +1)

    if nsigma_method == "donoho":
        # compute Donoho's Threshold Criterion
        nsigma_method = np.sqrt(2*np.log10(newend-newbeg +1))
        
#    if nsigma_method > 0:
        # Assume Gaussian statistics
        # Calculate the mean and standard deviations at each scale in the
        # time-scale window
        M = np.mean(np.abs(Wx[:,newbeg:newend]))
        S = np.std(np.abs(Wx[:,newbeg:newend]))
        P = M + nsigma_method * S;
        
    elif nsigma_method == "ECDF":
        # Estimate empirical CDF for each scale, calculate 99% confidence 
        # level for the noise threshold
        [nrow, ncol] = waveletSize(Wx) 
        #conf = 0.99
        conf = 1.0 - nlbound*0.01
        n_noise=newend-newbeg+1
        
        P=np.empty(nrow)
        for k in range(nrow):
            W=[]
            W[1:n_noise] = np.abs(Wx[k,newbeg:newend])
            [f,x] = ecdf(W)
            fnew = interp1d(f, x, kind='linear')
            #fnew = interp1d(x,f, kind='linear', fill_value="extrapolate")
            P[k] = fnew(conf)      
            #P[k] = np.interp(conf,x,f)
            
        M = np.mean(np.abs(Wx[:,newbeg:newend])) # need to define this? this is probably incorrect
        S = np.std(np.abs(Wx[:,newbeg:newend]))  # This too
        
    else:
        M = np.mean(np.abs(Wx[:,newbeg:newend]))
        S = np.std(np.abs(Wx[:,newbeg:newend]))
        P = M + nsigma_method * S;
            
    return [M, S, P, newbeg, newend]


def SNRdetect(Wx, M, newbeg, newend, snr_lowerbound):
    """
    Apply the SNR detection method if wanted

    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param M: 
    : type M:    
    :param snr_lowerbound: precent lower bound for SNR detection. Default is ``1.0``.
    :type snr_lowerbound: float

    :return M_new:
    : rtype M_new:
    :return S:
    : rtype S:
        
    """

    nlbound=snr_lowerbound*0.01
    M_max=np.max(np.abs(M))
    Wx=Wx/(M+nlbound*M_max)
    
    # recalculate the noise model for possible further use
    M_new = np.mean(np.abs(Wx[:,newbeg:newend]))
    S = np.std(np.abs(Wx[:,newbeg:newend]))
    
    return [M_new, S]

def noiseThresh(Wx, noise_threshold, P):
    """
    Apply hard/soft thresholding to the noise, if wanted (removing noise)

    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param noise_threshold: type of noise thresholding to be appied, the options are ``"none"`` for
        no non-linear thresholding, ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding.
        Default is ``"soft"``.
    :type noise_threshold: str
    :param P: 
    : type P:
        
    :return Wx_new: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx_new: :class:`numpy.ndarray`

    """
    if noise_threshold == "hard":        
        W_test=np.abs(Wx)
        Wx_new=Wx*np.transpose(P < np.transpose(W_test))
    
    if noise_threshold == "soft":
        W_test=np.abs(Wx)
        Wx_new=np.sign(Wx)*np.transpose(np.transpose(W_test) - P)*np.transpose(P < np.transpose(W_test))
        
    return Wx_new

def signalThresh(Wx, signal_threshold, P):
    """
    Apply hard/soft thresholding to the signal, if wanted (removing signal)

    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param signal_threshold: type of signal thresholding to be appied, the options are ``"none"`` for
        no non-linear thresholding, ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding.
        Default is ``"none"``.
    :type signal_threshold: str
    
    :return Wx_new: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx_new: :class:`numpy.ndarray`

    """
    #P = np.transpose(P)
    if signal_threshold == "hard":
        W_test=abs(Wx)
        Wx_new=Wx*np.transpose(P > np.transpose(W_test))
        
    if signal_threshold == "soft":
        W_test=abs(Wx)
        Wx_new = np.transpose(np.sign(np.transpose(Wx))* P) * np.transpose(P <= np.transpose(W_test)) + np.transpose(np.transpose(Wx)*(P > np.transpose(W_test)))

    return Wx_new
        
        
        
        
        
        