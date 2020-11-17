# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Ana Aguiar Moya (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea4@llnl.gov)
#
# Functions are based on the MATLAB Synchrosqueezing Toolbox by Eugene Brevdo.
# (http://www.math.princeton.edu/~ebrevdo/)

import math
import numpy as np
from scipy.integrate import quad


def _psihfn(xi,wave_type):
    return np.conj(wfiltfn(xi, wave_type))*wfiltfn(xi, wave_type)/xi


def wfilth(wave_type, N, a):
    """
    Fast fourier transform of the wavelet function.

    Outputs the FFT of a given ``wave_type`` of length N at scale a (psi(-t/a))^.
    
    :param wave_type: wavelet function.
    :type wave_type: str
    :param N: number of samples to calculate.
    :type N: int
    :param a: wavelet scale.
    :type a: float
    :return psih: wavelet sampling in frequency domain
    :rtype psih: :class:`numpy.ndarray`
    """
    k = np.arange(0, N, 1, dtype=int)
    xi = np.zeros(N, dtype=float)
    xi[:int(N/2)+1] = 2 * math.pi/N * np.arange(0,N/2+1, 1)
    xi[int(N/2)+1:] = 2 * math.pi/N * np.arange(-N/2+1, 0, 1)
    # psihfn = wfiltfn(xi, wtype)
    tmpxi = a*xi
    psih = wfiltfn(tmpxi, wave_type)
    # Normalizing
    psih = psih * math.sqrt(a) / math.sqrt(2*math.pi)
    # Center around zero in the time domain
    psih = psih * np.power((-1),k);
    
    return psih


def wfiltfn(xi, wave_type):
    """
    Wavelet transform function of the wavelet filter in fourier domain.
    
    :param xi: sampled time series.
    :type xi: class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :return psihfn: mother wavelet function.
    :rtype psihfn: :class:`numpy.ndarray`
    """
    if wave_type == "mhat":
        s=1
        psihfn = -np.sqrt(8) * s**(5/2) * (np.pi**(1/4)/np.sqrt(3)) * (xi**2) * np.exp((-s**2)*(xi**2/2))
    elif wave_type == "morlet":
        mu = 2*np.pi
        cs = (1 + np.exp(-mu**2) - 2*np.exp(-3/4*(mu**2)))**(-1/2)
        ks = np.exp(-1/2*(mu**2))
        psihfn = cs*(np.pi**(-1/4)) * np.exp(-1/2*(mu-xi)**2) - ks*np.exp(-1/2*(xi**2))
    elif wave_type == "shannon":
    	pass
    elif wave_type == "hhat":
    	pass
        
    return np.array(psihfn)


def forward_cwt(time_series, wave_type, nvoices, dt):
    """
    Continuous wavelet tranform using the wavelet function.

    :param time_series: input time series data.
    :type time_series: list or :class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :param nvoices: sampling of CWT in scale.
    :type nvoices: int
    :param dt: sampling period.
    :type dt: float
    :return Wx: wavelet transform of shape (len(scales), len(time_series))
    :rtype Wx: :class:`numpy.ndarray`
    :return scales: length vector containing the associated scales
    :rtype scales: :class:`numpy.ndarray`
    """
    time_series = np.asarray(time_series)
    n = len(time_series) # number of samples
    
    # Padding the signal
    N = int(2**(1+round(math.log2(len(time_series))+np.finfo(float).eps)))
    n1 = math.floor((N-n)/2)
    n2 = n1
    if math.fmod(2*n1+n, 2) == 1:
        n2 = n1 + 1
    time_series = np.pad(time_series, (n1,n2), mode="constant")
    
    # Choosing more than this means the wavelet window becomes too short
    noctaves = np.log2(N)-1 # number of octaves
    if noctaves <= 0 and math.fmod(noctaves,1) != 0:
        raise ValueError("noctaves has to be higher than zero.")
    if nvoices <= 0 and math.fmod(nvoices,1) != 0:
        raise ValueError("nvoices has to be higher than zero.")
    if dt <= 0:
        raise ValueError("dt has to be higher than zero.")
    if np.any(np.isnan(time_series)):
        raise ValueError("time_series has null values.")
     
    num_scales = int(noctaves*nvoices)
    mi = np.arange(1, num_scales+1, 1, dtype=int)
    scales = np.power(2**(1/nvoices), mi)
    Wx = np.zeros((num_scales, N), dtype=complex)
    time_series_fft = np.fft.fft(time_series)
    
    # for each octave (this part should be parallelized?)
    for i in range(num_scales):
        psih = wfilth(wave_type, N, scales[i]) # wavelet sampling
        Wx[i, :] = np.fft.ifftshift(np.fft.ifft(psih*time_series_fft))
        
    # Output scales for graphing purposes, scale by dt
    scales = scales * dt
    
    Wxout = Wx[:,n1+1:n1+n+1]
    
    return (Wxout, scales)


def inverse_cwt(Wx, wave_type, nvoices):
    """
    Inverse continuous wavelet tranform/

    Reconstructs the original signal from the wavelet transform.
    
    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :param nvoices: sampling of CWT in scale.
    :type nvoices: int
    :return time_series: time series data.
    :rtype time_series: :class:`numpy.ndarray`
    """    
    [num_scales, n] = np.shape(Wx)
    #Padding the siggnal
    N = int(2**(1+round(math.log2(n)+np.finfo(float).eps)))
    n1 = math.floor((N-n)/2)
    # n2 = n1
    # if math.fmod(2*n1+n,2)==1:
    #     n2 = n1 + 1
    Wx_padded = np.zeros((num_scales, N), dtype=complex)
    Wx_padded[:,n1+1:n1+n+1] = Wx
    
    noctaves = np.log2(N) - 1
    if math.fmod(noctaves,1) != 0:
        raise ValueError("noctaves has to be nonzero.")
    if nvoices <= 0 and math.fmod(nvoices,1) != 0:
        raise ValueError("nvoices has to be higher than zero.")

    mi = np.arange(1, num_scales+1, 1, dtype=int)
    scales = np.power(2**(1/nvoices), mi)
    
    # For type 'shannon', have to include the following
    if wave_type == "shannon":
        Cpsi = log(2)
    else:
        [Cpsi, Cpsi_err] = quad(_psihfn, 0, np.inf, args=(wave_type) )

    # Normalize
    Cpsi = Cpsi / (4*np.pi)
    x = np.zeros((1, N))
    # for each octave
    for ai in range(num_scales):
        Wxa = Wx_padded[ai, :]
        psih = wfilth(wave_type, N, scales[ai])
        # Convolution theorem
        Wxah = np.fft.fft(Wxa)
        xah = Wxah * psih
        xa = np.fft.ifftshift(np.fft.ifft(xah))
        x = x + xa/scales[ai]
        
    # Take real part and normalize by log_e(a)/Cpsi
    x = np.log(2**(1/nvoices))/Cpsi * np.real(x)

    # Keep the unpadded part
    time_series = x[0][n1+1:n1+n+1]
    
    return time_series