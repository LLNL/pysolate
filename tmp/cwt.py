# cwt.py
# 
# Code modified from the BCseis MatLab version from Langston et al. 
# at U. ofMenphis http://www.ceri.memphis.edu/people/clangstn/index.html
#
# ACAM 09/24/2020
#
# Forward continuous wavelet transform, discretized, as described
# in Mallat, S., Wavelet Tour of Signal Processing 3rd ed.Sec. 4.3.3.
#
#---------------------------------------------------------------------------------
# Modified after a wavelet transform matlab codes by Eugene Brevdo
# mofifications implemented in Matlab by Chuck Langston @ Memphis
#---------------------------------------------------------------------------------

import numpy as np
import math as m
import wfilt as wft
from scipy.integrate import quad

def cwt_fw(x, wtype, nv, dt):
    
    """
    Forward continuous wavelet tranform based
    
    Mallat, S., Wavelet Tour of Signal Processing 3rd ed.Sec. 4.3.3.
    
    :param x: time series data
    :type x: :class:`numpy.ndarray`
    :param wtype: wavelet filter type, options are
    :type wtype: str
    :param nv: number of voices
    :type nv: int
    :param dt: sampling period
    :type dt: float
    :return Wx: (na,n) size matrix, rows=scales and cols=times
    :rtype Wx: :class:`numpy.ndarray`
    :return aS: na lenght vector containing the associated scales
    :rtype aS: :class:`numpy.ndarray`
    """
    x = np.asarray(x)
    n = len(x)
    #Padding the siggnal
    N = int(2**(1+round(m.log2(len(x))+np.finfo(float).eps)))
    n1 = m.floor((N-n)/2)
    n2 = n1
    if m.fmod(2*n1+n,2)==1:
        n2 = n1 + 1
    x = np.pad(x, (n1,n2), mode='constant')
    # Choosing more than this means the wavelet window becomes too short
    noct = np.log2(N)-1
    assert noct > 0 and m.fmod(noct,1) == 0,"there is a problem with noct"
    assert nv>0 and m.fmod(nv,1)==0, "nv has to be higher than 0"
    assert dt>0, "dt has to be higher than 0"
    # assert np.isnan(x) is False,"x has null values"
     
    na = int(noct*nv)
    tmp = np.arange(1, na+1, 1, dtype = 'int')
    aS = np.power(2**(1/nv), tmp)
    Wx = np.zeros((na, N),dtype=complex)
    #x = x(:).'
    xh = np.fft.fft(x)
    # for each octave
    for ai in range(na):
        a = aS[ai]
        psih = wft.wfilth(wtype, N, a)
        xcpsi = np.fft.ifftshift(np.fft.ifft(np.multiply(psih, xh)))
        Wx[ai, :] = xcpsi
    
    # Output a for graphing purposes, scale by dt
    aS = aS * dt
    
    Wxout = Wx[:,n1+1:n1+n+1]
    return [Wxout, aS]


def cwt_iw(Wx, wtype, nv):
    
    """
    Inverse continuous wavelet tranform based
    
    Mallat, S., Wavelet Tour of Signal Processing 3rd ed.Sec. 4.3.3.
    
    :param Wx: (na,n) size matrix, rows=scales and cols=times
    : type Wx: :class:`numpy.ndarray`
    :param wtype: wavelet filter type, options are
    : type wtype: str
    :param nv: number of voices
    : type nv: int
    :return xi: time series data
    : rtype xi: :class:`numpy.ndarray`

    """    
    [na, n] = np.shape(Wx)
    #Padding the siggnal
    N = int(2**(1+round(m.log2(n)+np.finfo(float).eps)))
    n1 = m.floor((N-n)/2)
    # n2 = n1
    # if m.fmod(2*n1+n,2)==1:
    #     n2 = n1 + 1
    Wxp = np.zeros((na, N),dtype=complex)
    Wxp[:,n1+1:n1+n+1] = Wx
    
    Wx = Wxp
    
    noct = np.log2(N)-1
    tmp = np.arange(1, na+1, 1, dtype = 'int')
    aS = np.power(2**(1/nv), tmp)
    assert m.fmod(noct,1) == 0,"there is a problem with noct"
    assert nv>0 and m.fmod(nv,1)==0, "nv has to be higher than 0"
    
    # For type 'shannon', have to include the following
    # Cpsi = log(2);
    # otherwise just the next lines
    
    def psihfn(xi,wtype):
        return np.conj(wft.wfiltfn(xi, wtype))*wft.wfiltfn(xi, wtype)/xi
    
    [Cpsi, Cpsi_err] = quad(psihfn, 0, np.inf, args=(wtype) )
    # Normalize
    Cpsi = Cpsi / (4*np.pi)
    x = np.zeros((1, N))
    # for each octave
    for ai in range(na):
        a = aS[ai]
        Wxa = Wx[ai, :]
        psih = wft.wfilth(wtype, N, a)
        # Convolution theorem
        Wxah = np.fft.fft(Wxa)
        xah = Wxah * psih
        xa = np.fft.ifftshift(np.fft.ifft(xah))
        x = x + xa/a
        
    # Take real part and normalize by log_e(a)/Cpsi
    x = np.log(2**(1/nv))/Cpsi * np.real(x)
    # Keep the unpadded part
    xi = x[0][n1+1:n1+n+1]
    
    return xi