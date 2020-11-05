# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: obspy
#     language: python
#     name: obspy
# ---

import numpy as np
import math


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
    #x = np.asarray(x)
    n = len(x) # number of samples
    
    # Padding the signal
    N = int(2**(1+round(math.log2(len(x))+np.finfo(float).eps)))
    n1 = math.floor((N-n)/2)
    n2 = n1
    if math.fmod(2*n1+n, 2) == 1:
        n2 = n1 + 1
    x = np.pad(x, (n1,n2), mode="constant")
    
    # Choosing more than this means the wavelet window becomes too short
    noct = np.log2(N)-1
    assert noct > 0 and math.fmod(noct,1) == 0,"there is a problem with noct"
    assert nv>0 and math.fmod(nv,1)==0, "nv has to be higher than 0"
    assert dt>0, "dt has to be higher than 0"
    # assert np.isnan(x) is False,"x has null values"
     
    na = int(noct*nv)
    tmp = np.arange(1, na+1, 1, dtype="int")
    aS = np.power(2**(1/nv), tmp)
    Wx = np.zeros((na, N),dtype=complex)
    #x = x(:).'
    xh = np.fft.fft(x)
    
    # for each octave
    for ai in range(na):
        a = aS[ai]
        psih = wfilth(wtype, N, a) # wavelet sampling
        xcpsi = np.fft.ifftshift(np.fft.ifft(psih*xh))
        Wx[ai, :] = xcpsi
    
    # Output a for graphing purposes, scale by dt
    aS = aS * dt
    
    Wxout = Wx[:,n1+1:n1+n+1]
    
    return (Wxout, aS)


def wfilth(wtype, N, a):
    """
    
    Outputs the FFT of the wavelet of family 'type' with parameters
    in 'opt', of length N at scale a: (psi(-t/a))^.
    
    :param wtype: wavelet type
    :type wtype: str
    :param N: number of samples to calculate
    :type N: int
    :param a: wavelet scale parameter
    :type a: float
    :return psih: wavelet sampling in frequency domain
    :rtype psih: :class:`numpy.ndarray`
    """

    
    k = np.arange(0,N,1,dtype = 'int')
    xi = np.zeros(N,dtype='float')
    xi[:int(N/2)+1] = 2 * math.pi/N * np.arange(0,N/2+1,1)
    xi[int(N/2)+1:] = 2 * math.pi/N * np.arange(-N/2+1,0,1)
    # psihfn = wfiltfn(xi, wtype);
    tmpxi = a*xi
    psih = wfiltfn(tmpxi, wtype);
    # Normalizing
    psih = psih * math.sqrt(a) / math.sqrt(2*math.pi)
    # Center around zero in the time domain
    psih = psih * np.power((-1),k);
    
    return psih


def wfiltfn(xi, wtype):
    """
    Wavelet transform function of the wavelet filter in fourier domain.
    
    :param xi: sampled time series
    :type xi: class:`numpy.ndarray`
    :wvtype: wavelet type, options are mexican hat, morlet, shannon, or hermitian.
    :type wvtype: str
    :return psihfn: mother wavelet function
    :rtype psihfn: :class:`numpy.ndarray`
    """

    if wtype == 'mhat':
        s=1
        psihfn = -np.sqrt(8) * s**(5/2) * (np.pi**(1/4)/np.sqrt(3)) * (xi**2) * np.exp((-s**2)*(xi**2/2))
    if wtype == 'morlet':
        mu = 2*np.pi
        cs = (1 + np.exp(-mu**2) - 2*np.exp(-3/4*(mu**2)))**(-1/2)
        ks = np.exp(-1/2*(mu**2))
        psihfn = cs*(np.pi**(-1/4)) * np.exp(-1/2*(mu-xi)**2) - ks*np.exp(-1/2*(xi**2))
        #need to pass an error for unknown wavelet
        
    return np.array(psihfn)


# ### Execute Programs

import math
from obspy.core import read
import matplotlib.pyplot as plt

# +
file = "/Users/chiang4/Work/NNSA/LYNM/denoiser/bc_v1.1/data/5014.YW.0.sp0011.DPZ"
#st = read(file, headonly=False, byteorder=None, checksize=False)
st = read(file)
st.plot()

# All parameters
wtype = "morlet"
nvoices = 16
#nbpblck = 1
#scale_min = 1.0
#scale_max = 200.0
#bthresh=0.0
#nnbnd = 1
#tstrn = 0.0
#tfinn = 60.0
#noisethresh = 2
#signalthresh = 0
#nsig = -2.0
#nsnr = 0
#nsnrlb = 1.0
# -

tr = st[0]
testdata = tr.data
delta = tr.stats.delta
Wx_new,as_new = cwt_fw(testdata, wtype, nvoices, delta)
# +
# Plot the wavelet
fig = plt.figure(figsize=(10,6.5))
ax1 = fig.add_subplot(2,1,1)
ax1.imshow(abs(Wx_new), aspect = 'auto') # need to fix x-axis to time
ax1.set_ylabel("log10 Scale")
ax1.set_title("Time Frequency Representation (TRF)")

ax2 = fig.add_subplot(2,1,2)
ax2.plot(st[0].times(),st[0].data,color="black",linewidth=1)
ax2.set_xlim([0,max(st[0].times())])
ax2.set_xlabel("Time [s]")
plt.show()
# -

Wx_new_2,as_new = cwt_fw(testdata, wtype, nvoices, delta)


as_new
