# py_cwt_fw.py
# 
# Code modified from the BCseis MatLab version from Langston et al. 
# at U. ofMenphis http://www.ceri.memphis.edu/people/clangstn/index.html
#
# ACAM 09/24/2020
#
# Forward continuous wavelet transform, discretized, as described
# in Mallat, S., Wavelet Tour of Signal Processing 3rd ed.Sec. 4.3.3.
#
# [INPUTS]
# x: input signal vector.
# wtype: wavelet type, string
# nv: number of voices
# dt: sampling period
# opt: options structure
#
# [OUTPUTS]
# Wx: [na x n] size matrix (rows = scales, cols = times)
# as: na length vector containing the associated scales
#---------------------------------------------------------------------------------
# Modified after a wavelet transform matlab codes by Eugene Brevdo
# mofifications implemented in Matlab by Chuck Langston @ Memphis
#---------------------------------------------------------------------------------

import numpy as np
import math as m
# import pywt

def cwt_fw(x, wtype, nv, dt):
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
        psih = wfilth(wtype, N, a)
        xcpsi = np.fft.ifftshift(np.fft.ifft(np.multiply(psih, xh)))
        Wx[ai, :] = xcpsi
    
    # Output a for graphing purposes, scale by dt
    aS = aS * dt
    
    Wxout = Wx[:,n1+1:n1+n+1]
    return [Wxout, aS]

def wfilth(wtype, N, a):
    #  Outputs the FFT of the wavelet of family 'type' with parameters
    #  in 'opt', of length N at scale a: (psi(-t/a))^.
    #  [Inputs]
    #  wtype: wavelet type
    #  N: number of samples to calculate
    #  a: wavelet scale parameter
    #  [Outputs]
    #  psih: wavelet sampling in frequency domain
    
    k = np.arange(0,N,1,dtype = 'int')
    xi = np.zeros(N,dtype='float')
    xi[:int(N/2)+1] = 2 * np.pi/N * np.arange(0,N/2+1,1)
    xi[int(N/2)+1:] = 2 * np.pi/N * np.arange(-N/2+1,0,1)
    # psihfn = wfiltfn(xi, wtype);
    tmpxi = a*xi
    psih = wfiltfn(tmpxi, wtype);
    # Normalizing
    psih = psih * np.sqrt(a) / np.sqrt(2*np.pi)
    # Center around zero in the time domain
    psih = psih * np.power((-1),k);
    
    return psih


def wfiltfn(xi, wtype):
    # Wavelet transform function of the wavelet filter in question,
    # fourier domain.
    # [Input]
    # xi: 
    # wtype: string (see below)
    # [Output]
    # psihfn: mother wavelet function ( mexican hat, morlet, shannon, or hermitian)

    if wtype == 'mhat':
        s=1
        psihfn = -np.sqrt(8) * s**(5/2) * (np.pi**(1/4)/np.sqrt(3)) * (xi**2) * np.exp((-s**2)*(xi**2/2))
        # psihfn = list(map(lambda w: -m.sqrt(8)*m.pow(s,5/2)*m.pow(m.pi,1/4)/m.sqrt(3)*m.pow(w,2)*m.exp(m.pow(-s,2)* m.pow(w,2/2)), xi))
    if wtype == 'morlet':
        mu = 2*np.pi
        cs = (1 + np.exp(-mu**2) - 2*np.exp(-3/4*(mu**2)))**(-1/2)
        ks = np.exp(-1/2*(mu**2))
        psihfn = cs*(np.pi**(-1/4)) * np.exp(-1/2*(mu-xi)**2) - ks*np.exp(-1/2*(xi**2))
        
        # cs = m.pow(1 + m.exp(m.pow(-mu,2)) - 2*m.exp(-3/4*m.pow(mu,2)), -1/2)
        # ks = m.exp((-1/2) * m.pow(mu,2))
        # psihfn = list(map(lambda w: cs*m.pow(m.pi,-1/4)*m.exp(-1/2*m.pow((mu-w),2)) - ks*m.exp(-1/2*m.pow(w,2)), xi))
        
        #need to pass an error for unknown wavelet
        
    return np.array(psihfn)
        
        
# %%       #################### Main Program ########################       
    
import obspy as ob
import os
import matplotlib.pyplot as plt

mainpath = '/Users/aguiarmoya1/Research/Denoising/bc_v1.1/data/'
os.chdir(mainpath)

file1 = ob.read('5014.YW.0.sp0011.DPZ', headonly=False, byteorder=None, checksize=False)
file1.plot()
tr =file1[0]
testdata = tr.data
delta = tr.stats.delta

# All parameters
wtype = 'morlet'
nvoices=16
# nv=nvoices
# x=testdata
# nbpblck = 1
# scale_min = 1.0
# scale_max = 200.0
# bthresh=0.0
# nnbnd = 1
# tstrn = 0.0
# tfinn = 60.0
# noisethresh = 2
# signalthresh = 0
# nsig = -2.0
# nsnr = 0
# nsnrlb = 1.0

[Wx_new,as_new] = cwt_fw(testdata, wtype, nvoices, delta)

# %% plot the wavelet
Wxout=Wx_new
#plt.imshow(abs(Wxout), extent=[0, 45000, 1, 240], aspect = 'auto')
plt.imshow(abs(Wxout), aspect = 'auto')