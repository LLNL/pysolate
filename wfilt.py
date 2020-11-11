# wfilt.py

import numpy as np


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