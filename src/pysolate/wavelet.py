# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# authors:
#        Ana Aguiar (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea@llnl.gov)
#
# Wavelet transform functions are based on the MATLAB
# Synchrosqueezing Toolbox by Eugene Brevdo.
# (http://www.math.princeton.edu/~ebrevdo/)
"""
Routines for handling the wavelet transform
"""


import math
import numpy as np
import fnmatch
from scipy.integrate import quad
from obspy.core.trace import Stats
from obspy.core.util import AttribDict


def _psihfn(xi,wave_type):
    return np.conj(wfiltfn(xi, wave_type))*wfiltfn(xi, wave_type)/xi


def wfilth(wave_type, N, a):
    """
    Fast fourier transform of the wavelet function

    Outputs the FFT of a given ``wave_type`` of length N at scale a.
    
    :param wave_type: wavelet function.
    :type wave_type: str
    :param N: number of samples to calculate.
    :type N: int
    :param a: wavelet scale.
    :type a: float
    :return: wavelet sampling in frequency domain
    :rtype: :class:`numpy.ndarray`
    """
    k = np.arange(0, N, 1, dtype=int)
    xi = np.zeros(N, dtype=float)
    xi[:int(N/2)+1] = 2 * math.pi/N * np.arange(0,N/2+1, 1)
    xi[int(N/2)+1:] = 2 * math.pi/N * np.arange(-N/2+1, 0, 1)
    tmpxi = a*xi
    psih = wfiltfn(tmpxi, wave_type)
    # Normalizing
    psih = psih * math.sqrt(a) / math.sqrt(2*math.pi)
    # Center around zero in the time domain
    psih = psih * np.power((-1),k)
    
    return psih


def wfiltfn(xi, wave_type):
    """
    Wavelet transform function of the wavelet filter in fourier domain
    
    :param xi: sampled time series.
    :type xi: :class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :return: mother wavelet function.
    :rtype: :class:`numpy.ndarray`
    """
    if wave_type == "mhat":
        s = 1
        psihfn = -np.sqrt(8) * s**(5/2) * (np.pi**(1/4)/np.sqrt(3)) * (xi**2) * np.exp((-s**2)*(xi**2/2))
    elif wave_type == "morlet":
        mu = 2*np.pi
        cs = (1 + np.exp(-mu**2) - 2*np.exp(-3/4*(mu**2)))**(-1/2)
        ks = np.exp(-1/2*(mu**2))
        psihfn = cs*(np.pi**(-1/4)) * np.exp(-1/2*(mu-xi)**2) - ks*np.exp(-1/2*(xi**2))
    elif wave_type == "shannon":
        pass
    elif wave_type == "hhat":
        psihfn = 2/np.sqrt(5) * np.pi**(-1/4) * xi * (1+xi) * np.exp(-1/2*xi**2)
        
    return np.array(psihfn)


def cwt(trace, wave_type, nvoices):
    """
    Continuous wavelet transform using the wavelet function

    :param trace: input time series data.
    :type time_series: list or :class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :param nvoices: sampling of CWT in scale.
    :type nvoices: int
    :returns: the wavelet transform of shape (scales, time_series),
        and the length vector containing the associated scales.
    :rtype: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
    """
    time_series = trace.data
    dt = trace.stats.delta
    n = trace.stats.npts
    
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
    Inverse continuous wavelet tranform

    Reconstructs the original signal from the wavelet transform.
    
    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param wave_type: wavelet function.
    :type wave_type: str
    :param nvoices: sampling of CWT in scale.
    :type nvoices: int
    :return: time series data.
    :rtype: :class:`numpy.ndarray`
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


def blockbandpass(Wx, scales, scale_min, scale_max, block_threshold):
    """
    Apply a band reject filter to modify the wavelet coefficients
    over a scale bandpass
    
    :param Wx: wavelet transform of shape (len(scales), len(time_series))
    :type Wx: :class:`numpy.ndarray`
    :param scale_min: minimum time scale for bandpass blocking.
    :type scale_min: float
    :param scale_max: maximum time scale for bandpass blocking.
    :type scale_max: float
    :param block_threshhold: percent amplitude adjustment to the wavelet coefficients within
        ``scale_min`` and ``scale_max``.
    :type block_threshold: float
    :return: modified wavelet transform of shape (len(scales), len(time_series)).
    :rtype: :class:`numpy.ndarray`
    """    
    na, n = np.shape(Wx)
    thresh = block_threshold*0.01
      
    a = np.ones((1,na))
    a = a*[ (scales <= scale_min) | (scales >= scale_max) ]
    
    for k in range(na):
        if a[0,k] == 0:
            a[0,k] = thresh
    
    return Wx*a.T


class Wavelet(object):
    """
    One dimensional continues wavelet transform of a time series

    Main class for a single wavelet transform including
    the station headers, wavelet coefficients and scales.
    Noise model calculated from the function
    :func:`~pysolate.threshold.noise_model` is passed through
    additional ``kwargs``.

    :param coefs: Continuous wavelet transform of a time series,
        the first axis corresponds to the scales and the second axis
        corresponds to the length of the time series.
    :type coefs: :class:`numpy.ndarray`
    :param scales: Wavelet scales
    :type scales: :class:`numpy.ndarray`
    :param headers: header information of the data.
    :type headers: dict, :class:`~obspy.core.trace.Stats`
    :param M: mean of noise model.
    :type M: :class:`numpy.ndarray`
    :param S: standard deviation of noise model.
    :type S: :class:`numpy.ndarray`
    :param P: threshold of the noise signal.
    :type P: :class:`numpy.ndarray`

    .. rubric:: Attributes

    ``stats`` : :class:`~obspy.core.trace.Stats`
        header of the wavelet transform, including station info.
    ``scales`` : :class:`numpy.ndarray`
        wavelet scales.
    ``coefs`` : :class:`numpy.ndarray`
        wavelet coefficients.
    ``noise_model`` : :class:`~obspy.core.util.attribdict.AttribDict`
        noise model
    """
    def __init__(self, coefs=None, scales=None, headers=None, **kwargs):
        if headers is None:
            headers = {}
        self.stats = Stats(headers)
        self.scales = scales
        self.coefs = coefs
        self.noise_model = AttribDict(**kwargs)

    def get_id(self):
        """
        Returns the station ID

        :return: station ID containing network, station, location and
            channel codes.
        :rtype: str
        """
        out = "%(network)s.%(station)s.%(location)s.%(channel)s"
        return out%(self.stats)

    def __str__(self):
        """
        Return better readable string representation of Parameter object.
        """
        out = " | %(starttime)s - %(endtime)s | "

        attrs = ', '.join(
            [key for key in ["scales", "coefs"] if key is not None],
        )
        if len(self.noise_model) > 0:
            attrs = ', '.join([attrs, "noise_model"])

        return self.get_id() + out%(self.stats) + attrs

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class WaveletCollection(object):
    """
    A collection of wavelet objects

    Main class that contains wavelet transforms for multiple
    time series.

    :param wavelets: wavelet transform(s).
    :type wavelets: :class:`~pysolate.wavelet.Wavelet`, list
    """
    def __init__(self, wavelets=None):
        self.wavelets = []
        if isinstance(wavelets, Wavelet):
            wavelets = [wavelets]
        if wavelets:
            self.wavelets.extend(wavelets)

    def __len__(self):
        return len(self.wavelets)

    count = __len__

    def __setitem__(self, index, wavelet):
        """
        __setitem__ method.
        """
        self.wavelets.__setitem__(index, wavelet)

    def __getitem__(self, index):
        """
        __getitem__ method.

        :return: Wavelet objects
        """
        if isinstance(index, slice):
            return self.__class__(wavelets=self.wavelets.__getitem__(index))
        else:
            return self.wavelets.__getitem__(index)

    def __delitem__(self, index):
        """
        Passes on the __delitem__ method to the underlying list of wavelets.
        """
        return self.wavelets.__delitem__(index)

    def __getslice__(self, i, j, k=1):
        """
        __getslice__ method.

        :return: WaveletCollection object
        """
        return self.__class__(wavelets=self.wavelets[max(0, i):max(0, j):k])

    def select(self, network=None, station=None, location=None, channel=None,
               component=None):
        """
        Query wavelets

        Select wavelet transforms that matches the given station criteria.

        :param tag: processed data to plot.
        :type tag: str
        :param network: network code.
        :type network: str
        :param station: station code.
        :type station: str
        :param location: location code.
        :type location: str
        :param channel: channel code.
        :type channel: str
        :param component: component code.
        :type component: str
        """
        # Adapted from ObsPy Stream select method
        wavelets = []
        if component is not None and channel is not None:
            component = component.upper()
            channel = channel.upper()
            if (channel[-1:] not in "?*" and component not in "?*" and
                    component != channel[-1:]):
                msg = "Selection criteria for channel and component are " + \
                      "mutually exclusive!"
                raise ValueError(msg)
        for _i in self.wavelets:
            if network is not None:
                if not fnmatch.fnmatch(
                        _i.stats.network.upper(), network.upper()
                ):
                    continue
            if station is not None:
                if not fnmatch.fnmatch(
                        _i.stats.station.upper(), station.upper()
                ):
                    continue
            if location is not None:
                if not fnmatch.fnmatch(
                        _i.stats.location.upper(), location.upper()
                ):
                    continue
            if channel is not None:
                if not fnmatch.fnmatch(
                        _i.stats.channel.upper(), channel.upper()
                ):
                    continue
            if component is not None:
                if not fnmatch.fnmatch(
                        _i.stats.component.upper(), component.upper()
                ):
                    continue
            wavelets.append(_i)

        return self.__class__(wavelets=wavelets)

    def __str__(self):
        pretty_string = "\n".join([_i.__str__() for _i in self])

        return pretty_string

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
