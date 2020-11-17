# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Ana Aguiar Moya (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea4@llnl.gov)

import warnings

import numpy as np
from copy import deepcopy

from . import wavelet


class Parameter(object):
    """
    A container for parameters required to set up and run the denoiser.
    
    The Parameter class determines the appropriate processes to be applied
    to the seismograms.
    
    :param wave_type: wavelet filter type, options are ``"morlet"``,``"shannon"``,
        ``"mhat"``,``"hhat"``. Default is ``"morlet"``.
    :type wave_type: str
    :param nvoices: number of voices refers to the sampling of CWT in scale,
        higher number of voices give finer resolution. Default is ``16``.
    :type nvoices: int
    :param bandpass_blocking: Default value ``True`` will apply a band rejection filter where
        wavelet coefficients are modifed over a scale bandpass.
    :type bandpass_blocking: bool
    :param scale_min: minimum time scale for bandpass blocking. Default is ``0``.
    :type scale_min: float
    :param scale_max: maximum time scale for bandpass blocking. Default is ``200``.
    :type scale_max: float
    :param block_threshhold: percent amplitude adjustment to the wavelet coefficients within
        ``scale_min`` and ``scale_max``. For example a threshold of 5% means the wavelet cofficients
        in the band will be multipled by 0.05. Default is ``0``.
    :type block_threshold: float
    :param noise_model: flag to compute the noise model, default is ``True``.
    :type noise_model: bool
    :param noise_starttime: noise start time, default is ``0``.
    :type noise_starttime: float
    :param noise_endtime: noise end time, default is ``60``.
    :type noise_endtime: float
    :param noise_threshold: type of noise thresholding to be appied, the options are ``"none"`` for
        no non-linear thresholding, ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding.
        Default is ``"soft"``.
    :type noise_threshold: str
    :param signal_threshold: type of signal thresholding to be appied, the options are ``"none"`` for
        no non-linear thresholding, ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding.
        Default is ``"none"``.
    :type signal_threshold: str
    :param nsigma_method: method to determine the number of standard deviations for block thresholding.
        ``"donoho"`` for Donoho's Threshold criterion, ``"ECDF"`` for empirical cumulative probability
        distribution method. The default method ``"ECDF"`` is recommended.
    :type nsgima_method: str
    :param snr_detection: Flag to apply the SNR detection method, deafult is ``False``. If ``True`` it
        will be applied before hard thresholding.
    :type snr_detection: bool
    :param snr_lowerbound: precent lower bound for SNR detection. Default is ``1.0``.
    :type snr_lowerbound: float
    
    """
    # Default values
    _defaults = dict(
        wave_type = "morlet",
        nvoices = 16,
        bandpass_blocking = True,
        scale_min = 1.0,
        scale_max = 200.0,
        block_threshhold = 0.0,
        noise_model = True,
        noise_starttime = 0.0,
        noise_endtime = 60.0,
        noise_threshold = "soft",
        signal_threshold = "none",
        nsigma_method = "ECDF", # method to compute nsigma
        snr_detection = False,
        snr_lowerbound = 1.0,
    )
    # types
    _types = dict(
        wave_type = str,
        nvoices = int,
        bandpass_blocking = bool,
        scale_min = float,
        scale_max = float,
        block_threshhold = float,
        noise_model = bool,
        noise_starttime = float,
        noise_endtime = float,
        noise_threshold = str,
        signal_threshold = str,
        nsigma_method = str,
        snr_detection = bool,
        snr_lowerbound = float,
    )
    
    def __init__(self, *args, **kwargs):
        # Set to default values
        self.__dict__.update(self._defaults)
        # Overwrite default
        self.update(dict(*args, **kwargs))
             
    def update(self, adict={}):
        for (key, value) in adict.items():
            self.__setitem__(key, value)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        # type check
        if key in self._types and not isinstance(value,self._types[key]):
            value = self._types[key](value)
                
        # check methods
        if key in "wave_type":
            if value not in ("morlet","shannon","mhat","hhat"):
                msg = "'%s' wavelet type is not supported."%value
                raise ValueError(msg)
        if key == "noise_threshold":
            if value not in ("none","hard","soft"):
                msg = "'%s' thresholding is not supported."%value
                raise ValueError(msg)
        if key == "signal_threshold":
            if value not in ("none","hard","soft"):
                msg = "'%s' thresholding is not supported."%value
                raise ValueError(msg)
        if key == "nsigma_method":
            if value not in ("donoho","ECDF"):
                msg = "'%s' noise reduction method is not supported."%value
                raise ValueError(msg)
        
        self.__dict__[key] = value
        #if isinstance(value, dict):
        #    super().__setitem__(key, Struct(value))
        #else:
        #    super().__setitem__(key, value)
    
    def __delitem__(self, key):
        del self.__dict__[key]

    __setattr__ = __setitem__
    __delattr__ = __delitem__
    
    def __str__(self):
        """
        Return better readable string representation of Parameter object.
        """
        min_key_length = 12
        keys = self.__dict__.keys()
        # determine longest key name for alignment
        try:
            i = max(max([len(k) for k in keys]), min_key_length)
        except ValueError:
            # no keys
            return ""
        f = "{0:>%d}: {1}\n"%i
        pretty_string = "".join([
            f.format(key, str(getattr(self, key))) for key in keys
        ])
        
        return pretty_string
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Block(object):
    """
    Block Choice Seismic Analysis in Python
    
    :param choice: parameters required to run the denoiser.
    :type choice: :py:class:`~pyblockseis.block.Parameter`
    :param data: obspy stream or :class:`~numpy.ndarray`.
    :type data: :class:`~obspy.core.stream.Stream` or 
    :param dt: sampling period required if data is :class:`~numpy.ndarray`.
    :type dt: float
    """
    def __init__(self,choice=None,data=np.array([]),dt=None):
        if choice is None:
            choice = Parameter()
        choice = deepcopy(choice)
        
        if isinstance(data, np.ndarray):
            if dt is None:
                msg = "dt must be defined when data is a numpy.ndarray."
                warnings.warn(msg)
            self.dt = dt
    
        self.params = choice
        self.data = data
        
    def batch_process(self):
        # continuous wavelet transform (multi-threading here)
        cwt = []
        for trace in self.data:
            data_out, scales = wavelet.forward_cwt(
            	trace.data, self.params.wave_type, self.params.nvoices, trace.stats.delta
            	)
            cwt.append(data_out)

        # Reconstruction
        icwt = []
        for arr in cwt:
        	icwt.append(
        		wavelet.inverse_cwt(arr, self.params.wave_type, self.params.nvoices)
        		)

        self.cwt = cwt
        self.icwt = icwt
    
    def __repr__(self):
        return repr(self.__dict__)