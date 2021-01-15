# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Ana Aguiar Moya (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea4@llnl.gov)

#import warnings
import numpy as np
from copy import deepcopy

from .wavelet import cwt, inverse_cwt, blockbandpass
from .threshold import noise_model, noise_thresholding, signal_thresholding
import matplotlib.pyplot as plt


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
        wavelet coefficients are modified over a scale bandpass.
    :type bandpass_blocking: bool
    :param scale_min: minimum time scale for bandpass blocking. Default is ``0``.
    :type scale_min: float
    :param scale_max: maximum time scale for bandpass blocking. Default is ``200``.
    :type scale_max: float
    :param block_threshhold: percent amplitude adjustment to the wavelet coefficients within
        ``scale_min`` and ``scale_max``. For example a threshold of 5% means the wavelet cofficients
        in the band will be multipled by 0.05. Default is ``0``.
    :type block_threshold: float
    :param estimate_noise: flag to compute the noise model, default is ``True``.
    :type estimate_noise: bool
    :param noise_starttime: noise start time, default is ``0``.
    :type noise_starttime: float
    :param noise_endtime: noise end time, default is ``60``.
    :type noise_endtime: float
    :param noise_threshold: type of noise thresholding to be applied, the options are
        ``"hard"`` for hard thresholding and ``"soft"`` for soft thresholding. Default is ``None``.
    :type noise_threshold: str
    :param signal_threshold: type of signal thresholding to be appied, the options are
        ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding. Default is ``None``.
    :type signal_threshold: str
    :param nsigma_method: method to determine the number of standard deviations for block thresholding.
        ``"donoho"`` for Donoho's Threshold criterion and ``"ECDF"`` for empirical cumulative probability
        distribution method. You can also specify the number of standard deviations by entering a number.
        None ECDF method assumes Gaussian statistic. The default method ``"ECDF"`` is recommended.
    :type nsigma_method: str, int, float
    :param snr_detection: Flag to apply the SNR detection method, default is ``False``. If ``True`` it
        will be applied before hard thresholding.
    :type snr_detection: bool
    :param snr_lowerbound: Noise level percent lower bound. Default is ``1.0``.
    :type snr_lowerbound: float
    
    """
    # Default values
    _defaults = dict(
        wave_type = "morlet",
        nvoices = 16,
        bandpass_blocking = True,
        scale_min = 1.0,
        scale_max = 200.0,
        block_threshold = 0.0,
        estimate_noise = True, # params after require estimate_noise = True
        noise_starttime = 0.0,
        noise_endtime = 60.0,
        noise_threshold = None,
        signal_threshold = None,
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
        block_threshold = float,
        estimate_noise = bool,
        noise_starttime = float,
        noise_endtime = float,
        noise_threshold = (str, type(None)),
        signal_threshold = (str, type(None)),
        nsigma_method = (str, float, int),
        snr_detection = bool,
        snr_lowerbound = float,
    )
    
    def __init__(self, *args, **kwargs):
        # Set to default values
        self.__dict__.update(self._defaults)
        # Overwrite default
        self.update(dict(*args, **kwargs))

    def keys(self):
        return list(self.__dict__.keys())

    def update(self, adict={}):
        for (key, value) in adict.items():
            self.__setitem__(key, value)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        # type check
        if key in self._types:
            if not isinstance(value,self._types[key]):
                value = self._types[key](value)
        else:
            raise KeyError("'%s' is not supported."%key)
                
        # check methods
        if key in "wave_type":
            if value not in ("morlet","shannon","mhat","hhat"):
                msg = "'%s' wavelet type is not supported."%value
                raise ValueError(msg)
        if key == "noise_threshold" and value is not None:
            if value not in ("hard","soft"):
                msg = "'%s' thresholding is not supported."%value
                raise ValueError(msg)
        if key == "signal_threshold" and value is not None:
            if value not in ("hard","soft"):
                msg = "'%s' thresholding is not supported."%value
                raise ValueError(msg)
        if key == "nsigma_method":
            if isinstance(value, str):
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


class Wavelet(object):
    """
    Object containing outputs from CWT operations
    """
    def __init__(self):

        default_keys = ["input", "band_rejected", "noise_removed", "signal_removed"]
        self.scale = None
        self.cwt = {key: None for key in default_keys}
        self.icwt = {key: None for key in default_keys}
        self.M = None
        self.S = None
        self.P = None
        #self.window_start = None
        #self.window_end = None

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

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
        f = "{0:>%d}: {1}\n" % i
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
    :param data: obspy stream.
    :type data: :class:`~obspy.core.stream.Stream` or 
    :param dt: sampling period required if data is :class:`~numpy.ndarray`.
    :type dt: float
    :param num_of_traces: number of time series.
    :type num_of_traces: int
    """
    def __init__(self,choice=None, data=[]):
        if choice is None:
            choice = Parameter()
        choice = deepcopy(choice)

        self.params = choice
        self.data = data
        self.wavelet_function = None # last CWT operation
        self.last_params = None

    def run(self):
        if self.last_params is not None:
            changed_keys = [key for key in self.params.keys() if self.params[key] != self.last_params[key]]
            if len(changed_keys) > 0:
                self.update_run(changed_keys)
            else:
                print("Parameters did not change, nothing is done.")
        else:
            self.run_all()

        # Reconstruction of final cwt
        self.reconstruct(self.wavelet_function)

        # Keeping track of all CWT operations
        self.last_params = deepcopy(self.params)

    def run_all(self):
        # continuous wavelet transform of input data
        self.call_cwt()

        # Apply bandpass blocking
        if self.params.bandpass_blocking:
            self.apply_bandpass_blocking()

        # Estimate noise model
        if self.params.estimate_noise:
            self.call_noise_model()
            if self.params.noise_threshold is not None:
                self.apply_thresholding("noise_removed")
            if self.params.signal_threshold is not None:
                self.apply_thresholding("signal_removed")

    def call_cwt(self):
        for trace in self.data:
            trace.wavelet = Wavelet()
            wx, scale = cwt(trace.data, self.params.wave_type, self.params.nvoices, trace.stats.delta)
            trace.wavelet.cwt["input"] = wx
            trace.wavelet.scale = scale
        self.wavelet_function = "input"

    def apply_bandpass_blocking(self):
        # Apply on input data
        for trace in self.data:
            trace.wavelet.cwt["band_rejected"] = blockbandpass(
                trace.wavelet.cwt["input"], trace.wavelet.scale,
                self.params.scale_min, self.params.scale_max, self.params.block_threshold)
        self.wavelet_function = "band_rejected"

    def call_noise_model(self):
        for trace in self.data:
            wx = trace.wavelet.cwt[self.wavelet_function]
            M, S, P = noise_model(wx, trace.stats.delta, self.params.noise_starttime, self.params.noise_endtime,
                                  self.params.nsigma_method, self.params.snr_lowerbound, self.params.snr_detection)

            trace.wavelet.M = M
            trace.wavelet.S = S
            trace.wavelet.P = P

#    def call_snr_detection(self):
#        # Call this function if noise model exists
#        for trace in self.data:
#            wx = trace.wavelet.cwt[self.wavelet_function]
#            M, S = SNR_detect(wx, trace.wavelet.M,
#                              trace.wavelet.window_start, trace.wavelet.window_end, self.params.snr_lowerbound)
#            trace.wavelet.M = M
#            trace.wavelet.S = S

    def apply_thresholding(self,method):
        _get_function = {"noise_removed": (noise_thresholding, self.params.noise_threshold),
                         "signal_removed": (signal_thresholding, self.params.signal_threshold)
                        }
        func = _get_function[method]

        for trace in self.data:
            # noise/signal thresholding
            trace.wavelet.cwt[method] = func[0](trace.wavelet.cwt[self.wavelet_function], func[1], trace.wavelet.P)

        self.reconstruct(method)

    def reconstruct(self, wavelet_function, trace_id=None):
        """
        Reconstruct time series
        """
        if trace_id is None:
            for trace in self.data:
                trace.wavelet.icwt[wavelet_function] = inverse_cwt(
                    trace.wavelet.cwt[wavelet_function], self.params.wave_type, self.params.nvoices)
        elif isinstance(trace_id, int):
            trace = self.data[trace_id]
            icwt = inverse_cwt(trace.wavelet.cwt[wavelet_function],
                               self.params.wave_type, self.params.nvoices)
            trace.wavelet.icwt[wavelet_function] = icwt
            return icwt
        else:
            raise TypeError("trace_id must be an integer or NoneType")

    def update_run(self, changed_keys):
        if any(key in changed_keys for key in ["wave_type", "nvoices"]):
            print("Wavelet has been changed, re-run all data processing.")
            self.run_all()
        else:
            refresh_waves = [] # cwt/icwt to refreshed
            refresh_noise = [] # noise model parameters to refresh
            apply_blocking = False
            apply_noise_model = False
            #apply_snr_detection = False
            apply_noise_threshold = False
            apply_signal_threshold = False

            if self.params.bandpass_blocking:
                if "bandpass_blocking" in changed_keys:
                    # Turned on bandpass blocking
                    apply_blocking = True
                elif any(key in changed_keys for key in ["scale_min", "scale_max", "block_threshold"]):
                    # Changed any filter parameters
                    apply_blocking = True
            else:
                self.wavelet_function = "input"
                refresh_waves.append("band_rejected")

            if self.params.estimate_noise:
                if "estimate_noise" in changed_keys:
                    # Turned on noise
                    apply_noise_model = True
                elif any(key in changed_keys
                         for key in ["noise_starttime", "noise_endtime", "nsigma_method",
                                     "snr_lowerbound","snr_detection"]):
                    # changed any model parameters
                    apply_noise_model = True
                #else:
                #    # Check if SNR detection is turned on when noise model is the same
                #    if self.params["snr_detection"]:
                #        if "snr_detection" in changed_keys:
                #            # turned on detection
                #            apply_snr_detection = True
                #        elif "snr_lowerbound" in changed_keys: # on-on
                #            # changed detection bounds
                #            apply_snr_detection = True
                # Check and update noise thresholding
                if self.params.noise_threshold is None:
                    refresh_waves.append("noise_removed")
                else:
                    if "noise_threshold" in changed_keys:
                        apply_noise_threshold = True
                    # re-run if any changes to prior operation
                    elif apply_blocking:
                        apply_noise_threshold = True
                    elif apply_noise_model:
                        apply_noise_threshold = True
                # Update signal thresholding
                if self.params.signal_threshold is None:
                    refresh_waves.append("signal_removed")
                else:
                    if "signal_threshold" in changed_keys:
                        apply_signal_threshold = True
                    # re-run if any changes to prior operation
                    elif apply_blocking:
                        apply_signal_threshold = True
                    elif apply_noise_model:
                        apply_signal_threshold = True
            else:
                refresh_noise = ["M", "S", "P"]
                refresh_waves.append("noise_removed")
                refresh_waves.append("signal_removed")

            # Apply changes
            if apply_blocking:
                print("Apply new bandpass blocking.")
                self.apply_bandpass_blocking()
            if apply_noise_model:
                print("Apply new noise model.")
                self.call_noise_model()
            #if apply_snr_detection:
            #   print("Apply new snr detection.")
            #    self.call_snr_detection()
            if apply_noise_threshold:
                print("Apply new noise thresholding.")
                self.apply_thresholding("noise_removed")
            if apply_signal_threshold:
                print("Apply new signal thresholding.")
                self.apply_thresholding("signal_removed")
            # Refresh
            if len(refresh_waves) > 0:
                self.refresh_cwt_icwt(refresh_waves)
            if len(refresh_noise) > 0:
                self.refresh_noise_model(refresh_noise)

    def refresh_cwt_icwt(self,wavelet_functions):
        # refresh cwt and icwt
        for trace in self.data:
            for wave in wavelet_functions:
                trace.wavelet.cwt[wave] = None
                trace.wavelet.icwt[wave] = None

    def refresh_noise_model(self,noise_param):
        for trace in self.data:
            for param in noise_param:
                trace.wavelet[param] = None

    def plot(self, wavelet_function, trace_id=0):
        """
        Plot time-frequency representation (TFR) of a single time series.

        :param trace_id: time series to plot.
        :type trace_id: int
        """
        cwt = self.data[trace_id].wavelet.cwt[wavelet_function]
        y_axis = np.log10(self.data[trace_id].wavelet.scale)
        if cwt is None:
            raise TypeError("No CWT on %s time series."%wavelet_function)
        else:
            times = self.data[trace_id].times()
        if wavelet_function == "input":
            icwt = self.data[trace_id].data
        else:
            arr = self.data[trace_id].wavelet.icwt[wavelet_function]
            if arr is None:
                icwt = self.reconstruct(wavelet_function, trace_id=trace_id)
            else:
                icwt = arr

        fig = plt.figure(figsize=(10,6.5))
        ax1 = fig.add_subplot(2,1,1)
        extent = [min(times), max(times), min(y_axis), max(y_axis)]
        ax1.imshow(abs(cwt), extent=extent, aspect="auto") # need to fix x-axis to time
        ax1.set_xlim([np.min(times), np.max(times)])
        ax1.set_ylim([np.min(y_axis), np.max(y_axis)])
        ax1.set_ylabel("Scale [log10(s)]")
        ax1.set_title("Time Frequency Representation (TFR) of %s time series"%wavelet_function)

        ax2 = fig.add_subplot(2,1,2)
        ax2.plot(times, icwt, color="black", linewidth=1)
        ax2.set_xlim([min(times), max(times)])
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Amplitude")
        plt.show()

    def __str__(self):
        """
        Return better readable string representation of Block object.
        """
        pretty_string = '\n'.join(["| Denoising parameters for %d time series |"%len(self.data),
            self.params.__str__()])
        ""
        
        return pretty_string
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    #def __repr__(self):
    #    return repr(self.__dict__)
