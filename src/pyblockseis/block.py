# -*- coding: utf-8 -*-
# SPDX-License-Identifier: (LGPL-3.0)
# LLNL-CODE-XXXXX
# authors:
#        Ana Aguiar Moya (aguiarmoya1@llnl.gov)
#        Andrea Chiang (andrea4@llnl.gov)

import numpy as np
import pyasdf
from copy import deepcopy
from obspy.core.event import Event, Catalog

from .wavelet import Wavelet, WaveletCollection, cwt, inverse_cwt, blockbandpass
from .threshold import noise_model, noise_thresholding, signal_thresholding
from .waveforms import Waveforms

import matplotlib.pyplot as plt

TAG_OPTIONS = ["input", "band_rejected", "noise_removed", "signal_removed"]
CUSTOM_ATTR = "wavelet"


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
            if not isinstance(value, self._types[key]):
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
    :param event: Object containing information that describes the seismic event.
    :type event: :class:`~obspy.core.event.Event` or
        :class:`~obspy.core.event.Catalog`
    :param data: Waveform data in ObsPy trace or stream formats or something ObsPy can read.
    :type data: :class:`~obspy.core.stream.Stream`,
        :class:`~obspy.core.trace.Trace`, str, ...
    :param asdf_file: HDF5 file name.
    :type asdf_file: str
    :param field: path of waveform files within the ASDF volume, default is ``"raw_observed"``.
    :type field: str
    """
    def __init__(self,choice=None, **kwargs):
        # Parse kwargs
        asdf_file = kwargs.get("asdf_file", None)
        field = kwargs.get("field", "raw_observed")
        if asdf_file is None:
            event = kwargs.get("event", Event())
            data = kwargs.get("data", None)
            if data is None:
                raise ValueError("No data provided.")
        else:
            event, data = self._read_asdf_file(asdf_file, field)

        # Initialize wavelet parameters
        if choice is None:
            choice = Parameter()
        self.params = deepcopy(choice)
        self.current_params = None
        self.current_tag = None # Tagged data used to estimate noise model
        self.wavelets = {}

        # Initialize waveform object
        waveforms = Waveforms(TAG_OPTIONS)

        # Add input data
        waveforms.add_waveform(data, tag="input")

        self.current_tag = "input"
        self.event = event
        self.waveforms = waveforms

    def _read_asdf_file(self, asdf_file, field):
        with pyasdf.ASDFDataSet(asdf_file) as ds:
            if len(ds.events) > 1:
                raise ValueError("ASDF volume contains more than one seismic event.")
            event = ds.events[0]
            for i, stat in enumerate(ds.waveforms):
                if i == 0:
                    data = stat[field]
                else:
                    data += stat[field]

        return event, data

    def get_station_list(self):
        return self.waveforms.station_name

    def run(self):
        if self.current_params is not None:
            changed_keys = [key for key in self.params.keys() if self.params[key] != self.current_params[key]]
            if len(changed_keys) > 0:
                self.update_run(changed_keys)
            else:
                print("Parameters did not change, nothing is done.")
        else:
            self._run_all()

        # Current state of CWT operations
        self.current_params = deepcopy(self.params)

    def _run_all(self):
        """
        Run the entire denoising process in wavelet domain
        """
        self.tags = [] # all available tags
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

    def call_cwt(self, tag="input"):
        """
        Continues wavelet transform of waveform data
        """
        waves = []
        for trace in self.waveforms.data[tag]:
            wx, scales = cwt(trace.data, self.params.wave_type, self.params.nvoices, trace.stats.delta)
            waves.append(Wavelet(coefs=wx, scales=scales, headers=trace.stats))

        self.wavelets[tag] = WaveletCollection(wavelets=waves)
        self.current_tag = tag
        self.tags = tag

    def apply_bandpass_blocking(self, tag="band_rejected"):
        """
        Apply bandpass filtering on CWT of input data
        """
        waves = []
        for wave in self.wavelets["input"]:
            wx = blockbandpass(
                wave.coefs,
                wave.scales,
                self.params.scale_min,
                self.params.scale_max,
                self.params.block_threshold
            )
            waves.append(Wavelet(coefs=wx, scales=wave.scales, headers=wave.stats))

        self.wavelets[tag] = WaveletCollection(wavelets=waves)
        self.reconstruct(tag)
        self.current_tag = tag
        self.tags = tag

    def call_noise_model(self):
        """
        Construct noise model
        """
        for wave in self.wavelets[self.current_tag]:
            M, S, P = noise_model(
                wave.coefs, # cwt coefficients used to estimate noise model
                wave.stats.delta,
                self.params.noise_starttime,
                self.params.noise_endtime,
                self.params.nsigma_method,
                self.params.snr_lowerbound,
                self.params.snr_detection
            )
            wave.noise_model.M = M
            wave.noise_model.S = S
            wave.noise_model.P = P

    def apply_thresholding(self, method):
        """
        Apply noise/signal thresholding
        """
        _get_function = {"noise_removed": (noise_thresholding, self.params.noise_threshold),
                         "signal_removed": (signal_thresholding, self.params.signal_threshold)
                        }
        func = _get_function[method]
        waves = []
        for wave in self.wavelets[self.current_tag]:
            wx = func[0](wave.coefs, func[1], wave.noise_model.P)
            waves.append(Wavelet(coefs=wx, scales=wave.scales, headers=wave.stats))

        self.wavelets[method] = WaveletCollection(wavelets=waves)
        self.reconstruct(method)
        self.tags = method

    def reconstruct(self, tag):
        """
        Reconstruct time series.
        """
        if tag not in self.waveforms.data.keys():
            st = self.waveforms.data["input"].copy()
        else:
            st = self.waveforms.data[tag]

        for wave, trace in zip(self.wavelets[tag], st):
            trace.data = inverse_cwt(
                wave.coefs,
                self.params.wave_type,
                self.params.nvoices,
            )

        self.waveforms.data[tag] = st

    def update_run(self, changed_keys):
        bandpass_key = [
            "bandpass_blocking",
            "scale_min",
            "scale_max",
            "block_threshold"
        ]
        noise_model_key = [
            "estimate_noise",
            "noise_starttime",
            "noise_endtime",
            "nsigma_method",
            "snr_lowerbound",
            "snr_detection"
        ]
        if any(key in changed_keys for key in ["wave_type", "nvoices"]):
            print("Wavelet has been changed, re-run all wavelet processing.")
            self._run_all()
        else:
            refresh_waves = [] # cwt/icwt to refreshed
            refresh_noise = [] # noise model parameters to refresh
            apply_blocking = False
            apply_noise_model = False
            apply_noise_threshold = False
            apply_signal_threshold = False

            if self.params.bandpass_blocking:
                # Changed any filter parameters
                if any(key in changed_keys for key in bandpass_key):
                    apply_blocking = True
            else:
                self.current_tag = "input"
                refresh_waves.append("band_rejected")

            if self.params.estimate_noise:
                # Estimate new noise model
                if any(key in changed_keys for key in noise_model_key):
                    apply_noise_model = True

                # Update noise thresholding if needed
                if self.params.noise_threshold is None:
                    refresh_waves.append("noise_removed")
                else:
                    if "noise_threshold" in changed_keys:
                        apply_noise_threshold = True
                    # re-run if any changes to prior operations
                    elif any((apply_blocking, apply_noise_model)):
                        apply_noise_threshold = True

                # Update signal thresholding if needed
                if self.params.signal_threshold is None:
                    refresh_waves.append("signal_removed")
                else:
                    if "signal_threshold" in changed_keys:
                        apply_signal_threshold = True
                    # re-run if any changes to prior operation
                    elif any((apply_blocking, apply_noise_model)):
                        apply_signal_threshold = True
            else:
                # No thresholding
                refresh_noise = ["noise_model"]
                refresh_waves.append("noise_removed")
                refresh_waves.append("signal_removed")

            # Apply changes
            if apply_blocking:
                print("Apply new bandpass blocking.")
                self.apply_bandpass_blocking()
            if apply_noise_model:
                print("Estimate new noise model.")
                self.call_noise_model()
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

    def refresh_cwt_icwt(self, tags):
        # refresh cwt and wavelet processed data
        for tag in tags:
            for wave in self.wavelets[tag]:
                wave.coefs = None
            # Delete tagged data, raise error if tag (key) does not exist
            self.waveforms.data.pop(tag) # self.waveforms._data.pop(tag, None)
            self.tags.remove(tag)

    def refresh_noise_model(self, noise_param):
        for wave in self.wavelets[self.current_tag]:
            wave[noise_param].clear()

    def get_noise_model(self):
        """
        Returns a WaveletCollection with noise model
        """
        return self.wavelets[self.current_tag]

    def plot(self, tag, network=None, station=None, location=None, channel=None):
        """
        Plot time-frequency representation (TFR) of a single time series.

        :param trace_id: time series to plot.
        :type trace_id: int
        """
        # Plot all stations if none specified
        if all (v is None for v in [network, station, location, channel]):
            st = self.waveforms.data[tag]
            waves = self.wavelets[tag]
        else:
            st = self.waveforms.data[tag].select(
                network=network,
                station=station,
                location=location,
                channel=channel,
            )
            waves = self.wavelets[tag].select(
                network=network,
                station=station,
                location=location,
                channel=channel,
            )

        # matplotlib.patch.Patch properties for text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        for wave, tr in zip(waves, st):
            y_axis = np.log10(wave.scales)
            times = tr.times()

            fig = plt.figure(figsize=(10,6.5))
            ax1 = fig.add_subplot(2,1,1)
            extent = [min(times), max(times), min(y_axis), max(y_axis)]
            ax1.imshow(abs(wave.coefs), extent=extent, aspect="auto")
            ax1.set_xlim([np.min(times), np.max(times)])
            ax1.set_ylim([np.min(y_axis), np.max(y_axis)])
            ax1.set_ylabel("Scale [log10(s)]")
            ax1.set_title("Scalogram: %s | %s"%(tr.id, tag))

            ax2 = fig.add_subplot(2,1,2)
            ax2.plot(times, tr.data, color="black", linewidth=1)
            # place a text box in upper left in axes coords
            ax2.text(
                0.01, 0.95, r'%s'%tr.id,
                transform=ax2.transAxes, verticalalignment='top',
                bbox=props,
            )
            ax2.set_xlim([min(times), max(times)])
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Amplitude")

            ax2.set_title("%s - %s"%(tr.stats.starttime, tr.stats.endtime))
            plt.show()

    @property
    def event(self):
        return self._event

    @event.setter
    def event(self, event):
        if isinstance(event, Event):
            event = event
        elif isinstance(event, Catalog):
            if len(event) > 1:
                raise ValueError("More than one event exists.")
            event = event[0]
        else:
            raise TypeError("Must be an ObsPy event or catalog instance.")

        self._event = event

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tag):
        if isinstance(tag, str):
            tags = [tag]
        elif isinstance(tag, list):
            if all(isinstance(_i, str) for _i in tag):
                tags = tag
        else:
            raise TypeError("Must be a string or list of strings.")
        if len(tags) > 0:
            for tag in tags:
                if tag not in self._tags:
                    self._tags.append(tag)
        else:
            self._tags = []

    def __str__(self):
        """
        Return better readable string representation of Block object.
        """
        pretty_string = '\n'.join(
            [
                "Block contains %d trace(s):"%len(self.waveforms.data["input"]),
                self.params.__str__()
            ]

        )

        return pretty_string
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))
