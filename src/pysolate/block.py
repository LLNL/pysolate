# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# LLNL-CODE-841231
# authors:
#        Andrea Chiang (andrea@llnl.gov)
#        Ana Aguiar (aguiarmoya1@llnl.gov)
"""
Core module for pySolate
"""


import pyasdf
import numpy as np
import concurrent.futures as cf
from copy import deepcopy
from obspy.core.event import Event, Catalog

from .wavelet import Wavelet, WaveletCollection, cwt, inverse_cwt, blockbandpass
from .threshold import noise_model, noise_thresholding, signal_thresholding
from .waveforms import Waveforms

import matplotlib.pyplot as plt

TAG_OPTIONS = ["input", "band_rejected", "noise_removed", "signal_removed"]
CUSTOM_ATTR = "wavelet"


def _worker(function, values, wave_type, nvoices):
    futures = []
    with cf.ThreadPoolExecutor(max_workers=32) as executor:
        for val in values:
            futures.append(executor.submit(function, val, wave_type, nvoices))

    return futures


def read(params=None, **kwargs):
    """
    Read waveform files and processing parameters into
    a pySolate Block object

    The :func:`~pysolate.block.read` function accepts waveform files
    in either ObsPy Stream or Trace objects, or something ObsPy can read.
    If  a :class:`~pysolate.block.Parameter` object is not provided, the
    default processing values will be used instead.

    :param params: parameters required to run the denoiser.
    :type params: :py:class:`~pysolate.block.Parameter`
    :param event: Object containing information that describes the seismic event.
    :type event: :class:`~obspy.core.event.event.Event` or
        :class:`~obspy.core.event.Catalog`, optional
    :param data: Waveform data in ObsPy trace or stream formats or something ObsPy can read.
    :type data: :class:`~obspy.core.stream.Stream`,
        :class:`~obspy.core.trace.Trace`, str, ...
    :param asdf_file: HDF5 file name.
    :type asdf_file: str
    :param field: path of waveform files within the ASDF volume, default is
        ``"raw_observed"``.
    :type field: str

    .. rubric:: Examples

    (1) Reading from a SAC file

        >>> import pysolate as bcs
        >>> params = bcs.params((block_threshold=1.0, noise_threshold="hard")
        >>> block = bcs.read(params=params, data="testdata/5014.YW.0.sp0011.DPZ")
        >>> # Using default values
        >>> block = bcs.read(data="testdata/5014.YW.0.sp0011.DPZ")

    (2) Reading from a ASDF file

        >>> import pysolate as bcs
        >>> params = bcs.params((block_threshold=1.0, noise_threshold="hard")
        >>> block = bcs.read(params=params, asdf_file="testdata/578449.h5")
    """
    # Parse kwargs
    asdf_file = kwargs.get("asdf_file", None)
    field = kwargs.get("field", "raw_observed")
    if asdf_file is None:
        event = kwargs.get("event", Event())
        if isinstance(event, Event):
            pass
        elif isinstance(event, Catalog):
            if len(event) > 1:
                raise ValueError("More than one event exists.")
            event = event[0]
        else:
            raise TypeError("Must be an ObsPy event or catalog instance.")

        data = kwargs.get("data", None)
        if data is None:
            raise ValueError("No data provided.")
    else:
        event, data = _read_asdf_file(asdf_file, field)

    if params is None:
        params = Parameter()
    else:
        params = deepcopy(params)

    # Initialize waveform object and add input data
    waveforms = Waveforms(TAG_OPTIONS)
    waveforms.add_waveform(data, tag="input")

    # Create Block object
    block = Block(params=params, event=event, waveforms=waveforms)

    return block


def _read_asdf_file(asdf_file, field):
    """
    Reads a pyasdf file and returns ObsPy Event and Stream objects

    :param asdf_file: HDF5 file name.
    :type asdf_file: str
    :param field: path of waveform files within the ASDF volume.
    :type field: str
    """
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


class Parameter(object):
    """
    A container of parameters required to run
    :py:class:`~pysolate.block.Block`
    
    The Parameter class determines the appropriate processes to be applied
    to the seismograms.
    
    :param wave_type: wavelet filter type, options are ``"morlet"``, ``"shannon"``,
        ``"mhat"``, ``"hhat"``. Default is ``"morlet"``.
    :type wave_type: str
    :param nvoices: number of voices, or the sampling of CWT in scale.
        Higher number of voices give finer resolution. Default is ``16``.
    :type nvoices: int
    :param bandpass_blocking: Default value ``True`` will apply a band rejection filter where
        wavelet coefficients are modified over a scale bandpass.
    :type bandpass_blocking: bool
    :param scale_min: minimum time scale for bandpass blocking. Default is ``1``.
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
        """
        Returns a list of object attributes.
        """
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
#        else:
#            raise KeyError("'%s' is not supported."%key)

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
            f.format(key, str(getattr(self, key))) for key in keys if key in self._defaults
        ])
        
        return pretty_string
    
    def _repr_pretty_(self, p, cycle):
        p.text(str(self))


class Block(object):
    """
    Root object for the time series and CWT

    Main class object that handles the processing of seismic data
    using the continuous wavelet transform. The ``tags`` attribute
    is a list of available processed waveforms and their wavelet transforms,
    depending on the choices set in :class:`~pysolate.block.Parameter`.
    See all possible tags below.

    :param params: CWT operations.
    :type params: :class:`~pysolate.block.Parameter`
    :param event: seismic event information.
    :type event: :class:`~obspy.core.event.event.Event`
    :param waveforms: seismic data.
    :type waveforms: :class:`~pysolate.waveforms.Waveforms`

    .. rubric:: Attributes

    ``params`` : Parameter object
        CWT operations.
    ``event`` : ObsPy event
        event information.
    ``waveforms`` : Waveforms object
        seismic waveforms.
    ``wavelets`` : WaveletCollection object
        wavelet transforms of the processed data.
    ``noise_model_tag`` : str
        wavelet transform of data used to estimate the noise model,
        depending on the processing choices this will be
        ``"input"``, ``"band_rejected"`` or ``None``.

    .. rubric:: Available Tags

    .. cssclass: table-striped

    ================   =====================   ===============================
    ``tag``            parameters              description
    ================   =====================   ===============================
    "input"            none                    input data
    "band_rejected"    ``bandpass_blocking``   applied a band rejection filter
    "noise_removed"    ``noise_threshold``     noise removed from data
    "signal_removed"   ``signal_threshold``    signal removed from data
    ================   =====================   ===============================

    .. rubric:: Basic Usage

    >>> import pysolate as bcs
    >>> block = bcs.read(data="testdata/5014.YW.0.sp0011.DPZ")
    >>> block.run()
    """
    def __init__(self, params, event, waveforms):
        self.params = params
        self.event = event
        self.waveforms = waveforms

        self.current_params = None
        self.current_tag = "input"
        self.noise_model_tag = None

        self.wavelets = {}

    def get_station_list(self):
        """
        Function to return a list of available stations
        """
        return self.waveforms.station_name

    def run(self):
        """
        Function to run the CWT-based thresholding operations

        Apply the nonlinear thresholding operations given the values
        in the ``params`` attribute and reconstruct the time series.
        """
        if self.current_params is not None:
            changed_keys = [
                key for key in self.params.keys() if self.params[key] != self.current_params[key]
            ]
            if len(changed_keys) > 0:
                self._update_run(changed_keys)
            else:
                print("Parameters did not change, nothing is done.")
        else:
            self._run_all()

        # Current state of CWT operations
        self.current_params = deepcopy(self.params)
        if self.current_params.estimate_noise:
            self.noise_model_tag = self.current_tag

    def _run_all(self):
        """
        Run the entire denoising process in wavelet domain
        """
        self.tags = [] # all available tags
        # continuous wavelet transform of input data
        self._call_cwt()

        # Apply bandpass blocking
        if self.params.bandpass_blocking:
            self._apply_bandpass_blocking()

        # Estimate noise model
        if self.params.estimate_noise:
            self._call_noise_model()
            if self.params.noise_threshold is not None:
                self._apply_thresholding("noise_removed")
            if self.params.signal_threshold is not None:
                self._apply_thresholding("signal_removed")

    def _update_run(self, changed_keys):
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
                self._apply_bandpass_blocking()
            if apply_noise_model:
                print("Estimate new noise model.")
                self._call_noise_model()
            if apply_noise_threshold:
                print("Apply new noise thresholding.")
                self._apply_thresholding("noise_removed")
            if apply_signal_threshold:
                print("Apply new signal thresholding.")
                self._apply_thresholding("signal_removed")
            # Refresh
            if len(refresh_waves) > 0:
                self._refresh_cwt_icwt(refresh_waves)
            if len(refresh_noise) > 0:
                self._refresh_noise_model(refresh_noise)

    def _refresh_cwt_icwt(self, tags):
        # refresh cwt and wavelet processed data
        for tag in tags:
            for wave in self.wavelets[tag]:
                wave.coefs = None
            # Delete tagged data, raise error if tag (key) does not exist
            self.waveforms.data.pop(tag) # self.waveforms._data.pop(tag, None)
            self.tags.remove(tag)

    def _refresh_noise_model(self, noise_param):
        for wave in self.wavelets[self.current_tag]:
            wave[noise_param].clear()

    def _call_cwt(self, tag="input"):
        """
        Continues wavelet transform of waveform data
        """
        st = self.waveforms.data[tag]
        out = _worker(cwt, st, self.params.wave_type, self.params.nvoices)
        waves = []
        for trace, f in zip(st, out):
            wx, scales = f.result()
            waves.append(Wavelet(coefs=wx, scales=scales, headers=trace.stats))
        #return [f.result() for f in futures]
        #for (wx, scales), trace in zip(out, st):
        #    waves.append(Wavelet(coefs=wx, scales=scales, headers=trace.stats))

        self.wavelets[tag] = WaveletCollection(wavelets=waves)
        self.current_tag = tag
        self.tags = tag

    def _apply_bandpass_blocking(self, tag="band_rejected"):
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

    def _call_noise_model(self):
        """
        Construct noise model
        """
        if hasattr(self.params, "external_noise_model"):
            kwargs = {k:v for k, v in self.params.external_noise_model.items() }
            self._get_external_noise_model(**kwargs)
        else:
            self._compute_noise_model()

    def _compute_noise_model(self):
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

    def _get_external_noise_model(self, M=None, S=None, P=None):
        """
        Pass a different noise model
        """
        if M is None or S is None or P is None:
            raise ValueError("Invalid noise model parameter(s): M, S and/or P.")

        for wave in self.wavelets[self.current_tag]:
            wave.noise_model.M = M
            wave.noise_model.S = S
            wave.noise_model.P = P

    def _apply_thresholding(self, method):
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
        Reconstruct time series

        A function that performs the inverse continuous wavelet transform
        to reconstruct the time series from the wavelet coefficients.

        :param tag: time series to reconstruct.
        :type tag:
        """
        # check if tags are available?
        # self.wavelets.keys()
        if tag not in self.waveforms.data.keys():
            st = self.waveforms.data["input"].copy()
        else:
            st = self.waveforms.data[tag]

        coefs = [wave.coefs for wave in self.wavelets[tag]]
        out = _worker(inverse_cwt, coefs, self.params.wave_type, self.params.nvoices)
        for trace, f in zip(st, out):
            trace.data = f.result()

        self.waveforms.data[tag] = st

    def get_waveforms(self, tag):
        """
        Returns a :class:`~pysolate.waveforms.Waveforms`
        object

        A function that returns the waveforms of a tagged dataset

        :param tag: processed data
        :type tag: str
        """
        if tag not in self.tags:
            msg = ', '.join([ "'%s'"%_i for _i in self.tags])
            msg = ' '.join(
                ["Wavelet transform is not available, available tags are", msg]
            )
            print(msg)
        else:
            return self.waveforms.data[tag]

        return self.waveforms[tag]

    def get_wavelets(self, tag):
        """
        Returns a :class:`~pysolate.wavelet.WaveletCollection`
        object

        A function that returns the wavelet transform of a tagged dataset

        :param tag: processed data
        :type tag: str
        """
        if tag not in self.tags:
            msg = ', '.join([ "'%s'"%_i for _i in self.tags])
            msg = ' '.join(
                ["Wavelet transform is not available, available tags are", msg]
            )
            print(msg)
        else:
            return self.wavelets[tag]

    def write(self, tag, output=None, filename=None, format="npz",
              network=None, station=None, location=None, channel=None, component=None):
        """
        Write the processed data to file

        Function to save the waveforms or wavelet transforms to a single
        uncompressed NumPy .npz format. Additional formats for the
        waveforms are supported through ObsPy, such as binary SAC.
        See :meth:`obspy.core.stream.Stream.write` for the supported waveform
        data formats.

        :param tag: input or processed dataset to save.
        :type tag: str
        :param output: type of output to save, ``"waveforms"`` for time series data, or
            ``"cwt"`` for the continuous wavelet transform.
        :type output: str
        :param filename: name of the file to write, optional.
        :type filename: str
        :param format: output file format, default is ``"npz"``. For waveform data
            see :meth:`~obspy.core.stream.Stream.write` for additional supported
            formats.
        :type format: str
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
        if output is None:
            print("No output type selected, nothing is done.")
        else:
            output = output.lower()
            format = format.lower()
            # Write all stations if none specified
            stations = all(
                v is None for v in [
                    network, station, location, channel, component]
            )
            if output == "waveforms":
                if stations:
                    st = self.waveforms.data[tag]
                else:
                    st = self.waveforms.data[tag].select(
                        network=network,
                        station=station,
                        location=location,
                        channel=channel,
                        component=component,
                    )
                if format == "npz":
                    _write_waveforms_npz(st, filename)
                else:
                    if filename is None:
                        for tr in st:
                            if len(tr.stats.network) > 0:
                                filename = "%s.%s"%(tr.id, format)
                            else:
                                filename = "%s.%s.%s.%s"%(
                                    tr.stats.station,
                                    tr.stats.location,
                                    tr.stats.channel,
                                    format
                                )
                            tr.write(filename, format=format)
                    else:
                        st.write(filename, format=format)
            elif output == "cwt":
                if stations:
                    waves = self.wavelets[tag]
                else:
                    waves = self.wavelets[tag].select(
                        network=network,
                        station=station,
                        location=location,
                        channel=channel,
                        component=component,
                    )
                if format == "npz":
                    _write_wavelets_npz(waves, filename)
                else:
                    msg = "%s is not a valid format for wavelet transforms."%format
                    raise ValueError(msg)
            else:
                msg = "%s is not a valid output type."%output
                raise ValueError(msg)

    def plot(self, tag,
             network=None, station=None, location=None, channel=None, component=None):
        """
        Plot the time-frequency representation, or scalogram, of the current
        pysolate Block object

        Plot the scalograms for all traces in the object. Alternatively,
        specific traces can be selected that matches the given station criteria.

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

        .. rubric:: Example

        >>> block.plot("noise_removed", network="BK", channel="HH*")
        """
        # Plot all stations if none specified
        if all (
                v is None for v in [network, station, location, channel, component]
        ):
            st = self.waveforms.data[tag]
            waves = self.wavelets[tag]
        else:
            st = self.waveforms.data[tag].select(
                network=network,
                station=station,
                location=location,
                channel=channel,
                component=component,
            )
            waves = self.wavelets[tag].select(
                network=network,
                station=station,
                location=location,
                channel=channel,
                component=component
            )

        # matplotlib.patch.Patch properties for text box
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        for wave, tr in zip(waves, st):
            y_axis = np.log10(wave.scales)
            times = tr.times()

            fig = plt.figure(figsize=(10,6.5))
            ax1 = fig.add_subplot(2,1,1)
            extent = [min(times), max(times), max(y_axis), min(y_axis)]
            ax1.imshow(abs(wave.coefs), extent=extent, aspect="auto")
            ax1.set_xlim([np.min(times), np.max(times)])
            ax1.set_ylim([np.max(y_axis), np.min(y_axis)])
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
        """
        Returns an ObsPy event object containing information
        that describes the seismic event
        """
        return self._event

    @event.setter
    def event(self, event):
        if isinstance(event, Event):
            self._event = event
        else:
            raise TypeError("Must be an ObsPy event instance.")

    @property
    def tags(self):
        """
        Returns a list of available tags in the dataset after applying
        the wavelet thresholding operations
        """
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
        Return better readable string representation of Block object
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


def _write_waveforms_npz(st, filename):
    if filename is None:
        for tr in st:
            if len(tr.stats.network) > 0:
                filename = tr.id + ".npz"
            else:
                filename = "%s.%s.%s.npz" % (
                    tr.stats.station,
                    tr.stats.location,
                    tr.stats.channel,
                )
            np.savez(filename, times=tr.times(), data=tr.data)
    else:
        if isinstance(filename, str):
            kwds = {}
            for tr in st:
                kwds["%s_times"%tr.id] = tr.times()
                kwds["%s_data"%tr.id] = tr.data
            np.savez(filename, **kwds)
        else:
            raise TypeError("'filename' must be a string.")


def _write_wavelets_npz(waves, filename):
    if filename is None:
        for w in waves:
            filename = "%s.cwt.npz"%w.get_id()
            if filename.startswith("."):
                filename = filename[1:]
            np.savez(filename, scales=w.scales, coefs=w.coefs)
    else:
        if isinstance(filename, str):
            kwds = {}
            for w in waves:
                kwds["%s_scales"%w.get_id()] = w.scales
                kwds["%s_coefs"%w.get_id()] = w.coefs
            np.savez(filename, **kwds)
        else:
            raise TypeError("'filename' must be a string.")
