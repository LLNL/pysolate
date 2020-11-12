# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# 1. Input data: time series (numpy arrays)
# 2. Input parameters
# 3. Functions
#   - compute continuous wavelet transform (CWT)
#   - apply a block bandpass
#   - calculate noise model and threshold function
#   - apply SNR detection method
#   - apply hard thresholding to the noise (noise removal)
#   - apply soft thresholding to the noise (noise removal)
#   - apply hard thresholding to the signal (signal removal)
#   - apply soft thresholding to the signal (signal removal)
#   - compute the inverse CWT

# +
# Parameter container
# Emulating container types
# https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

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
        ``"Donoho"`` for Donoho's Threshold criterion, ``"ECDF"`` for empirical cumulative probability
        distribution method. The default method ``"ECDF"`` is recommended.
    :type nsgima_method: str
    :param snr_detection: Flag to apply the SNR detection method, deafult is ``False``. If ``True`` it
        will be applied before hard thresholding.
    :type snr_detection: bool
    :param snr_lowerbound: precent lower bound for SNR detection. Default is ``1.0``.
    :type snr_lowerbound: float
    
    """
    # Default values
    defaults = dict(
        wave_type = "morlet",
        nvoices = 16,
        bandpass_blocking = True,
        scale_min = 1.0,
        scale_max = 200.0,
        block_threshhold = 0.0,
        noise_model = True,
        noise_starttime = 0.0,
        noise_endtime = 60.0,
        noise_threshold = 2,
        signal_threshold = 0,
        nsigma_method = "ECDF", # method to compute nsigma
        snr_detection = False,
        snr_lowerbound = 1.0,
    )
    def __init__(self, *args, **kwargs):
        self.__dict__.update(self.defaults)
        self.update(dict(*args, **kwargs))
        
    def check_keys(self, adict={}):
        # check keys and types
        #
        
    def update(self, adict={}):
        for (key, value) in adict.items():
            self.__setitem__(key, value)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

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

        
# -

# Test
param = Parameter(nvoices=8)
print(param)
param.nsigma = "Donoho"
print(param)


