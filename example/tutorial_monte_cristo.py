# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import obspy
import obspy.clients.fdsn
import pysolate
import numpy as np
import matplotlib.pyplot as plt

# ### Available wavelet processing parameters
#
# * **wave_type**: wavelet filter type, options are ``"morlet"``, ``"shannon"``, ``"mhat"``, ``"hhat"``.
# Default is ``"morlet"``.
#
# * **nvoices**: number of voices, or the sampling of CWT in scale.
# Higher number of voices give finer resolution. Default is ``16``.
#
# * **bandpass_blocking**: Default value ``True`` will apply a band rejection filter where
# wavelet coefficients are modified over a scale bandpass.
#
# * **scale_min**: minimum time scale for bandpass blocking. Default is ``1``.
#
# * **scale_max**: maximum time scale for bandpass blocking. Default is ``200``.
#
# * **block_threshhold:** percent amplitude adjustment to the wavelet coefficients within
# ``scale_min`` and ``scale_max``. For example a threshold of 5% means the wavelet cofficients
# in the band will be multipled by 0.05. Default is ``0``.
#
# * **estimate_noise**: flag to compute the noise model, default is ``True``.
#
# * **noise_starttime**: noise start time, default is ``0``.
#
# * **noise_endtime**: noise end time, default is ``60``.
#
# * **noise_threshold**: type of noise thresholding to be applied, the options are
# ``"hard"`` for hard thresholding and ``"soft"`` for soft thresholding. Default is ``None``.
#
# * **signal_threshold**: type of signal thresholding to be appied, the options are
# ``"hard"`` for hard thresholding, and ``"soft"`` for soft thresholding. Default is ``None``.
#
# * **nsigma_method**: method to determine the number of standard deviations for block thresholding.
# ``"donoho"`` for Donoho's Threshold criterion and ``"ECDF"`` for empirical cumulative probability
# distribution method. You can also specify the number of standard deviations by entering a number.
# None ECDF method assumes Gaussian statistic. The default method ``"ECDF"`` is recommended.
#
# * **snr_detection**: Flag to apply the SNR detection method, default is ``False``. If ``True`` it
# will be applied before hard thresholding.
#
# * **snr_lowerbound**: Noise level percent lower bound. Default is ``1.0``.
#

# +
# Example event
# origin_time = 2020-05-19T11:51:45.99
# evla = 38.1247
# evlo = -117.8002
# depth = 7.7
# ml = 1.5
# author = "NEIC"

client = obspy.clients.fdsn.Client("IRIS")

# Get event
catalog = client.get_events(
    minlongitude=-118, maxlongitude=-117,
    minlatitude=38, maxlatitude=39,
    starttime=obspy.UTCDateTime("2020-05-19T11:50:00"),
    endtime=obspy.UTCDateTime("2020-05-19T11:55:00")
)

# Get waveforms
origin_time = obspy.UTCDateTime("2020-05-19T11:51:45.99")
network = "NN"
station = "LHV"
location = ""
channel = "HHZ"
time_before = 60 # pre-event noise
time_after = 125
t1 = origin_time - time_before
t2 = origin_time + time_after
st = client.get_waveforms(network, station, location, channel, t1, t2)

# Pre-processing
st.detrend()
st.decimate(factor=4, strict_length=False)

# +
# CWT operations
params = pysolate.Parameter(
    scale_min=1.,
    scale_max=st[0].stats.endtime-st[0].stats.starttime,
    bandpass_blocking=True,
    block_threshold=2.0,
    noise_endtime=50,
    noise_threshold="soft",
)

# Initalize and run pysolate
block = pysolate.read(params=params, data=st)
block.run()

print("Available data")
print(block.get_station_list())
print(block.tags)
# -

# ### Comparing CWT and the noise model
#
# You can select data that matches the station criteria:
# - network code
# - station code
# - location code
# - channel code
# - component code

# +
# Get the noise model
# Depending on the processing it can be the "input" waveforms or "band_rejected"
tag = block.noise_model_tag
print("Noise model estimated from '%s' data."%tag)

# Select the CWT for LHV
wave = block.get_wavelets(tag).select(station="LHV")[0]

# You can select the seismograms the same way
tr = block.waveforms.data[tag].select(station="LHV")[0]

# Select the first event
scales = np.log10(wave.scales)
threshold = wave.noise_model.P
mean = wave.noise_model.M

fig = plt.figure(figsize=(12,5))
ax1 = plt.subplot2grid((2,3),(0,0))
ax1.plot(threshold, scales,"k",label="threshold")
ax1.plot(mean, scales, "g", label="mean")
ax1.invert_xaxis()
ax1.invert_yaxis()
ax1.legend()
ax1.set_title("Noise model")

# Plot scalogram
extent = [time_before, time_after, max(scales), min(scales)]
ax2 = plt.subplot2grid((2,3),(0,1), colspan=2)
ax2.imshow(abs(wave.coefs), extent=extent, aspect="auto")
ax2.set_title("Scalogram: %s data"%tag)

# Plot waveform
ax3 = plt.subplot2grid((2,3),(1,1), colspan=2)
ax3.plot(tr.times(), tr.data, color="k", linewidth=0.6)

plt.show()
# -

# ### Comparing Input Data and Denoised data

block.plot("input", station="LHV")
block.plot("noise_removed", station="LHV")

# ### Write waveform data to file

# +
# Use the write function to save your waveforms to any ObsPy supported
# formats or numpy npz format.

tag = "noise_removed" # save the denoised data
output = "waveforms"

# SAC format
block.write(tag, output=output, format="SAC")

# Numpy npz format (compressed binary)
block.write(tag, output=output, format="npz")
# -

# ### Write CWT to file

# +
# Save the wavelet transforms, only numpy npz is supported right now

tag = "noise_removed" # save the denoised CWT
output = "cwt"

# Numpy npz format (compressed binary)
block.write(tag, output=output, format="npz")
