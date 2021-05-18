# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: obspy
#     language: python
#     name: obspy
# ---

# ### pyBlockSeis - Block Choice Seismic Analysis in Python

# +
import sys
sys.path.append("src")

import numpy as np
import pyblockseis as bcs
import matplotlib.pyplot as plt

from obspy.core import read, Stream

# %matplotlib inline
# -

# ### Dataset
#
#
# **Nuclear explosions from Bondar et al. 2005 (Figure 5)**
#
# | EVID | ORID | NAME | DATE | MSEC | LON | LAT | DEPTH | ML |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | 511553 | 3962829 | Clairette | 1981-02-05T18:00:00 | 120 | -116.033 | 37.011 | 0.354 | 3.2 |
# | 514907 | 3962836 | Havarti   | 1981-08-05T13:41:00 | 90  | -116.036 | 37.154 | 0.200 | 2.8 |
# | 515492 | 3962838 | Trebbiano | 1981-09-04T15:00:00 | 100 | -116.067 | 37.160 | 0.294 | 3.8 |
# | 578449 | 3962949 | Tahoka    | 1987-08-13T14:00:00 | 90  | -116.046 | 37.061 | 0.639 | 5.5 |

# +
# Pre-processing
time_before = 0 # Cut data relative to origin
time_after = 200
sampling_rate = 40

# CWT operations
params = bcs.Parameter(
    scale_min=1.,
    scale_max=100.,
    bandpass_blocking=True,
    block_threshold=5.0,
    noise_endtime=60.,
    noise_threshold="soft",
)

# Create a block object for each event
blocks = []
for evid in [511553, 514907, 515492, 578449]:
    
    # Do some pre-processing to the data
    st = read("testdata/bondar_2015_data/%d/waveforms/*.v"%evid, format="SAC")
    st.resample(40.0)
    for tr in st:
        # Find origin time from SAC header
        origin_time = tr.stats.starttime + (tr.stats.sac.o-tr.stats.sac.b)
        tr.stats.origin_time = origin_time
        tr.trim(origin_time+time_before, origin_time+time_after)
       
    # Initalize and run block object
    block = bcs.read(params=params, data=st)
    block.run()
    blocks.append(block)

# -

for i, block in enumerate(blocks):
    print("Available data for event %d"%i)
    print(block.get_station_list())
    print(block.tags)

# ### Comparing CWT and the noise model
#
# You can select data that matches the station criteria:
# - network code
# - station code
# - location code
# - channel code
# - component code
#
# Example below will plot the results of Elko (ELK) for the first event in the table, Clairette.

# +
# Get the noise model
# Depending on the processing it can be the "input" waveforms or "band_rejected"
block = blocks[0]
tag = block.noise_model_tag 
print("Noise model estimated from '%s' data."%tag)

# Select the CWT for ELK will return 4 WaveletCollection objects
# for each event
wave = block.get_wavelets(tag).select(station="ELK")[0]

# You can select the seismograms the same way
tr = block.waveforms.data[tag].select(station="ELK")[0]

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
# -

# ### Denoised data

block.plot("noise_removed")


