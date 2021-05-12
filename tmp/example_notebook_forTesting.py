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

# #### Software package name
# pyBlockSeis - Block Choice Seismic Analysis in Python
#
# 1. Input data: time series, delta
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
# 4. Example of computation in parallel https://deepgraph.readthedocs.io/en/latest/tutorials/pairwise_correlations.html
#
# 5. Defaults in source code:
#         wave_type = "morlet",
#         nvoices = 16,
#         bandpass_blocking = True,
#         scale_min = 1.0,
#         scale_max = 200.0,
#         block_threshold = 0.0,
#         estimate_noise = True, # params after require estimate_noise = True
#         noise_starttime = 0.0,
#         noise_endtime = 60.0,
#         noise_threshold = None,
#         signal_threshold = None,
#         nsigma_method = "ECDF", # method to compute nsigma
#         snr_detection = False,
#         snr_lowerbound = 1.0,
#
# ### Acknowledgements
# - Python adapation of the Matlab software Block Choice Seismic Analysis (BCseis, version 1.1) by Charles A. Langston and S. Mostafa Mousavi.
# - Forward and inverse CWTs functions based on the Synchrosqueezing Toolbox V1.21 by Eugene Brevdo and Gaurav Thakur.  (https://github.com/ebrevdo/synchrosqueezing).

# +
import sys
sys.path.append("src")

import pyblockseis as bcs
from obspy.core import read
import matplotlib.pyplot as plt 

import timeit

# +
# Read example data from BCseis
#sacfile = "/Users/aguiarmoya1/Research/Denoising/bc_v1.1/data/5014.YW.0.sp0011.DPZ"
sacfile = "/Users/chiang4/Work/NNSA/LYNM/denoiser/bc_v1.1/data/5014.YW.0.sp0011.DPZ" # multiple traces
st = read(sacfile)

start = timeit.timeit()
# Use the default values to process time series
# Refer to :class:pyblockseis.Parameter docstring for details
params = bcs.Parameter(block_threshold=1.0, noise_threshold="hard", signal_threshold="hard", 
                       bandpass_blocking = True)

# Initalize the block processing module
block = bcs.Block(choice=params, data=st)

# Run the denoiser
block.run()
end = timeit.timeit()
print("Run took %.4f seconds"%(end - start))

# Plot results
block.plot("input")
block.plot("band_rejected")
#block.plot("noise_removed")
#block.plot("signal_removed")
# -

tr = block.data[0].copy()
print(tr.wavelet.icwt)
tr.data = tr.wavelet.icwt["noise_removed"]
tr.write("noise_removed.sac",format="SAC")



import numpy as np
plt.figure()
plt.title("Noise model")
plt.plot(block.data[0].wavelet.P,"k",label="P-python")
plt.plot(block.data[0].wavelet.M,"k--",label="mean-python")
matP = np.loadtxt("tmp/P.txt")
plt.plot(matP,"r",label="P-matlab")
plt.legend()
