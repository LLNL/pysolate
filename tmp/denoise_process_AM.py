# %%
#denoise_process.py

import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cwt
import thresh_calc as th

st = read('../../bc_v1.1/data/5014.YW.0.sp0011.DPZ', headonly=False, byteorder=None, checksize=False)
st.plot()
tr =st[0]
testdata = tr.data
delta = tr.stats.delta

# All parameters
wtype = 'morlet'
nvoices=16
nv=nvoices
# x=testdata
bandpass_blocking = "True"
scale_min = 1.0
scale_max = 200.0
block_threshold=1.0
noise_model = "True"
noise_starttime = 0.0
noise_endtime = 60.0
noise_threshold = "hard"
signal_threshold = "hard"
snr_detection = "False"
snr_lowerbound = 1.0
nsigma_method = "ECDF"

# compute the continuous wavelet transform
[Wx_new,as_new] = cwt.cwt_fw(testdata, wtype, nvoices, delta)


# %% PLOTS
# Plot the wavelet

plt.figure(figsize=(20,12.5)) #create one of the figures that must appear with the chart
plt.rcParams.update({'font.size': 26})
gs = gridspec.GridSpec(3,1)

ax1 = plt.subplot(gs[:2, :])
ax1.imshow(abs(Wx_new), aspect = 'auto') # need to fix x-axis to time and y-axis to log
ax1.set_ylabel("log10 Scale")
ax1.set_title("Time Frequency Representation (TRF)")
ax1 = plt.subplot(gs[2, :]) #create the second subplot, that MIGHT be there
ax1.plot(st[0].times(),st[0].data,color="black",linewidth=1)
ax1.set_xlim([0,max(st[0].times())])
ax1.set_xlabel("Time [s]")
plt.show()

# Wxout=Wx_new
# #plt.imshow(abs(Wxout), extent=[0, 45000, 1, 240], aspect = 'auto')
# plt.imshow(abs(Wxout), aspect = 'auto')

# %% Compute the inverse wavelet transform

x_old = cwt.cwt_iw(Wx_new, wtype, nvoices)

# Plot the inverse wavelet
t = np.arange(1., len(testdata)+1, 1)
plt.figure(figsize=(50,10))
plt.rcParams.update({'font.size': 42})
plt.subplot()
plt.plot(t, x_old, 'b', t, testdata, 'r--')
#plt.show()

# %% Do thresholding or remove noise or data

if bandpass_blocking  == "True":
    Wxb = th.blockbandpass(Wx_new, as_new, scale_min, scale_max, block_threshold)
else: Wxb = Wx_new
    
if noise_model == "True" or noise_threshold != "none" or signal_threshold != "none":
    print("Computing noise model...")
    [M, S, P, newbeg, newend] = th.noiseModel(Wxb, delta, noise_model, noise_starttime, noise_endtime, nsigma_method, snr_lowerbound)
    
if noise_model == "True" and snr_detection == "True":
    print("Applying SBR model...")
    [M_new, S_new] = th.SNRdetect(Wxb, M, newbeg, newend, snr_lowerbound)
    
if noise_threshold != "none":
    print("Compute noise Threshold of type: " + noise_threshold)
    Wx_sig = th.noiseThresh(Wxb, noise_threshold, P)
    
if signal_threshold != "none":
    print("Compute signal Threshold of type: " + signal_threshold)
    Wx_noise = th.signalThresh(Wxb, signal_threshold, P)


try:
    Wx_sig
    x_new = cwt.cwt_iw(Wx_sig, wtype, nvoices)
    plt.figure(figsize=(50,10))
    plt.rcParams.update({'font.size': 42})
    plt.subplot()
    plt.plot(t, x_new, 'b') 
except NameError:
    print("No noise threshold")    
    
try:
    Wx_noise
    x_new2 = cwt.cwt_iw(Wx_noise, wtype, nvoices)
    plt.figure(figsize=(50,10))
    plt.rcParams.update({'font.size': 42})
    plt.subplot()
    plt.plot(t, x_new2, 'r')
    
except NameError:
    print("No signal threshold")
    
    
# else: x_new = cwt.cwt_iw(Wxb, wtype, nvoices)

# # Plot the inverse wavelet
# t = np.arange(1., len(testdata)+1, 1)
# plt.figure(figsize=(50,10))
# plt.rcParams.update({'font.size': 42})
# plt.subplot()
# plt.plot(t, x_new2, 'r') 



# %%
