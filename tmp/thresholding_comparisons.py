# %%
#denoise_process.py

import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
import cwt
import thresh_calc as th


# %%
st = read('../../bc_v1.1/data/5014.YW.0.sp0011.DPZ')
st.plot()
tr =st[0]
testdata = tr.data
delta = tr.stats.delta
t = np.arange(1., len(testdata)+1, 1)
# All parameters
wtype = 'morlet'
nvoices=16
nv=nvoices
# x=testdata
bandpass_blocking = True
scale_min = 1.0
scale_max = 200.0
block_threshold=1.0
noise_model = True
noise_starttime = 0.0
noise_endtime = 60.0
noise_threshold = "none"
signal_threshold = "soft"
snr_detection = False
snr_lowerbound = 1.0
nsigma_method = "ECDF"

# compute the continuous wavelet transform
[Wx_new,as_new] = cwt.cwt_fw(testdata, wtype, nvoices, delta)

# compute the inverse cwt
wx_inverse = cwt.cwt_iw(Wx_new, wtype, nvoices)

# %%
plt.figure(figsize=(20,5))
plt.plot(wx_inverse,"k",linewidth=0.5)
trace = read("icwt.sac",format="SAC")[0]
plt.plot(trace.data,"r-",linewidth=0.5)
#plt.xlim([10000,35000])

# %%
#from copy import deepcopy

if bandpass_blocking:
    Wxb = th.blockbandpass(Wx_new, as_new, scale_min, scale_max, block_threshold)
else:
    Wxb = Wx_new.copy()
    
wxb_inverse = cwt.cwt_iw(Wxb, wtype, nvoices)
plt.figure(figsize=(20,5))
plt.plot(wxb_inverse,"k",linewidth=0.5)
trace = read("icwtblock.sac",format="SAC")[0]
plt.plot(trace.data,"r--",linewidth=0.5)
plt.xlim([10000,35000])

# %% Do thresholding or remove noise or data
# Estimate noise
if noise_model:
    print("Computing noise model...")
    [M, S, P, newbeg, newend] = th.noiseModel(Wxb, delta, noise_model, noise_starttime, noise_endtime, nsigma_method, snr_lowerbound)
    
plt.figure()
plt.plot(P,"k",label="python")
matP = np.loadtxt("P.txt")
plt.plot(matP,"r--",label="matlab")
plt.legend()

    #if noise_model and snr_detection:
#    print("Applying SBR model...")
#    [M_new, S_new] = th.SNRdetect(Wxb, M, newbeg, newend, snr_lowerbound)
    
#if noise_threshold != "none":
#    print("Compute noise Threshold of type: " + noise_threshold)
#    Wx_sig = th.noiseThresh(Wxb, noise_threshold, P)
    
#if signal_threshold != "none":
#    print("Compute signal Threshold of type: " + signal_threshold)
#    Wx_noise = th.signalThresh(Wxb, signal_threshold, P)




# %%
Wx_sig = th.noiseThresh(Wxb, "soft", P)
x_new2 = cwt.cwt_iw(Wx_sig, wtype, nvoices)

plt.figure(figsize=(20,5))
plt.plot(x_new2,"k",linewidth=0.5,label="python")
trace = read("icwtblock_noisesoft.sac",format="SAC")[0]
plt.plot(trace.data,"r--",linewidth=0.5,label="matlab")
plt.xlim([16000,22000])
plt.title("Soft thresholding to remove noise")
plt.legend()

# %%
Wx_sig = th.noiseThresh(Wxb, "hard", P)
x_new2 = cwt.cwt_iw(Wx_sig, wtype, nvoices)

plt.figure(figsize=(20,5))
plt.plot(x_new2,"k",linewidth=0.5,label="python")
trace = read("icwtblock_noisehard.sac",format="SAC")[0]
plt.plot(trace.data,"r--",linewidth=0.5,label="matlab")
plt.xlim([16000,22000])
plt.title("Hard thresholding to remove noise")
plt.legend()

# %%
Wx_sig = th.signalThresh(Wxb, "hard", P)
x_new2 = cwt.cwt_iw(Wx_sig, wtype, nvoices)

plt.figure(figsize=(20,5))
plt.plot(x_new2,"k",linewidth=0.5,label="python")
trace = read("icwtblock_signalhard.sac",format="SAC")[0]
plt.plot(trace.data,"r--",linewidth=0.5,label="matlab")
plt.xlim([10000,22000])
plt.title("Hard thresholding to remove signal")
plt.legend()

# %%
Wx_sig = th.signalThresh(Wxb, "soft", P)
x_new2 = cwt.cwt_iw(Wx_sig, wtype, nvoices)

plt.figure(figsize=(20,5))
plt.plot(x_new2,"k",linewidth=0.5,label="python")
trace = read("icwtblock_signalsoft.sac",format="SAC")[0]
plt.plot(trace.data,"r--",linewidth=0.5,label="matlab")
plt.xlim([10000,22000])
plt.title("Soft thresholding to remove signal")
plt.legend()

# %%
# Matlab
#import scipy.io as sio
#mat_contents = sio.loadmat("../../bc_v1.1/softnoise_freq.mat")
#frommat = cwt.cwt_iw(mat_contents['Wx_new'], wtype, nvoices)

#trace = read("icwtblock_noisesoft.sac",format="SAC")[0]

#W_test = np.abs(Wxb)
#Wx_new = np.sign(Wxb).T * (W_test.T-P) * (P<W_test.T)
#x_new2 = cwt.cwt_iw(Wx_new.T, wtype, nvoices)
#plt.figure(figsize=(20,5))
#plt.plot(frommat,"r--",linewidth=0.5,label="matlab")
#plt.plot(trace,"k",linewidth=0.5)
#plt.plot(x_new2,"k",linewidth=0.5,label="python")
#plt.xlim([16000,22000])
