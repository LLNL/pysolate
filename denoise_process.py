#denoise_process.py

import numpy as np
from obspy.core import read
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cwt

st = read('/Users/aguiarmoya1/Research/Denoising/bc_v1.1/data/5014.YW.0.sp0011.DPZ', headonly=False, byteorder=None, checksize=False)
st.plot()
tr =st[0]
testdata = tr.data
delta = tr.stats.delta

# All parameters
wtype = 'morlet'
nvoices=16
# nv=nvoices
# x=testdata
# nbpblck = 1
# scale_min = 1.0
# scale_max = 200.0
# bthresh=0.0
# nnbnd = 1
# tstrn = 0.0
# tfinn = 60.0
# noisethresh = 2
# signalthresh = 0
# nsig = -2.0
# nsnr = 0
# nsnrlb = 1.0

# compute the continuous wavelet transform
[Wx_new,as_new] = cwt.cwt_fw(testdata, wtype, nvoices, delta)

# compute the inverse wavelet transform
x_new = cwt.cwt_iw(Wx_new, wtype, nvoices)

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

# %%
# Plot the inverse wavelet
t = np.arange(1., len(testdata)+1, 1)
plt.figure(figsize=(50,10))
plt.rcParams.update({'font.size': 42})
plt.subplot()
plt.plot(t, x_new, 'b', t, testdata, 'r--')
#plt.show()