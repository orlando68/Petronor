# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 12:23:09 2019

@author: 106300
"""
import pywt
from scipy.signal import chirp, spectrogram
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec
import scipy.io as sio

def prbs(nbits,nsamples):
    bit = np.ones(nsamples)
    out = np.zeros(nbits*nsamples)
    for i in range(nbits):
        out[i*nsamples:nsamples*(i+1)] = np.sign( np.random.randn(1) +1e-100)*bit        
    return out

fs  = 1

#t = np.linspace(0, 10, 5001)
#w = chirp(t, f0=1, f1=6, t1=10, method='linear')
#-------------------------------White gaussian noise
#points = 2**10
#w = np.random.randn(npoints)
#t = np.arange(npoints)


#------------------------------BFSK
bits    = prbs(100,128*2)
npoints = np.size(bits)
n       = np.arange(npoints)

#w        = np.sin(2*np.pi*t*(4+1*bits)/64)
fi  = 1/16
df  = 1/16/16+0.0001*0
w   = np.sin(2*np.pi*n*(fi+df*bits)/fs)

coeffs   = wavedec(w, 'coif3', level=4)
n_plots  = 2* (np.size(coeffs)+1)
fig, ax1 = plt.subplots(num=None, figsize=(18, 18), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(n_plots,1,1)
plt.title("Linear Chirp, f(0)=6, f(10)=1")
plt.plot(n,w,n,bits)
plt.xlabel('t (sec)')

plt.subplot(n_plots,1,2)
f     = np.arange(npoints)/(npoints-1)
spect = 20*np.log10(np.abs(np.fft.fft(w)/npoints))
plt.plot   (f,spect)
plt.axis([0,0.5,-80,np.max(spect)])

n_max = n[-1]

for counter,i in enumerate(coeffs):
    l1 = np.size(i)
    
    plt.subplot(n_plots,1,2*(counter+1)+1)
    n1 = n_max*np.arange (l1)/(l1-1)
    plt.plot(n1,i,n,bits)
    plt.grid(False)
    fs1 = l1/npoints
    f1  = fs1* np.arange(l1)/(l1-1)
    spec = 20*np.log10(np.abs(np.fft.fft(i)/l1))
    plt.subplot(n_plots,1,2*(counter+1)+2)
    plt.plot(f1[0:int(l1/2)],spec[0:int(l1/2)])
    plt.axis([0,0.5,np.min(spec),np.max(spec)])
    plt.grid(True)
    

print(fs)
