# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 12:14:26 2019

@author: 106300
"""

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
#npoints = 1000
#t = np.arange(npoints)
#x= 0.1*np.random.randn(npoints)  + np.sin(2*np.pi*1/64*t)
#wp = pywt.WaveletPacket(data=x, wavelet='db1', mode='symmetric')
#print(wp.maxlevel)
#plt.figure()
#plt.plot(x)
#plt.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
print(prbs(5,8))
path = "C://OPG106300//TRABAJO//Proyectos//Petronor-075879.1 T 20000//Trabajo//python//wavelets//"
mat_contents = sio.loadmat(path+'noisine.mat')
xn = mat_contents['xn'][0]


coeffs = wavedec(xn, 'sym2', level=4)

mat_contents = sio.loadmat(path+'c.mat')
coeffs_matlab = mat_contents['c'][0]

coeffs_python=np.array([])
for i in coeffs:
    
    coeffs_python = np.append(coeffs_python,i)    

plt.plot(coeffs_matlab)
plt.plot(coeffs_python)

# Show spectrogram and wavelet packet coefficients

