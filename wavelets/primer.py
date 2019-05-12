# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:45:01 2019

@author: 106300
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(200)
y = np.sin(2*np.pi*x/32)
n_points = 10
scales = np.arange(1,n_points)
coef, freqs=pywt.cwt(y,scales,'gaus1')
#plt.figure()
plt.matshow(coef) 
plt.show()
plt.figure()
plt.imshow(coef, extent=[-1, 1, 1, n_points], cmap='PRGn', aspect='auto', vmax=abs(coef).max(), vmin=-abs(coef).max())  
plt.show() 

import pywt
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
widths = np.arange(1, n_points)
cwtmatr, freqs = pywt.cwt(y, widths, 'gaus1')
plt.figure()
plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
plt.show() 