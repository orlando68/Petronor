# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 16:48:44 2019

@author: 106300
"""

import pywt
import pywt.data
import numpy as np
import matplotlib.pyplot as plt

x = pywt.data.ecg()
plt.plot(x)
plt.legend(['Original signal'])
plt.show()
w = pywt.Wavelet('sym5')
plt.plot(w.dec_lo)
coeffs = pywt.wavedec(x, w, level=6)