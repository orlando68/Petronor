# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:05:42 2019

@author: 106300
"""

import numpy as np
import pandas as pd
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pandas import DataFrame




#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#------------------------------------------------------------------------------

file   = 'Spectral_FP_Local_DB_H4-FA-0002_SH4__12_2018.xlsx'
df_in  = pd.read_excel(Path_out+file, index_col=0)
x1     = df_in.loc[:]['RMS 1.0'].values
x1[x1 !=0]
l      = np.size(x1)
mean   = np.sum(x1)/l
std    = np.sqrt( np.sum( (x1-mean)**2 ) /l)

mean  = 1.35
std   = 1.15
x     = np.random.lognormal(mean,std,100000)
#x = np.random.randn(10000)
n, bins, patches = plt.hist(x, 1000, normed=True, facecolor='g', alpha=0.75)

print (np.mean(x), np.exp(mean+ (std**2 )/2))
print (np.std(x) , np.sqrt(np.exp(2*mean+std**2) * ( np.exp(std**2)-1 )) )

#--------------------------------------------------------------------
mean  = 1.35
std   = 1.15

l_mean =  2 * np.log(mean) - np.log(mean**2+std**2)/2
l_std  = np.sqrt(-2 * np.log(mean) + np.log(mean**2+std**2))
x     = np.random.lognormal(l_mean,l_std,100000)
#x = np.random.randn(10000)
n, bins, patches = plt.hist(x, 1000, normed=True, facecolor='g', alpha=0.75)

print (np.mean(x))
print (np.std(x) )




plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([0, 30, 0, 0.04])
plt.grid(True)
plt.show()