# -*- coding: utf-8 -*-
"""
Editor de Spyder


"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from PETRONOR_lyb import load_json_file
from PETRONOR_lyb import PETROspectro

import pandas as pd
import os
from pandas import DataFrame
#------------------------------------------------------------------------------
def load_vibrationData(rootdir, assetId):
    data = []
    date = []
    array_days = np.array([])
    for root, dirs, files in os.walk(rootdir):
        #print (dirs)
        for filename in files:
            if filename.endswith((".json")):
                fullpath = os.path.join(root, filename) 
                #print(fullpath)
                # read the entire file into a python array
                with open(fullpath, 'rb') as f:
                    file = f.read().decode("utf-8-sig").encode("utf-8")
                res = pd.read_json(file, lines=True)
                if res.AssetId.values[0] == assetId:
                    
                    array_days = np.append(array_days,res.Value.values[0])
                    b= res.Props
                    s = b[0]
                    v= s[4]
                    cal_factor = np.float(v['Value'])
                    data.append(np.asarray(res.Value.values[0])*cal_factor*9.81)
                    date.append(res.SourceTimeStamp.values[0])
                    break
                
    return DataFrame(data=data, index=date), array_days
#------------------------------------------------------------------------------

pi = np.pi


path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018'
month_day = '\\10\\10'
path      = path + month_day

kk, mes = load_vibrationData(path,'H4-FA-0002')

n       = 16384
fs      = 5120.0
b, a    = signal.butter(5,2*10/fs,'highpass',analog=False)
mes     = signal.filtfilt(b, a, mes)

f, t, Zxx = signal.stft(mes, fs=fs, window = 'hann', nperseg=n,noverlap = 0,nfft=1*n)
t         = t/3.2
Z         = color_mesh = np.abs(Zxx)
plt.pcolormesh(t, f, color_mesh,vmin=0, vmax=np.max(color_mesh))
plt.show()
"""
fmax       = 500
fmin       = 0
n_fmax     = np.int(np.size(f)*fmax/(fs/2))
n_fmin     = np.int(np.size(f)*fmin/(fs/2))
#plt.pcolormesh(t, f[n_fmin:n_fmax], np.abs(Zxx[n_fmin:n_fmax,:]), vmin=0, vmax=30)

fig     = plt.figure()
ax      = fig.gca(projection='3d')
X, Y    = np.meshgrid(t, f[0:n_fmax])
surf    = ax.plot_surface(X, Y, Z[0:n_fmax,:], cmap=cm.coolwarm,linewidth=0, antialiased=False)
plt.show()
"""



fig = plt.figure()
ax = fig.gca(projection='3d')
verts = []

shape=kk.shape
zs = np.arange(24)
l = shape[1]
f = np.arange(l)/l*fs
fmax       = 500
n_fmax     = np.int(l*fmax/(fs))
for counter,indice in enumerate(kk.index):
    curva = kk.loc[indice]
    curva = signal.filtfilt(b, a, curva)
    curva = np.abs(np.fft.fft(curva/l))
    curva[0], curva[-1] = 0, 0
    verts.append(list(zip(f[0:n_fmax], curva[0:n_fmax])))


poly = PolyCollection(verts)
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, fmax)
ax.set_ylabel('Y')
ax.set_ylim3d(0, 24)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 0.5)

plt.show()
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
plt.show()
"""