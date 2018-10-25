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
from matplotlib.colors import colorConverter

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
                    print(fullpath)
                    #print(filename,res.SourceTimeStamp.values[0])
                    b= res.Props
                    s = b[0]
                    v= s[4]
                    cal_factor = np.float(v['Value'])
                    data.append(np.asarray(res.Value.values[0])*cal_factor*9.81)
                    #data.append(res.Value.values[0])
                    date.append(res.SourceTimeStamp.values[0])
                    break
                
    return DataFrame(data=data, index=date)
#------------------------------------------------------------------------------

pi = np.pi


path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018'
month_day = '\\10\\09'
path      = path + month_day

df_accel  = load_vibrationData(path,'H4-FA-0002')
shape     = df_accel.shape
l         = shape[1]

df_speed  = pd.DataFrame(index=df_accel.index,columns=np.arange(l),data = np.ones((24,l)))
counter   = 0
for indice in df_accel.index:
    mean                   = np.mean(df_accel.iloc[counter])
    df_speed.iloc[counter] = np.cumsum(df_accel.iloc[counter]-mean)
    counter                = counter+1

df_day    = df_speed
fs        = 5120.0
b, a      = signal.butter(5,2*10/fs,'highpass',analog=False)

fig       = plt.figure()
ax        = fig.gca(projection='3d')
verts     = []
shape     = df_day.shape
days      = np.arange(24)
l         = shape[1]

fmax      = 200
n_fmax    = np.int(l*fmax/(fs))
f         = np.arange(n_fmax)/n_fmax*fmax

color     = np.ones((n_fmax,24))
for counter,indice in enumerate(df_day.index):
    curva               = df_day.loc[indice]
    curva               = signal.filtfilt(b, a, curva)
    curva               = 2*np.abs(np.fft.fft(np.hanning(l)*curva/l))/np.sqrt(2)
    color[:,counter]    = curva[0:n_fmax]
    curva[0], curva[-1] = 0, 0
    verts.append(list(zip(f, curva[0:n_fmax])))

cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.3)
poly = PolyCollection(verts, facecolors=[cc('g')])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=days, zdir='y')
ax.view_init(10, -45)
ax.set_xlabel('Hertz')
ax.set_xlim3d(0, fmax)
ax.set_ylabel('Days')
ax.set_ylim3d(0, 24)
ax.set_zlabel('RMS value')
ax.set_zlim3d(0, np.max(color))

plt.tight_layout()
#plt.show()


#----------------------------------------------------------------
plt.figure()
color= np.asarray(color)
plt.pcolormesh(days,f,color,vmin=0, vmax=np.max(color))
#plt.pcolormesh(df_accel.values,vmin=0, vmax=np.max(color))
plt.xlabel('Days')
plt.ylabel('Hz')
plt.show()

