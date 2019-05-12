# -*- coding: utf-8 -*-
"""
Created on Fri May  3 11:03:57 2019

@author: 106300
"""

import numpy as np
import matplotlib.pyplot as plt

import requests
from PETRONOR_lyb import *

pi        = np.pi
E1        = 0.15
fs        = 5120.0
l         = 16384
#--------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'U3-P-0006-B',
        'Localizacion' : 'BA4', #BH3 (horizontal), BA4 (axial) y BV4 (vertical)
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-03-06T10:00:46.9988564Z',
        'FechaInicio'  : '2018-10-12T00:52:46.9988564Z',
        'NumeroTramas' : '2',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '11',#'12'
        'Hour'         : ''    
    }

    df_speed_BH3,df_SPEED_BH3,df_speed_BA4,df_SPEED_BA4,df_speed_BV4,df_SPEED_BV4 = Load_Vibration_Data_Global_Pumps(parameters)
    
    t = np.arange(l)/fs
    f = fs*np.arange(l)/(l-1)
    
        
    

    
    n_samples = 2**15
    n = np.arange(n_samples)
    y= np.sin(2*pi*n/32)
    noise = 0.1*np.random.randn(n_samples)
    num_bins = 10
    signal = y + noise
    
    
    y = df_speed_BA4.iloc[0].values
    ratio = 4
    fs2 = ratio*fs
    l2 = ratio*l
    t2 = np.arange(l2)/fs2
    f2 = fs2*np.arange(l2)/(l2-1)
    y2 = np.interp(t2, t, y)
    Y = np.abs(np.fft.fft(y*np.hanning(l)/l))
    Y2 = np.abs(np.fft.fft(y2*np.hanning(l2)/(l2)))
    # the histogram of the data
    #n, bins, patches = plt.hist(y, num_bins, normed=1, facecolor='blue', alpha=0.5)
    #plt.figure(1)
    #n, bins, patches = plt.hist(x=y, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    #plt.figure(2)
    #n, bins, patches = plt.hist(x=noise, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.figure(3)
    n, bins, patches = plt.hist(x=signal, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.figure(4)
    
    plt.plot(t,y,t2,y2)
    plt.figure(5)
    plt.plot(f,Y,f2,Y2)
    plt.show()
    

    
    
    
    
    
    