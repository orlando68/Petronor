# -*- coding: utf-8 -*-
"""
Editor de Spyder


"""
import datetime, time
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import pandas as pd
from pandas import DataFrame

import matplotlib
#matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import os

#------------------------------------------------------------------------------
def PETROspectro(waveform, fs,titulo,ylabel,**kwargs):
    path_plot  = 'C://OPG106300//TRABAJO//Proyectos//Petronor-075879.1 T 20000//Trabajo//data//plots//Petronor//'
    b, a       = signal.butter(3,2*10/fs,'highpass',analog=False)
    l          = np.size(waveform)
    #l_mitad    = int(l/2)
    l_end      = int(1000 * l / 5120)
    RBW        = fs/l
    f          = np.arange(l)/l*fs
    t          = np.arange(l)/fs

    #----------quitamos los 2 HZ del principio
    label      = ''
    cte_p      = 1 / np.sqrt(2)
    cte_c      = 1
    for k in kwargs:
        if k == 'Detection':
            if  kwargs.get(k) == 'RMS':
                cte_p = 1 / np.sqrt(2)
                cte_c = 1
                label = 'RMS'
            if  kwargs.get(k) == 'Peak':
                label = 'Peak'
                cte_p = 1
                cte_c = np.sqrt(2)

    hann       = np.hanning(l)
    wave_f     = signal.filtfilt(b, a, waveform)
    sptrm_P    = np.abs(np.fft.fft(wave_f * 2    * hann)/l)
    sptrm_C    = np.abs(np.fft.fft(wave_f * 1.63 * hann)/l)
        
    n_maxi_C   = np.argmax(sptrm_C)

    radio      = 2 
    int_window = 2*radio+1
    
    leyenda    = np.sum( sptrm_C[n_maxi_C-radio : n_maxi_C-radio+int_window]**2 )
    leyenda    = cte_c * np.sqrt(2*leyenda)

    #fig        = plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    plt.figure()
    ax1        = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    #plt.plot(f[0:l_end] , cte_p*2*sptrm_P[0:l_end],'b')
    #plt.text(f[n_maxi_C ] , cte_p*2*sptrm_P[n_maxi_C ],str('{:.4f}'.format(leyenda)))
    plt.plot(t,wave_f,'r')
    plt.ylabel(ylabel)# +'/'   + str(RBW)+'Hz')
    plt.title(titulo  +'   ' + label)
    plt.xlabel('Hz')
    plt.grid(True)
    
    ax2     = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(t,wave_f,'r')
    plt.grid(True)
    #plt.tight_layout()
    plt.show()
    
    return

def plot_vibrationData(rootdir, assetId,MeasurePointId):
    data = []
    date = []
    #t          = np.arange(l)/fs
    fc   = 24.8
    delta = 3
    wn = 2*np.array([fc-delta,fc+delta])/fs
    b, a       = signal.butter(3,wn,'bandpass',analog=False)
    l = 16384
    
    start = 1*int(0.25/Ts)
    end   = 1*int((0.25 +0.2) /Ts)
    counter = 0
    for root, dirs, files in os.walk(rootdir):
        nfiles = 0
        for filename in files:
            fullpath = os.path.join(root, filename) 
            with open(fullpath, 'rb') as f:
                file = f.read().decode("utf-8-sig").encode("utf-8")
            res = pd.read_json(file, lines=True)
            
            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                #print(res.MeasurePointId.values[0],res.MeasurePointName.values[0] )
                accel = np.asarray(res.Value.values[0])*cal_factor
                speed = 9.81 * 1000*np.cumsum(accel-np.mean(accel))/5120
                speed = signal.filtfilt(b, a, speed)
                speed = speed[start:16384-end]
                data.append(speed)
                fecha = res.ServerTimeStamp.values[0]
                print(fecha[14:16])
                #print(fecha[17:40])
                #print(np.float(fecha[17:40].split('Z')[0]))
                segundos = np.float(fecha[14:16])*60 + np.float(fecha[17:40].split('Z')[0]) +0.25
                print(segundos)

                date.append(segundos)
                
                nfiles = nfiles + 1
            if nfiles == 10: #----files per day per day
                #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<se rompe')
                break 
            

        counter = counter +1
    df_out     = DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out


pi = np.pi
G = 9.81

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018'
month     = '10'
day       = '01\\00'
path      = path + '\\' +month + '\\' +day
fs        = 5120.0
b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
#------------------------------------------------------------------------------


#f_in = datetime.datetime(2018, 9, 1, 0, 0)
#f_en = datetime.datetime(2010, 9, 21, 0, 0)
#df_accel         = load_vibrationDataII(path,'H4-FA-0002','SH4',f_in,f_en)  #------------en G´s-

maquina          = 'H4-FA-0002'
localizacion     = 'SH4'

Ts = 1/5120.0

df_velocity_BPF = plot_vibrationData(path,maquina,localizacion)

harmonic= np.array([])



l = df_velocity_BPF.shape[1]
t1 = np.arange(l)/5120

frec = 24.775
phase = 1.858
seno = 6.45*np.sin(2*np.pi*frec*t1+phase)
#plt.plot(df_velocity_BPF.iloc[0].values)
#PETROspectro(df_velocity_BPF.iloc[0].values,fs,'Velocidad con HPF','RMS' ,Detection = 'RMS')
plt.figure();
kk = int(l/1)
plt.plot(t1[0:kk],df_velocity_BPF.iloc[0].values[0:kk],t1[0:kk],seno[0:kk]);
plt.grid(True)
plt.show()


dt = df_velocity_BPF.index[1]-df_velocity_BPF.index[0]
t2 = t1+ dt
seno2 = 4.82*np.sin(2*np.pi*frec*t2+phase)
plt.figure()
plt.plot(t2[0:kk],df_velocity_BPF.iloc[1].values[0:kk],t2[0:kk],seno2[0:kk]);
plt.grid(True)
plt.show()


t3= np.arange(int(np.round(dt/Ts)+l))*Ts

seno3 = 4.82*np.sin(2*np.pi*frec*t3+phase)
plt.figure()
plt.plot(t3,seno3,t1[0:kk],df_velocity_BPF.iloc[0].values,t2,df_velocity_BPF.iloc[1].values)
plt.grid(True)
plt.show()
