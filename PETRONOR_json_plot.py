# -*- coding: utf-8 -*-
"""
Editor de Spyder


"""
import datetime, time
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
import pandas as pd

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import os

#------------------------------------------------------------------------------
def PETROspectro(waveform, fs,titulo,ylabel,name_file,**kwargs):
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

    fig        = plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    ax1        = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    plt.plot(f[0:l_end] , cte_p*2*sptrm_P[0:l_end],'b')
    #plt.text(f[n_maxi_C ] , cte_p*2*sptrm_P[n_maxi_C ],str('{:.4f}'.format(leyenda)))
    plt.ylabel(ylabel)# +'/'   + str(RBW)+'Hz')
    plt.title(titulo  +'   ' + label)
    plt.xlabel('Hz')
    plt.grid(True)
    
    ax2     = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(t,wave_f,'r')
    plt.grid(True)
    #plt.tight_layout()
    
    fig.savefig(path_plot+str(name_file))
    fig.clear()
    return
#------------------------------------------------------------------------------
def PETRO_spectro_Module_Angle(waveform, fs,titulo,ylabel,name_file,path_plot,**kwargs):
    path_plot  = 'C://OPG106300//TRABAJO//Proyectos//Petronor-075879.1 T 20000//Trabajo//data//plots//SH4//debugging_50//'
    b, a       = signal.butter(3,2*10/fs,'highpass',analog=False)
    l          = np.size(waveform)
    #l_mitad    = int(l/2)
    l_end      = int(50 * l / 5120)
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
    FFT        = np.fft.fft(wave_f * 2    * hann)/l
    sptrm_P    = np.abs(FFT)
    angle_P    = np.angle(FFT)
    sptrm_C    = np.abs(np.fft.fft(wave_f * 1.63 * hann)/l)
        
    n_maxi_C   = np.argmax(sptrm_C)

    radio      = 2 
    int_window = 2*radio+1
    
    leyenda    = np.sum( sptrm_C[n_maxi_C-radio : n_maxi_C-radio+int_window]**2 )
    leyenda    = cte_c * np.sqrt(2*leyenda)

    
    fig        = plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    ax1        = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    plt.plot(f[0:l_end] , cte_p*2*sptrm_P[0:l_end],'b')
    #plt.text(f[n_maxi_C ] , cte_p*2*sptrm_P[n_maxi_C ],str('{:.4f}'.format(leyenda)))
    plt.ylabel(ylabel)# +'/'   + str(RBW)+'Hz')
    plt.title(titulo  +'   ' + label)
    plt.xlabel('Hz')
    plt.grid(True)
    
    ax2     = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(f[0:l_end] , angle_P[0:l_end],'b')
    plt.grid(True)
    #plt.tight_layout()
    
    fig.savefig(path_plot+str(name_file))
    fig.clear()
    return
#------------------------------------------------------------------------------
def plot_vibrationData(rootdir, assetId,MeasurePointId):
    text = []
    format = "%Y-%m-%dT%H:%M:%S"
    counter = 0
    for root, dirs, files in os.walk(rootdir):
        nfiles = 0
        for filename in files:
            fullpath = os.path.join(root, filename) 
            with open(fullpath, 'rb') as f:
                file = f.read().decode("utf-8-sig").encode("utf-8")
            res = pd.read_json(file, lines=True)
            
            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                #cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                #print(res.MeasurePointId.values[0],res.MeasurePointName.values[0] )
                #accel = np.asarray(res.Value.values[0])*cal_factor
                #speed = 9.81 * 1000*np.cumsum(accel-np.mean(accel))/5120
                fecha = res.ServerTimeStamp.values[0]
                datetime_obj = datetime.datetime.strptime(fecha[0:19],format)
                #nombre = str(datetime_obj.day)+'_'+str(datetime_obj.month)+'_'+'_'+str(datetime_obj.year)+'_''_'+str(datetime_obj.hour)+'_''_'+str(datetime_obj.minute)+'_''_'+str(datetime_obj.second)
                #print(nfiles,nombre)
                #PETROspectro(accel,5120,'Acceleration','Gs  ','Acceleration '+nombre+'.png',Detection = 'RMS')
                #PETROspectro              (speed,5120,'Velocity    ','mm/s','Velocity     '+nombre+'.png',Detection = 'RMS')
                #PETRO_spectro_Module_Angle(speed,5120,'Velocity    ','mm/s','Velocity     '+nombre+'.png',Detection = 'RMS')
                #print (assetId,datetime_obj)
                text.append(datetime_obj)
                nfiles = nfiles + 1
            if nfiles == 10: #----files per day per day
                #print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<se rompe')
                break 
            

        counter = counter +1
    
    return text


pi = np.pi
G = 9.81

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018'
month     = '10'
day       = '01'
path      = path + '\\' +month + '\\' +day
fs        = 5120.0
b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
#------------------------------------------------------------------------------


#f_in = datetime.datetime(2018, 9, 1, 0, 0)
#f_en = datetime.datetime(2010, 9, 21, 0, 0)
#df_accel         = load_vibrationDataII(path,'H4-FA-0002','SH4',f_in,f_en)  #------------en G´s-

maquina          = 'H4-FA-0002'
localizacion     = 'SH3'

text0 = plot_vibrationData(path,maquina,localizacion)