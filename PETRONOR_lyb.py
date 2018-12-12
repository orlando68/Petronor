# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:56:32 2018
Librerias para el proyecto PETRONOR
@author: 106300
"""
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

majorLocator = MultipleLocator(20)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(5)

import json
from detect_peaks import detect_peaks

import os
import pandas as pd
#------------------------------------------------------------------------------ 
def US_corr(signal,in_signal):
    out    = np.zeros(np.size(signal))
    l_in_s = np.size(in_signal)
    length = np.size(signal)-l_in_s
    print (length)
    for i in np.arange(length):

        out[i] = np.sum(signal[i:i+l_in_s] * in_signal)
    return out
#------------------------------------------------------------------------------

def PETROspectro(waveform, fs,titulo,ylabel,**kwargs):
    b, a       = signal.butter(3,2*10/fs,'highpass',analog=False)
    l          = np.size(waveform)
    l_mitad    = int(l/2)
    RBW        = fs/l
    f          = np.arange(l)/l*fs
    t          = np.arange(l)/fs

    #----------quitamos los 2 HZ del principio
    label = ''
    cte_p = 1 / np.sqrt(2)
    cte_c = 1
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
    percentage = 80
    maximo     = np.max(sptrm_P[0:l_mitad])
    TRH        = 0*np.std(sptrm_P[0:l_mitad])/3
    print (maximo/percentage,np.mean(sptrm_P[0:l_mitad]),np.std(sptrm_P[0:l_mitad]))
    #indexes    = detect_peaks(sptrm_P[0:l_mitad], mph = TRH , mpd = 5*l/fs)
    indexes, properties = find_peaks( cte_p*2*sptrm_C[0:l_mitad],height  = TRH ,prominence = 0.01 , width=1 , rel_height = 0.75)
    
    radio      = int(1*l/fs) #integro en 1Hz
    int_window = 2*radio+1
    
    leyenda = np.sum( sptrm_C[n_maxi_C-radio : n_maxi_C-radio+int_window]**2 )
    leyenda = cte_c * np.sqrt(2*leyenda)

    #print ('leyenda >>>>>>>>>>>>>>>>>>>>>>><',leyenda)
    
    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    plt.plot(f[0:l_mitad] , cte_p*2*sptrm_P[0:l_mitad],'b')
    plt.plot(f[indexes]   , cte_p*2*sptrm_P[indexes]  ,'o')
    plt.hlines(cte_p*2*TRH,0,f[l_mitad], colors='k', linestyles='dashed')
    #plt.text(f[n_maxi_C ] , cte_p*2*sptrm_P[n_maxi_C ],str('{:.4f}'.format(leyenda)))
    plt.ylabel(ylabel)# +'/'   + str(RBW)+'Hz')
    plt.title(titulo  +'   ' + label)
    plt.xlabel('Hz')
    plt.grid(True)
    
    plt.vlines(x=f[indexes], ymin=cte_p*2*sptrm_P[indexes] - properties["prominences"],ymax = cte_p*2*sptrm_P[indexes], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=f[properties["left_ips"].astype(int)],xmax=f[properties["right_ips"].astype(int)], color = "C1")
#    plt.hlines(y=properties["width_heights"], xmin=f[properties["left_bases"].astype(int)],xmax=f[properties["right_bases"].astype(int)], color = "C2")
    
    
    ax1.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(t,wave_f,'r')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return sptrm_C, f

#------------------------------------------------------------------------------

def clearance(waveform, fs,titulo,ylabel,**kwargs):
    df_out     = pd.DataFrame(data = np.ones((3,8))-2,index = ['Hz','N_freq','N_Amp'],columns = ['0.5','1.0','1.5','2.0','2.5','3.0','3.5','4.0'])
    f_1x       = 1480/60
    f_1x       = 24.7

    l          = np.size(waveform)
    l_mitad    = int(l/2)
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
    wave_f     = waveform
    sptrm_P    = np.abs(np.fft.fft(wave_f * 2    * hann)/l)
    sptrm_C    = np.abs(np.fft.fft(wave_f * 1.63 * hann)/l)
        
    n_maxi_C   = np.argmax(sptrm_C[0:l_mitad])
    maxi_C     = sptrm_C[n_maxi_C]
    
    percentage = 80
    maxi_P     = np.max(sptrm_P[0:l_mitad])
    TRH        = np.std(sptrm_P[0:l_mitad])/2
    #print (maxi_P/percentage,np.mean(sptrm_P[0:l_mitad]),np.std(sptrm_P[0:l_mitad]))
    f_1x       = f[n_maxi_C]
    
    indexes    = detect_peaks(sptrm_P[0:l_mitad], mph = TRH , mpd = 5*l/fs)
    
    D_Hz_Mod   = int(5*l/fs)
    int_wind_M = 2*D_Hz_Mod+1
    
    D_Hz_Peak  = int(1*l/fs)
    int_wind_P = 2*D_Hz_Peak+1
    Max_value  = np.sqrt(np.sum( sptrm_C[n_maxi_C-D_Hz_Peak : n_maxi_C-D_Hz_Peak+int_wind_P]**2 ))
    print ("frecuencia fundamnetal",f_1x,Max_value)
    deL_Hz = 5
    
    for k in indexes:
        if  f_1x/2 - deL_Hz   <= f[k] <= f_1x/2   + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Mod   : k-D_Hz_Mod  +int_wind_M]**2 ))
            #print ('0.5',np.round(10*f[k]/f_1x)/10,f[k], piko / Max_value)
            df_out.loc['Hz']    ['0.5'] = f[k]
            df_out.loc['N_freq']['0.5'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['0.5'] = piko / Max_value
            #print(sptrm_C[k],piko)
            #print()
        if  f_1x - deL_Hz     <= f[k] <= f_1x     + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Peak  : k-D_Hz_Peak +int_wind_P]**2 ))
            #print ('1.0',np.round(10*f[k]/f_1x)/10,f[k], piko/Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']['1.0']     = f[k]
            df_out.loc['N_freq']['1.0'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp']['1.0']  = piko / Max_value
            #print()
        if  f_1x*3/2 - deL_Hz <= f[k] <= f_1x*3/2 + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Mod   : k-D_Hz_Mod  +int_wind_M]**2 ))
            #print ('1.5',np.round(10*f[k]/f_1x)/10,f[k], piko / Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['1.5'] = f[k]
            df_out.loc['N_freq']['1.5'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['1.5'] = piko / Max_value
            #print()
            
        if  f_1x*2 - deL_Hz   <= f[k] <= f_1x*2   + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Peak  : k-D_Hz_Peak +int_wind_P]**2 ))
            #print ('2.0',np.round(10*f[k]/f_1x)/10,f[k], piko/Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['2.0'] = f[k]
            df_out.loc['N_freq']['2.0'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['2.0'] = piko / Max_value
            #print()
        if  f_1x*5/2 - deL_Hz <= f[k] <= f_1x*5/2 + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Mod   : k-D_Hz_Mod  +int_wind_M]**2 ))
            #print ('2.5',np.round(10*f[k]/f_1x)/10,f[k], piko / Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['2.5'] = f[k]
            df_out.loc['N_freq']['2.5'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['2.5'] = piko / Max_value
            #print()
        if  f_1x*3 - deL_Hz   <= f[k] <= f_1x*3   + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Peak  : k-D_Hz_Peak +int_wind_P]**2 ))
            #print ('3.0',np.round(10*f[k]/f_1x)/10,f[k], piko/Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['3.0'] = f[k]
            df_out.loc['N_freq']['3.0'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['3.0'] = piko / Max_value
            #print()
        if  f_1x*7/2 - deL_Hz <= f[k] <= f_1x*7/2 + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Mod   : k-D_Hz_Mod  +int_wind_M]**2 ))
            #print ('3.5',np.round(10*f[k]/f_1x)/10,f[k], piko / Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['3.5'] = f[k]
            df_out.loc['N_freq']['3.5'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['3.5'] = piko / Max_value
            #print()
        if  f_1x*4- deL_Hz    <= f[k] <= f_1x*4   + deL_Hz:
            piko = np.sqrt(np.sum( sptrm_C[k-D_Hz_Peak  : k-D_Hz_Peak +int_wind_P]**2 ))
            #print ('4.0',np.round(10*f[k]/f_1x)/10,f[k], piko/Max_value)
            #print(sptrm_C[k],piko)
            df_out.loc['Hz']    ['4.0'] = f[k]
            df_out.loc['N_freq']['4.0'] = np.round(10*f[k]/f_1x)/10
            df_out.loc['N_Amp'] ['4.0'] = piko / Max_value
            #print()
   
    
    radio      = int(1*l/fs) #integro en 1Hz
    int_window = 2*radio+1
    
    leyenda    = np.sum( sptrm_C[n_maxi_C-radio : n_maxi_C-radio+int_window]**2 )
    leyenda    = cte_c * np.sqrt(2*leyenda)
    """
    plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax1        = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    plt.plot(f[0:l_mitad] , cte_p*2*sptrm_P[0:l_mitad],'b')
    plt.plot(f[indexes]   , cte_p*2*sptrm_P[indexes]  ,'o')
    plt.hlines(cte_p*2*TRH,0,f[l_mitad], colors='k', linestyles='dashed')
    #plt.text(f[n_maxi_C ] , cte_p*2*sptrm_P[n_maxi_C ],str('{:.4f}'.format(leyenda)))
    plt.ylabel(ylabel)# +'/'   + str(RBW)+'Hz')
    plt.title(titulo  +'   ' + label)
    plt.xlabel('Hz')
    plt.grid(True)
    
    ax2       = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(t,wave_f,'r')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """
    return sptrm_C, f, df_out


#------------------------------------------------------------------------------



def spectro(waveform, fs,titulo,ylabel,**kwargs):
    l           = np.size(waveform)
    RBW         = fs/l
    f           = np.arange(l)/l*fs
    t           = np.arange(l)/fs
    
    #----------quitamos los 2 HZ del principio
    n_f_init = 0
    label = ''
    cte = 1
    for k in kwargs:
        if k == 'Nyquits':
            cte = kwargs.get(k)
        if k == 'Escala':
            if  kwargs.get(k) == 'RMS':
                cte = cte / np.sqrt(2)
                label = 'RMS'
            if  kwargs.get(k) == 'Peak':
                label = 'Peak'
        if k == 'f_init':
            n_f_init = np.int(kwargs.get(k)*l/fs)
            
    #sptrm             = cte*np.abs(np.fft.fft(waveform*np.hanning(l))/l)
    sptrm             = cte*np.abs(np.fft.fft(waveform)/l)
    sptrm[0:n_f_init] = 0
    l_mitad           = int(l/2)
    maximo            = np.max(sptrm[0:l_mitad])
    indexes           = detect_peaks(sptrm[0:l_mitad], mph=maximo/50, mpd=5)
    
    plt.figure()
    ax1       = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
     
    plt.plot(f[0:l_mitad],sptrm[0:l_mitad])
    plt.plot(f[0:n_f_init],sptrm[0:n_f_init],'r',linewidth=3)
    plt.plot(f[indexes],sptrm[indexes],'o')
    plt.ylabel(ylabel+label+'/'+str(RBW)+'Hz')
    plt.title(titulo)
    plt.xlabel('Hz')
    plt.grid(True)
    ax2 = plt.subplot2grid((4,4), (3,0), colspan=4, rowspan=1)
    plt.plot(t,waveform)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return 
#------------------------------------------------------------------------------
      
def getKey(item):
    return item['AssetId']
#------------------------------------------------------------------------------
def unwrap_phase(phase):
    
    end = np.size(phase)
    xu  = phase
    
    
    for i in range(2, end):
        difference = phase[i]-phase[i-1]
        if difference > pi:
            xu[i:end] = xu[i:end] - 2*pi
        else:
            if difference < -pi:
                xu[i:end] = xu[i:end] + 2*pi
    return xu


#------------------------------------------------------------------------------

def list_files(directory, extension):
    out = ([])
    files = os.listdir(directory)
    for name in files:
        [_,ext]=name.split('.')
        print(ext)
        if ext == extension:
            #print (name)
            out= np.append(out,name)
            #print(out)
    return out
#------------------------------------------------------------------------------
def load_json_file(fichero_name):
    fichero = open(fichero_name, "r")#----habro en modo txt
    texto    = fichero.read()
    l_texto  = len(texto)
    if texto[0:3] == 'ï»¿':
        texto    = texto[3:l_texto]  #----salto los 3 primeros carácteres
        #fichero = open(address, "w")
        #fichero.write(texto)
        #fichero.close               #----salvo --(precindible)
    datos = json.loads(texto)
    return datos
#------------------------------------------------------------------------------