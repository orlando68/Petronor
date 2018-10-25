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
import json

import os
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
def PETROspectro(waveform, fs,titulo,ylabel,**options):
   
    l           = np.size(waveform)
    RBW         = fs/l

    f           = np.arange(l)/l*fs
    b, a        = signal.butter(5,2*10/fs,'highpass',analog=False)
    wave_f      = signal.filtfilt(b, a, waveform)
    wave_f_fft  = np.fft.fft(wave_f*np.hanning(l))/l
    sptrm       = 2*np.abs(wave_f_fft)/np.sqrt(2) # 2 zonas nyquits y Veff
    if options.get("plot") == True:
        maximo    = np.max(sptrm)
        peaks,_   = find_peaks(sptrm,height=maximo-10)
        #y         = signal.savgol_filter(mod_sptrm, 99, 1,mode='interp')
        l_mitad   = int(l/2)
        plt.figure()
        #plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad])  
        plt.plot(f[0:l_mitad],sptrm[0:l_mitad])
        plt.ylabel(ylabel+'/'+str(RBW)+'Hz')
        plt.title(titulo)
        plt.xlabel('Hz')
        plt.grid(True)
        plt.show()

    return sptrm, f
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