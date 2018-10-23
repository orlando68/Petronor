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
    
    wave_fft    = np.fft.fft(waveform)/l
    mod_sptrm   = np.abs(wave_fft)/np.sqrt(2)

    
    f           = np.arange(l)/l*fs
    b, a        = signal.butter(5,2*10/fs,'highpass',analog=False)
    wave_f      = signal.filtfilt(b, a, waveform)
    wave_f_fft  = np.fft.fft(wave_f)/l
    sptrm       = np.abs(wave_f_fft)/np.sqrt(2)
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
    
   
def pyt_espectro(waveform, fs,titulo,ylabel,**options):
   
    l           = np.size(waveform)  
    media       = np.mean(waveform)
    
    wave_fft    = np.fft.fft(waveform)/l
    mod_sptrm   = np.abs(wave_fft)/np.sqrt(2)
    
    wave_fft_b  = np.fft.fft(waveform-media)/l
    mod_sptrm_b = np.abs(wave_fft_b)/np.sqrt(2)
    
    f           = np.arange(l)/l*fs
    f_rpm       = f * 60
    RBW         = fs/l
    if options.get("plot") == True:
        maximo    = np.max(mod_sptrm_b)
        peaks,_   = find_peaks(mod_sptrm_b, height=maximo-10)
        #y         = signal.savgol_filter(mod_sptrm, 99, 1,mode='interp')
        l_mitad   = int(l/2)
        plt.figure()
        ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
        plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad])
        
        plt.ylabel(ylabel)
        plt.title(titulo+' RBW='+str(RBW)+'Hz')
        plt.grid(True)
        ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, colspan=4, rowspan=1)
        plt.semilogy(f[0:l_mitad],mod_sptrm[0:l_mitad],'r')
        plt.ylabel(ylabel)
        plt.xlabel('Hz')
        plt.grid(True)
        plt.show()

    return mod_sptrm, f
  
def pyt_espectro_velocity(waveform, fs,fx):
   
    l           = np.size(waveform)  
    media       = np.mean(waveform)
   
    
    wave_fft    = np.fft.fft(waveform)/l
    mod_sptrm   = np.abs(wave_fft)/np.sqrt(2)
    
    wave_fft_b  = np.fft.fft(waveform-media)/l
    mod_sptrm_b = np.abs(wave_fft_b)/np.sqrt(2)
    
    f           = np.arange(l)/l*fs
    RBW         = fs/l
   
    maximo    = np.max(mod_sptrm_b)
    peaks,_   = find_peaks(mod_sptrm_b, height=maximo-10)
    #y         = signal.savgol_filter(mod_sptrm, 99, 1,mode='interp')
    l_mitad   = int(l/2)
    plt.figure()
    ax1       = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad])
    f1        = np.int(np.size(f)*f_DC/(fs/2))
    plt.plt
    plt.ylabel('RMS(mm/s)/RBW')
    plt.title('Velocity RBW='+str(RBW)+'Hz')
    plt.grid(True)
    ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, colspan=4, rowspan=1)
    plt.semilogy(f[0:l_mitad],mod_sptrm[0:l_mitad],'r')
    plt.ylabel(ylabel)
    plt.xlabel('Hz')
    plt.grid(True)
    plt.show()

    return mod_sptrm, f
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
def STFT_waterfall(waveform,f_sampling):
    n          = 2**9
    f, t, Zxx  = signal.stft(waveform, fs=f_sampling, window = 'hann', nperseg=n,noverlap = n-1,nfft=1*n)
    
#    fmax       = 1120
#    fmin       = 1080
#    n_fmax     = np.int(np.size(f)*fmax/(f_sampling/2))
#    n_fmin     = np.int(np.size(f)*fmin/(f_sampling/2))
    
    f_DC       = 10
    n_f_DC     = np.int(np.size(f)*f_DC/(f_sampling/2))
    color_mesh = np.abs(Zxx)
    Vmin = np.min(color_mesh)
    Vmax = np.max(color_mesh[n_f_DC:,:]) #tomamos el valor mas grande que no sea DC
    plt.figure()
    #ax1 = plt.subplot(211)
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    #plt.pcolormesh(t, f[n_fmin:n_fmax], np.abs(Zxx[n_fmin:n_fmax,:]), vmin=0, vmax=30)
    plt.pcolormesh(t, f, color_mesh, vmin=Vmin, vmax=Vmax)
    
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    
    
    #ax2 = plt.subplot(212, sharex=ax1)
    ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, colspan=4, rowspan=1)
    plt.plot(tiempo,acceleration)
    #plt.title('Captured signal')
    plt.ylabel('amplitude')
    #plt.xlabel('Time [sec]')
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    
    return f, t, Zxx

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