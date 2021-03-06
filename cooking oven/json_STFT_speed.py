# -*- coding: utf-8 -*-
"""
Editor de Spyder

Bueno,
1. me he fijado en la fase para que me ayude a posicionar armonicos, y no estaba claro,
2. he promediado la fase con el filtro de savgol => nada interesante
3. he correlado el espectro con el pico maximo para ver si asi se detectaban más calmente los 
    otros pico, pero nada de nada.
4. si resto al modulo de la FFT su promediado con el filtro de savgol, tengo un 
    espectro mas plano, mas consecuente para buscar picos. Pero no concluyo nada
"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import json
from scipy.signal import hilbert

#------------------------------------------------------------------------------ 
def US_corr(signal,in_signal):
    out    = np.zeros(np.size(signal))
    l_in_s = np.size(in_signal)
    length = np.size(signal)-l_in_s
    print (length)
    for i in np.arange(length):

        out[i] = np.sum(signal[i:i+l_in_s] * in_signal)
    return out

   
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








pi = np.pi

path = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data'
file = 'sh3_json3.json'
file = 'Json2018.09.25.json'


with open(path+'//'+file, "r") as read_file:
    data = json.load(read_file)

counter      = 10
a            = data[counter]
print (counter,a['AssetId'],a['AssetName'])
acceleration = np.asarray(a['Value'])
l            = np.size(acceleration) 

b          = a['Props']
c          = b[0]
fs_m       = np.float(c['Value'])

c          = b[4]
cali       = np.float(c['Value'])
print ('calibracion',cali)
acceleration = acceleration * cali * 9.8 
speed      = 1000*np.cumsum(acceleration)# -np.mean(acceleration))
tiempo     = np.arange(l)  / fs_m

#STFT_waterfall(acceleration, fs_m)
titulo = 'Acceleration'
ylabel = 'RMS(m/s2)/RBW'
pyt_espectro(acceleration,fs_m,titulo,ylabel,plot = True)
titulo = 'Velocity'     
ylabel = 'RMS(mm/s)/RBW'
pyt_espectro(speed       ,fs_m,titulo,ylabel,plot = True)



pota = sorted(data, key=getKey)
for counter,a in enumerate(pota,0):
    print (counter,a['AssetId'],a['AssetName'])


but_ord    = 5      
f_low      = 2000
f_up       = 0.49*fs_m
b, a       = signal.butter(but_ord, ([2*f_low/fs_m , 2*f_up/fs_m]), 'bandpass', analog=False)
gSE        = (signal.filtfilt(b, a, acceleration)) 
envelope   = np.abs(hilbert(gSE))
max_index  = np.argmax(envelope)
max_value  = envelope[max_index]
peaks, _   = find_peaks(envelope, height=0.90*max_value)
#gSE =([1,2,3,4,5,6,6,6,6,6])
plt.figure()


n, bins, patches = plt.hist(gSE, 1000,density=True, facecolor='g', alpha=0.75)
plt.show()
distance = bins[1]-bins[0]
pdf      = n *distance
probability = 0
counter = 0
while True:
    probability = probability + pdf[counter]
    counter = counter +1
    if probability >= 0.998:
        print ('counter',counter)
        break
print(bins[counter+1])



plt.figure()
plt.plot(tiempo,gSE,tiempo,envelope,tiempo,-envelope)
plt.plot(tiempo[peaks],envelope[peaks],'o')
plt.hlines(bins[counter+1],tiempo[0],tiempo[l-1])
plt.show()
