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

import matplotlib.pyplot as plt
import json

#------------------------------------------------------------------------------ 
def US_corr(signal,in_signal):
    out    = np.zeros(np.size(signal))
    l_in_s = np.size(in_signal)
    length = np.size(signal)-l_in_s
    print (length)
    for i in np.arange(length):

        out[i] = np.sum(signal[i:i+l_in_s] * in_signal)
    return out

   
def pyt_espectro(waveform, fs):
   
    l         = np.size(waveform)
    RBW       = fs/l
    SPAN_r    = int(np.ceil(5/RBW))
    
    media     = np.mean(waveform)
    wave_fft  = np.fft.fft(waveform)/l
    angle     = np.angle(wave_fft)
    mod_sptrm = 10*np.log10(np.abs(wave_fft)**2)
    
    wave_fft_b  = np.fft.fft(waveform-media)/l
    mod_sptrm_b = 10*np.log10(np.abs(wave_fft_b)**2)
    n_max     = np.argmax(mod_sptrm_b)
    
    window_p  = mod_sptrm[n_max-SPAN_r:n_max+(SPAN_r+1)]
    
    power     = np.std(waveform-media)
    print ("mean :", media,";  power AC",power**2)
    
    f         = np.arange(l)/l*fs
    window_t  = f[n_max-SPAN_r:n_max+(SPAN_r+1)]
    kk        = US_corr(mod_sptrm,window_p)
    maximo    = np.max(mod_sptrm_b)
    minimo    = np.min(mod_sptrm)
    peaks,_   = find_peaks(mod_sptrm_b, height=maximo-10)
    
    y         = signal.savgol_filter(mod_sptrm, 99, 1,mode='interp')
    y_angle   = signal.savgol_filter(angle, 39, 5,mode='interp')
    
    
    l_mitad   = int(l/2)
    plt.figure()
    
    plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad])
    
    plt.plot(f[0:l_mitad],y[0:l_mitad])
    #plt.plot(f[peaks], mod_sptrm[peaks],'o')
    
    
    plt.grid()
    
    
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










pi = np.pi

path = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-072611\\data'
file = 'sh3_json3.json'
file = 'Json2018.09.25.json'


with open(path+'//'+file, "r") as read_file:
    data = json.load(read_file)

counter    = 10
a          = data[counter]
print (counter,a['AssetId'],a['AssetName'])
waveform   = np.asarray(a['Value'])
 
waveform   = waveform-np.mean(waveform) 
l          = np.size(waveform)
b          = a['Props']
c          = b[0]
fs_m       = np.float(c['Value'])
tiempo     = np.arange(l)  / fs_m   

n          = 1500 #2**9
f, t, Zxx  = signal.stft(waveform, fs=fs_m, window = 'hann', nperseg=n,noverlap = n-1,nfft=1*n)

but_ord    = 4       
fc         = 1*24.8
fd         = 5
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt1 = signal.filtfilt(b, a, waveform)

fc         = 2*24.8
#fd         = 10
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt2 = signal.filtfilt(b, a, waveform)

fc         = 3*24.8
#fd         = 20
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt3 = signal.filtfilt(b, a, waveform)

fc         = 4*24.8
#fd         = 15
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt4 = signal.filtfilt(b, a, waveform)

fc         = 5*24.8
#fd         = 15
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt5 = signal.filtfilt(b, a, waveform)

fc         = 11*24.8
#fd         = 15
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt6 = signal.filtfilt(b, a, waveform)

fc         = 44*24.8
#fd         = 15
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt7 = signal.filtfilt(b, a, waveform)

fc         = 45*24.8
#fd         = 20
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt8 = signal.filtfilt(b, a, waveform)

fc         = 46*24.8
#fd         = 20
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt9 = signal.filtfilt(b, a, waveform)
fc         = 47*24.8
#fd         = 20
b, a       = signal.butter(but_ord, ([2*(fc-fd)/fs_m , 2*(fc+fd)/fs_m]), 'bandpass', analog=False)
wave_filt10 = signal.filtfilt(b, a, waveform)

plt.figure(1)
plt.plot(tiempo,waveform,linewidth=0.5)
plt.plot(tiempo,wave_filt1+wave_filt2+wave_filt3+wave_filt4+wave_filt5+wave_filt6+wave_filt7+wave_filt8+wave_filt9+wave_filt10)
plt.show()


plt.figure(2)
ax1 = plt.subplot(911)
plt.plot(tiempo,wave_filt1)
plt.plot(tiempo,np.abs(hilbert(wave_filt1)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*1*24.8/(fs_m/2)),:]),'r')
plt.subplot(912,sharex=ax1)
plt.plot(tiempo,wave_filt2)
plt.plot(tiempo,np.abs(hilbert(wave_filt2)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*2*24.8/(fs_m/2)),:]),'r')
plt.subplot(913,sharex=ax1)
plt.plot(tiempo,wave_filt3)
plt.plot(tiempo,np.abs(hilbert(wave_filt3)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*3*24.8/(fs_m/2)),:]),'r')
plt.subplot(914,sharex=ax1)
plt.plot(tiempo,wave_filt4)
plt.plot(tiempo,np.abs(hilbert(wave_filt4)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*4*24.8/(fs_m/2)),:]))
plt.subplot(915,sharex=ax1)
plt.plot(tiempo,wave_filt5,linewidth = 0.5)
plt.plot(tiempo,np.abs(hilbert(wave_filt5)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*5*24.8/(fs_m/2)),:]),'r')
plt.subplot(916,sharex=ax1)
plt.plot(tiempo,wave_filt6,linewidth = 0.5)
plt.plot(tiempo,np.abs(hilbert(wave_filt6)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*11*24.8/(fs_m/2)),:]),'r')
plt.subplot(917,sharex=ax1)
plt.plot(tiempo,wave_filt7,linewidth = 0.5)
plt.plot(tiempo,np.abs(hilbert(wave_filt7)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*44*24.8/(fs_m/2)),:]),'r')
plt.subplot(918,sharex=ax1)
plt.plot(tiempo,wave_filt8,linewidth = 0.5)
plt.plot(tiempo,np.abs(hilbert(wave_filt8)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*45*24.8/(fs_m/2)),:]),'r')
plt.subplot(919,sharex=ax1)
plt.plot(tiempo,wave_filt9,linewidth = 0.5)
plt.plot(tiempo,np.abs(hilbert(wave_filt9)))
plt.plot(t,1*np.abs(Zxx[np.int(np.size(f)*46*24.8/(fs_m/2)),:]),'r')
plt.tight_layout()
plt.subplots_adjust(hspace=0)
plt.show()


"""
xu1 =unwrap_phase(np.angle(Zxx[10, :]))
xu2 =unwrap_phase(np.angle(Zxx[15, :]))
plt.figure(3)
plt.plot(xu1)
plt.plot(xu2)
plt.show()

"""

#mod_sptrm, f = pyt_espectro(waveform, fs_m)
#maximo = np.max(mod_sptrm)
#minimo = np.min(mod_sptrm)

fmax       = 1120
fmin       = 1080

n_fmax     = np.int(np.size(f)*fmax/(fs_m/2))
n_fmin     = np.int(np.size(f)*fmin/(fs_m/2))

color_mesh = 20*np.log10(np.abs(Zxx))
Vmin = np.min(color_mesh)
Vmax = np.max(color_mesh)
plt.figure(4)
#ax1 = plt.subplot(211)
ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
#plt.pcolormesh(t, f[n_fmin:n_fmax], np.abs(Zxx[n_fmin:n_fmax,:]), vmin=0, vmax=30)
plt.pcolormesh(t, f, color_mesh, vmin=10, vmax=Vmax)

#plt.setp(ax1.get_xticklabels(), fontsize=6)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')


#ax2 = plt.subplot(212, sharex=ax1)
ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, colspan=4, rowspan=1)
plt.plot(tiempo,waveform)
#plt.title('Captured signal')
plt.ylabel('amplitude')
#plt.xlabel('Time [sec]')
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.show()


            
"""
pota = sorted(data, key=getKey)
for counter,a in enumerate(pota,0):
    print (counter,a['AssetId'],a['AssetName'])
    
    

sensor_n    = 96#int(input('numero de sensor: '))
sensor_data = data[sensor_n]
y           = np.asarray(sensor_data['Value'])
l           = np.size(y)
media       = np.mean(y)
print ('longitud de muestra :',l,'media :', media)
y           = y - media


b           = a['Props']
c           = b[0]
fs          = np.float(c['Value'])
pyt_espectro(y, fs)






f = fs*np.arange(l)/l
Y = 20 *np.log10(np.abs( np.fft.fft(y/l)))

plt.figure(1,figsize=(16, 10), dpi=80)
peaks, _ = find_peaks(Y, height=5)
plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=1)
plt.plot(y)
plt.title('Time domain')

plt.subplot2grid((4,4), (1,0), colspan=4, rowspan=3)
#plt.subplot(2, 1, 2)
plt.plot(f,Y,linewidth =0.4)
plt.plot(f[peaks],Y[peaks],'x')



plt.title('Freq domain')
plt.ylabel('Modulo')
plt.xlabel('Hz')
#plt.axis([0,fs/2,-10,60])

#plt.subplots_adjust(top=0)
plt.tight_layout()


plt.grid()
plt.show()
"""