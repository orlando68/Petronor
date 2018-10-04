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
import xlrd
#import openpyxl
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
    plt.subplot(211)
    plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad])
    
    plt.plot(f[0:l_mitad],y[0:l_mitad])
    plt.plot(f[peaks], mod_sptrm[peaks],'o')
    
    #plt.axis([0, fs/2, minimo-10, maximo+10])                  
    plt.grid()
    plt.subplot(212)
    #plt.plot(window_t,window_p)
    #plt.plot(window_t,window_p,'o')
    plt.plot(f[0:l_mitad],mod_sptrm[0:l_mitad]-y[0:l_mitad])
    #plt.axis([0, fs/2, minimo-10, maximo+10])
    #plt.axis([0, fs/2, -np.pi, np.pi]) 
    plt.grid()
    
    
    
    #plt.tight_layout()
    plt.show()

    return mod_sptrm, f
   
#------------------------------------------------------------------------------    
def getKey(item):
    return item['AssetId']



path = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-072611\\data'
file = 'sh3_json3.json'
file = 'Json2018.09.25.json'


with open(path+'//'+file, "r") as read_file:
    data = json.load(read_file)

iDes= ['GA-859-S', 'H4-FA-0001', 'H4-P-0002-B', 'P-1303-A', 'P-1404-A', 'P-1404-B', 'P-1405-A']

if type(data) == dict:               #--------Solo de uno
    a= data
    #print (a['AssetId'],a['AssetName'])
else:                                #--------MAS uno
    for counter,a in enumerate(data,0):
# GA-859-S H4-FA-0001 H4-P-0002-B P-1303-A P-1404-A P-1404-B  P-1405-A   
        if a['AssetId'] == iDes[1]:
            print (counter,a['AssetId'],a['AssetName'])
            y = np.asarray(a['Value'])
            
            b           = a['Props']
            c           = b[0]
            fs          = np.float(c['Value'])
            pyt_espectro(y, fs)



pota = sorted(data, key=getKey)

for counter,a in enumerate(pota,0):
    print (counter,a['AssetId'],a['AssetName'])
    
    
"""
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