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

from PETRONOR_lyb import load_json_file
from PETRONOR_lyb import PETROspectro



pi = np.pi

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\10\\10'
json_file = 'ab98949e-17e8-46eb-6d20-165279207208_amplitude.json'

data      = load_json_file(path+'\\'+json_file)
print (data["AssetId"])        


accel      = np.asarray(data['Value'])
l           = np.size(accel) 

b          = data['Props']
c          = b[0]
fs_m       = np.float(c['Value'])

c          = b[4]
cal_factor = np.float(c['Value'])
accel      = accel * cal_factor * 9.81 

speed      = 1000*np.cumsum(accel -np.mean(accel))
tiempo     = np.arange(l)  / fs_m
#
##STFT_waterfall(acceleration, fs_m)
titulo     = 'Acceleration ' +data["AssetName"]+'('+data["AssetId"]+')' 
ylabel     = 'RMS(m/s2)'
PETROspectro(accel,fs_m,titulo,ylabel,plot = True)
titulo     = 'Velocity ' +data["AssetName"]+'('+data["AssetId"]+')'    
ylabel     = 'RMS(mm/s)'
PETROspectro(speed,fs_m,titulo,ylabel,plot = True)
#
#
#"""
#pota = sorted(data, key=getKey)
#for counter,a in enumerate(pota,0):
#    print (counter,a['AssetId'],a['AssetName'])
#"""
#
#but_ord    = 5      
#f_low      = 2000
#f_up       = 0.49*fs_m
#b, a       = signal.butter(but_ord, ([2*f_low/fs_m , 2*f_up/fs_m]), 'bandpass', analog=False)
#gSE        = (signal.filtfilt(b, a, acceleration)) 
#envelope   = np.abs(hilbert(gSE))
#max_index  = np.argmax(envelope)
#max_value  = envelope[max_index]
#peaks, _   = find_peaks(envelope, height=0.90*max_value)
##gSE =([1,2,3,4,5,6,6,6,6,6])
#plt.figure()
#
#
#n, bins, patches = plt.hist(gSE, 1000,density=True, facecolor='g', alpha=0.75)
#plt.show()
#distance = bins[1]-bins[0]
#pdf      = n *distance
#probability = 0
#counter = 0
#while True:
#    probability = probability + pdf[counter]
#    counter = counter +1
#    if probability >= 0.998:
#        print ('counter',counter)
#        break
#print(bins[counter+1])
#
#
#
#plt.figure()
#plt.plot(tiempo,gSE,tiempo,envelope)
#plt.plot(tiempo[peaks],envelope[peaks],'o')
#plt.hlines(bins[counter+1],tiempo[0],tiempo[l-1])
#plt.show()
