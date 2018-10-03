# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
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
def pyt_espectro(value, fs):
    #args = fi, [],[] (arrays temporales)
      
    #l         = np.float(np.size(value))
    l         = np.size(value)
    output    = np.fft.fft(value)/l
    output    = np.abs(output)**2

    power     = np.std(value)
    power_HZ  = 10*np.log10(power**2 /fs/1000000)
    print ("mean (tiempo) =", np.mean(value),";  power = std(tiempo)**2 =",np.std(value)**2)
    print ("dB(Hz) AC=", power_HZ, 'dB/Hz')
    output           = 10*np.log10(output)
    f                = np.arange(l)/l*fs
    maximo = np.max(output)
    peaks,_ = find_peaks(output, height=maximo-10)
    
    f_avg, Pxx_den = signal.welch(value, fs,128)
    output_avg = 10*np.log10(Pxx_den)
    plt.figure()
    plt.plot(f,output)
    plt.plot(f_avg,output_avg)
    plt.plot(f[peaks], output[peaks],'o')
    plt.hlines(power_HZ, f[0] , f[l-1], colors='r', linestyles='dashed', linewidth=1, label=''  )
    
    plt.axis([0, fs/2, power_HZ-10, maximo+10])                  
    plt.grid()
    plt.show()

    return output, f
   
#------------------------------------------------------------------------------    



path = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-072611\\data'
file = 'sh3_json3.json'
file = 'Json2018.09.25.json'

with open(path+'//'+file, "r") as read_file:
    data = json.load(read_file)

if type(data) == dict:               #--------Solo de uno
    a= data
    #print (a['AssetId'],a['AssetName'])


else:                                #--------MAS uno
    for counter,a in enumerate(data,0):
        
        if a['AssetId'] == 'P-1404-B':
            print (counter,a['AssetId'],a['AssetName'])
            y = np.asarray(a['Value'])
            
            b           = a['Props']
            c           = b[0]
            fs          = np.float(c['Value'])
            pyt_espectro(y, fs)



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