# -*- coding: utf-8 -*-
"""
Editor de Spyder


"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

from PETRONOR_lyb_old import load_json_file
from PETRONOR_lyb_old import PETROspectro
from PETRONOR_lyb_old import spectro


pi = np.pi
g= 9.81

r2 = np.sqrt(2)
path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\10\\10'
json_file = 'ab98949e-17e8-46eb-6d20-165279207208_amplitude.json'

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\19\\11'
json_file = '0ecbe08c-295c-4e77-5401-9e0c8a627990_amplitude.json'

#path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\21\\11'
#json_file = '0ecbe08c-295c-4e77-5401-9e0c8a627990_amplitude.json'

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data'
json_file = 'calibracion.json'

datas     = load_json_file(path+'\\'+json_file)

data      = datas['Calibration'][6]
#data= datas
 
print(data["AssetId"]         )
print(data["AssetName"]       )    
print(data["AssetType"]       ) 
print(data["BusinessId"]      ) 
print(data["DeviceId"]        ) 
print(data["MeasurePointId"]  )    
print(data["MeasurePointName"])
print(data["MessageId"]       )
print(data["Name"]            ) 
print(data["PlantId"]         ) 
print(data["SensorId"]        )
print(data["SensorName"]      )
print(data["SensorType"]      )
print(data["ServerTimeStamp"] )
print(data["SourceTimeStamp"] )
print('')

accell      = np.asarray(data['Value'])
l          = np.size(accell) 

fs_m       = np.float(data['Props'][0]['Value'])
cal_factor = np.float(data['Props'][4]['Value'])
tiempo     = np.arange(l)  / fs_m
f          = np.arange(l)/l*fs_m
accell      = accell * cal_factor 
DC         = np.mean(accell)

#print(np.sqrt(np.sum(accel**2)/l))
#print(np.sqrt((np.std(accel)**2+DC**2)))
print('acel RMS  :',(np.std(accell) ))

accel      = accell-DC
ACCEL      = np.abs(np.fft.fft(accel)/l)

print('RMS accel :',format(np.sqrt(np.sum(accel**2)/l),'0.2f'))
print('RMS ACCEL :',format(np.sqrt(np.sum(ACCEL**2)),'0.2f'))
print()
indexes_a, properties_a = find_peaks(ACCEL[0:int(l/2)],height  = 0.1 ,prominence = 0.01 , width=1 , rel_height = 0.85)

pico = np.sqrt(2*np.sum(ACCEL[int(np.round(properties_a["left_ips"][0])): int(np.round(properties_a["right_ips"][0])) + 1]**2))

print('RMS accel:',format( np.sqrt(np.sum(accel**2)/l), '0.2f'))
print('RMS ACCEL:',format( np.sqrt(np.sum(ACCEL**2)),   '0.2f'))
print('RMS x1.0  :',format( pico,                       '0.2f'),'Max Spectrum :',np.sqrt(2*np.max(ACCEL)**2))

print()



plt.figure()
plt.plot(f,2*ACCEL)
#plt.axis([159-4,159+4, 0 ,2])
plt.grid(True)
plt.show()

for k in range(20):
    lon = l-1*k
    f = np.arange(lon)/lon*fs_m
    array = accell[0:lon]
    array = array-np.mean(array)
    ARRAY = np.abs(np.fft.fft(array)/lon)
    indice = np.argmax(ARRAY)
    print (lon,f[indice],'Hz    Veff===>', np.sum(ARRAY**2),r2*ARRAY[indice])

#titulo     = 'Acceleration ' +data["AssetName"]+'('+data["AssetId"]+')' 
#ylabel     = 'g'
#PETROspectro(accel,fs_m,titulo,ylabel,Detection = 'Peak')

"""

speed      = g * 1000*np.cumsum(accel)/fs_m
b, a       = signal.butter(3,2*10/fs_m,'highpass',analog=False)
speed      = signal.filtfilt(b, a, speed)
SPEED      = np.abs(np.fft.fft(speed)/l)


print('RMS speed :',format(np.sqrt(np.sum(speed**2)/l),'0.2f'))
print('RMS SPEED :',format(np.sqrt(np.sum(SPEED**2)),'0.2f'))
print()
indexes_s, properties_s = find_peaks(SPEED[0:int(l/2)],height  = 2 ,prominence = 0.01 , width=1 , rel_height = 0.85)

pico = np.sqrt(2*np.sum(SPEED[int(np.round(properties_s["left_ips"][0])): int(np.round(properties_s["right_ips"][0])) + 1]**2))

print('Pico speed:',format(r2 * np.sqrt(np.sum(speed**2)/l), '0.2f'))
print('Pico SPEED:',format(r2 * np.sqrt(np.sum(SPEED**2)),   '0.2f'))
print('Pico x1.0 :',format(r2 * pico,                        '0.2f'),'Max Spectrum :',r2*np.sqrt(2*np.max(SPEED)**2))

#titulo     = 'Velocity ' +data["AssetName"]+'('+data["AssetId"]+')'    
#ylabel     = 'mm/s'
#PETROspectro(speed,fs_m,titulo,ylabel,Detection = 'Peak')

plt.figure()
plt.plot(f,2*SPEED)
plt.axis([159-4,159+4, 0 ,20])
plt.grid(True)
plt.show()
"""