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

from PETRONOR_lyb import load_json_file
from PETRONOR_lyb import PETROspectro



pi = np.pi

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\10\\10'
json_file = 'ab98949e-17e8-46eb-6d20-165279207208_amplitude.json'

data      = load_json_file(path+'\\'+json_file)
print (data["AssetId"])
print (data["AssetName"])    
print (data["AssetType"]) 
print (data["BusinessId"]) 
print (data["DeviceId"]) 
print (data["MeasurePointId"])    
print (data["MeasurePointName"])
print (data["MessageId"])
print (data["Name"])
print (data["PlantId"]) 
print (data["SensorId"])
print (data["SensorName"])
print (data["SensorType"])
print (data["ServerTimeStamp"])
print (data["SourceTimeStamp"])
print('')

accel      = np.asarray(data['Value'])
l          = np.size(accel) 

b          = data['Props']
c          = b[0]
fs_m       = np.float(c['Value'])

c          = b[4]
cal_factor = np.float(c['Value'])
accel      = accel * cal_factor * 9.81 

speed      = 1000*np.cumsum(accel -np.mean(accel))
speed_real = 1000*np.cumsum(accel)
#speed_real = speed
tiempo     = np.arange(l)  / fs_m

##STFT_waterfall(acceleration, fs_m)

print('ACCELERACION')
print('valor RMS :',np.std(accel))
print('Kurtosis  :',kurtosis(accel,fisher=False))
print('Media     :',np.mean(accel))
print('Skewness  :',sp.stats.skew(accel))
print('Maximun   :',np.max(accel))
print('Minimum   :',np.min(accel))
print('Varianza  :',np.std(accel)**2)
print('')
titulo     = 'Acceleration ' +data["AssetName"]+'('+data["AssetId"]+')' 
ylabel     = 'RMS(m/s2)'
PETROspectro(accel,fs_m,titulo,ylabel,plot = True)

print('VELOCITY')
print('valor RMS :',np.std(speed_real))
print('Kurtosis  :',kurtosis(speed_real,fisher=False))
print('Media     :',np.mean(speed_real))
print('Skewness  :',sp.stats.skew(speed_real))
print('Maximun   :',np.max(speed_real))
print('Minimum   :',np.min(speed_real))
print('Varianza  :',np.std(speed_real)**2)
print('')
titulo     = 'Velocity ' +data["AssetName"]+'('+data["AssetId"]+')'    
ylabel     = 'RMS(mm/s)'
PETROspectro(speed,fs_m,titulo,ylabel,plot = True)
