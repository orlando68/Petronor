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
from PETRONOR_lyb import spectro


pi = np.pi
g= 9.81


path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\10\\10'
json_file = 'ab98949e-17e8-46eb-6d20-165279207208_amplitude.json'

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\19\\11'
json_file = '0ecbe08c-295c-4e77-5401-9e0c8a627990_amplitude.json'

#path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\21\\11'
#json_file = '0ecbe08c-295c-4e77-5401-9e0c8a627990_amplitude.json'

#path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data'
#json_file = 'calibracion.json'

datas     = load_json_file(path+'\\'+json_file)

#data      = datas['Calibration'][6]
data= datas
 
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

accel      = np.asarray(data['Value'])
l          = np.size(accel) 

fs_m       = np.float(data['Props'][0]['Value'])
cal_factor = np.float(data['Props'][4]['Value'])
accel      = accel * cal_factor 

#t           = np.arange(l)
#accel      = 1*np.sin(2*np.pi*t*159/fs_m)
titulo     = 'Acceleration ' +data["AssetName"]+'('+data["AssetId"]+')' 
ylabel     = 'g'
#PETROspectro(accel,fs_m,titulo,ylabel,Detection = 'Peak')



speed      = g * 1000*np.cumsum(accel-np.mean(accel))/fs_m
tiempo     = np.arange(l)  / fs_m

titulo     = 'Velocity ' +data["AssetName"]+'('+data["AssetId"]+')'    
ylabel     = 'mm/s'
PETROspectro(speed,fs_m,titulo,ylabel,Detection = 'Peak')

"""
plt.figure()
plt.plot(tiempo,accel)
plt.show()
plt.figure()
plt.plot(tiempo,speed*g*1000)
plt.show()
"""