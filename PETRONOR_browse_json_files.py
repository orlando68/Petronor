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

import os
from PETRONOR_lyb import load_json_file


pi = np.pi

path    = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\'
path    = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018\\10\\10\\10'

for dirname, dirnames, filenames in os.walk(path):
    # print path to all subdirectories first.
#    for subdirname in dirnames:
#        directorio = os.path.join(dirname, subdirname)
#        print(directorio)
    #print(filenames)    
    for counter,filename in enumerate(filenames):
        address = os.path.join(dirname, filename)

        data = load_json_file(address)
        #print (data["AssetId"],address)
        if data["AssetId"] == 'H4-FA-0002':
            print (address) 
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
            print('----------------------------------------------------------')





