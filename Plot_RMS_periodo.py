# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 09:14:53 2019

@author: 106300
"""

import requests

import datetime, time
from datetime import timedelta, date
import numpy as np

from scipy.signal import hilbert, chirp
#from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

from scipy.signal import find_peaks

import pandas as pd
import os
from pandas import DataFrame

pi        = np.pi
E1        = 0.15
fs        = 5120.0
l         = 16384
path      = 'C:/OPG106300/TRABAJO/Proyectos/Petronor-075879.1 T 20000/Trabajo/python'


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)
#------------------------------------------------------------------------------
def load_vibrationData(input_data, Parameters):#MeasurePointId, num_tramas, assetId):

    data           = []
    date           = []
    lista_maquinas = []
    MeasurePointId = Parameters['Localizacion']
    num_tramas     = Parameters['NumeroTramas']
    assetId        = Parameters['IdAsset']
    format = "%Y-%m-%dT%H:%M:%S"

    # get waveform data
    waveform = input_data[0]['waveform']
    # get waveform data specific to a sensor (global variable)
    waveform = [sensor for sensor in waveform if sensor['IdPosicion'] == MeasurePointId]

    # loop thorugh each of the tramas for a specific sensor
    for num_trama in range(int(num_tramas)):
        values_trama = waveform[0]['ValoresTrama'][num_trama]

        if values_trama != None:

            res = pd.DataFrame.from_dict(values_trama, orient='index')
            res = res.transpose()

            word = str(res.AssetId.values[0])+' '+str(res.MeasurePointId.values[0])
            if (word in lista_maquinas) == False:
                lista_maquinas.append( word )

            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                #print(root)
                #print(res.Props.iloc[0][0]['Value'])
                #print(res.Props.iloc[0][4]['Value'])
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                data = np.asarray(res.Value.values[0])*cal_factor

                #data.append(res.Value.values[0])
                #print(res.MeasurePointId.values[0],res.MeasurePointName.values[0] )
                fecha = res.ServerTimeStamp.values[0]
                #print(fecha,np.float(fecha[19:len(fecha)-1]))
                datetime_obj = datetime.datetime.strptime(fecha[0:19],format) # pierde la parte decimal de los segundos
                                                                              # sumo la patrte decimal de los segundos
                segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1])
                #------------para pasar el YY MM DD hh mm ss EXACTO!!!!------------
                #----------datetime.datetime.fromtimestamp(segundos)-----------

                print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )

                fecha = segundos
                

    df_out     = data
    
    return df_out, fecha

#------------------------------------------------------------
    


#--------------------------------------------------------------------------------
if __name__ == '__main__':
    
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81

    parameters = {
        'IdPlanta': 'BPT',
        'IdAsset': 'H4-FA-0001',
        'Localizacion' : 'SH4', #SH4/MH2
        'Fecha':       '2019-01-05T00:00:00',
        'FechaInicio': '2018-10-12T00:52:46',
        'NumeroTramas': '2',
        'Parametros': 'waveform'
    }
#    if parameters['Localizacion']== 'SH4':
#        flag_pos = 1
#    if parameters['Localizacion']== 'MH2':
#        flag_pos = 0
    #localizacion     = 'SH4' 
    
    
    start     = datetime.datetime(2019, 2, 11, 0, 0)
    start_s   = time.mktime(start.timetuple())
    
    end       = date.today()
    end       = datetime.datetime(2019, 2, 18, 0, 0)
    end_s     = time.mktime(end.timetuple())
    RMS_array = np.array([])
    instante_old = 0
    #for segundos in np.arange(start_s, end_s,60*5):
    for segundos in range(1):
       
        #fecha = str(datetime.datetime.fromtimestamp(segundos)).split(' ')[0]
        #hora  = str(datetime.datetime.fromtimestamp(segundos)).split(' ')[1]
        
        #parameters['Fecha'] = fecha+'T'+hora
        
    #for i in range(3):    
        print (parameters['Fecha'])
        api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
        '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
        '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])
    
        # make GET request to API endpoint
        response = requests.get(api_endpoint)
        # convert response from server into json object
        response_json = response.json()
    
        # check if server is up or not. They usually restart the server every day...
        # check if there are Nones in 'ValoresTrama
        if response_json[0]['waveform'][0]['IdPosicion'] == parameters['Localizacion']:
            flag_pos = 0
        if response_json[0]['waveform'][1]['IdPosicion'] == parameters['Localizacion']:
            flag_pos = 1
        flag = response_json[0]['waveform'][flag_pos]['ValoresTrama']
        # if there are Nones
        if None in flag:
            print("Server is down...")
        # if there are not, run script functions
        else:
    
            assetId = parameters['IdAsset']
            numTramas = parameters['NumeroTramas']
            trace,instante = load_vibrationData(response_json, parameters)
            if instante_old != instante:
                trace                   = trace - np.mean(trace)
                trace                   = np.cumsum (G*1000*trace/fs)    #---velocidad-     
                trace_speed             = signal.filtfilt(b, a, trace) 
                RMS = np.std(trace_speed)
                print(RMS)
                RMS_array= np.append(RMS_array,RMS)
                #trace_SPEED             = np.abs(np.fft.fft(trace_speed* hann/l))
            else:
                print('trama repetida')
            instante_old = instante


print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FINNNNNNNNNNNNNNNN')


