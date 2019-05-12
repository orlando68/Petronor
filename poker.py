#import requests
#from PETRONOR_lyb import *
import requests
import os
import datetime, time
import numpy as np


from scipy import signal

import pandas as pd
from numba import jit
#--------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------

def JsonTime_Seconds(JsonTime):
    format = "%Y-%m-%dT%H:%M:%S"
    #print('>>>>>>>>>>>>>>>   ',JsonTime.split('Z')[0])
    datetime_obj = datetime.datetime.strptime(JsonTime.split('.')[0],format)
    decimals   = JsonTime.split('.')[1].split('Z')[0]
    decimals_s = np.float(decimals)/10**(len(decimals) )
    seg          = time.mktime( datetime_obj.timetuple()) + decimals_s
    #print(datetime_obj,seg)
    return seg

def Seconds_JsonTime(seconds):
    tiempo = datetime.datetime.fromtimestamp(seconds)        
    tiempo_string = str(tiempo)
    JsonTime = tiempo_string.replace(' ','T')+'Z'
    return JsonTime


if __name__ == '__main__':
    l         = 16384
    fs        = 5120
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81
    fecha_stop = '2019-03-27T00:20:00.9988564Z'
    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Localizacion' : 'SH4', #SH3/4
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-03-27T23:20:00.9988564Z',
        'FechaInicio'  : '',
        'NumeroTramas' : '5',
        'Parametros'   : 'waveform'}

    data_SH3           = []
    date_SH3           = []
    data_SH4           = []
    date_SH4           = []
    
    iter = 0

    while JsonTime_Seconds(fecha_stop) < JsonTime_Seconds(parameters['Fecha']) :
        
        
    
        print(iter,'--------Accediendo servidor Petronor-------------',parameters['Fecha'])
        # -----------------------------------------------construct API endpoint
        api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
        '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
        '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])
    
        # -------------------------------------make GET request to API endpoint
        response          = requests.get(api_endpoint)
        # -------------------------------------convert response from server into json object
        response_json     = response.json()        
        
        
        print (response_json[0]['waveform'][0]['IdPosicion'])
        for lecture in response_json[0]['waveform'][0]['ValoresTrama']:
            if lecture != None:
                fecha_SH3    = lecture['ServerTimeStamp']
    
                seg_SH3      = JsonTime_Seconds (fecha_SH3)  
                cal_f        = np.float(lecture['Props'][4]['Value'])
                array_SH3    = np.asarray(lecture['Value']) *cal_f
                std_SH3      = np.std(array_SH3)
                #print(fecha_SH3,std_SH3)
                if  std_SH3 > 0.001:
                    #print('hola')
                    array_SH3 = array_SH3-np.mean(array_SH3)
                    array_SH3 = np.cumsum (G*1000*array_SH3/fs)    #---velocidad-     
                    array_SH3 = signal.filtfilt(b, a, array_SH3) 
                    array_SH3 = 1.63 * np.fft.fft(array_SH3* hann/l)
                    data_SH3.append(array_SH3)
                    date_SH3.append(seg_SH3)        
            else:
                print('lectura corrupta')
                    
        print (response_json[0]['waveform'][1]['IdPosicion'])
        for lecture in response_json[0]['waveform'][1]['ValoresTrama']:
            if lecture != None:
                fecha_SH4    = lecture['ServerTimeStamp']
     
                seg_SH4      = JsonTime_Seconds (fecha_SH4)
                cal_f        = np.float(lecture['Props'][4]['Value'])
                array_SH4    = np.asarray(lecture['Value']) *cal_f
                std_SH4      = np.std(array_SH4)
                #print(fecha_SH4,std_SH4)
                if std_SH4 > 0.001:
                    array_SH4 = array_SH4-np.mean(array_SH4)
                    array_SH4 = np.cumsum (G*1000*array_SH4/fs)    #---velocidad-     
                    array_SH4 = signal.filtfilt(b, a, array_SH4) 
                    array_SH4 = 1.63 * np.fft.fft(array_SH4* hann/l)
                    data_SH4.append(array_SH4)
                    date_SH4.append(seg_SH4)
            else:
                print('lectura corrupta')
                
        last_seg      = min(seg_SH3,seg_SH4)
        
        last_data_SH3 = Seconds_JsonTime(seg_SH3)
        last_data_SH4 = Seconds_JsonTime(seg_SH4)
        last_data     = Seconds_JsonTime(last_seg)
        
        #☺print(seg_SH3,seg_SH4)
        #print(last_data_SH3,last_data_SH4,'---',last_data)
        
        
        parameters['Fecha'] = last_data

        #iter = iter +1
        print('Fin = ',fecha_stop, 'ahora = ', last_data)
        print (JsonTime_Seconds(fecha_stop),JsonTime_Seconds(parameters['Fecha']),'-------------', JsonTime_Seconds(parameters['Fecha'])-JsonTime_Seconds(fecha_stop))
        print('sigo?????  ',JsonTime_Seconds(fecha_stop) < JsonTime_Seconds(parameters['Fecha']))
#        if iter == 3:
#            break
    df_SH3     = pd.DataFrame(data=data_SH3, index=date_SH3)
    df_SH3.sort_index(inplace=True)
        
    df_SH4     = pd.DataFrame(data=data_SH4, index=date_SH4)
    df_SH4.sort_index(inplace=True)
    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
