#import requests
#from PETRONOR_lyb import *
import requests
import os
import datetime, time
import numpy as np


from scipy import signal

import pandas as pd
from numba import jit
from PETRONOR_lyb import *

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
    tiempo       = datetime.datetime.fromtimestamp(seconds)        
    tiempo_string = str(tiempo)
    JsonTime = tiempo_string.replace(' ','T')+'Z'
    return JsonTime

def STD_mean(df):
    df.loc['mean'] = np.zeros(df.shape[1])
    df.loc['std']  = np.zeros(df.shape[1])
    for counter,k in enumerate(df.columns):
        if k[0:3] == 'RMS':
            columna          = df[k].values
            columna          = columna[columna != 0]
            df.loc['mean',k] = np.mean(columna)
            df.loc['std',k]  = np.std(columna)
            
    return df
#------------------------------------------------------------------------------  

def Save_File(parameters,df_freq_fingerprint,label):

    file_suffix =   '_'+parameters['Fecha Stop'].split('T')[0].replace('-','_')+'_to_'+parameters['Fecha Start'].split('T')[0].replace('-','_') 
    #file_suffix = 'kokokololo'
    df_freq_fingerprint.to_pickle (Path_out+'Freq_FP_'+parameters['IdAsset']+'_'+label+file_suffix + '.pkl')
    print('Spectral Finger Print Saved .pkl')
    writer      = pd.ExcelWriter  (Path_out+'Freq_FP_'+parameters['IdAsset']+'_'+label+file_suffix + '.xlsx')
    df_freq_fingerprint.to_excel(writer, 'DataFrame')
    writer.save()
    print('Spectral Finger Print Saved .xls')
    
    return

def Blower_Diagnostics(Finger_p):
    
    Finger_p = Blower_Wheel_Unbalance(Finger_p)
    Finger_p = Plain_Bearing_Clearance(Finger_p)
    Finger_p = Oil_Whirl(Finger_p)
    Finger_p = Oil_Whip(Finger_p)
    Finger_p = Blade_Faults(Finger_p)
    Finger_p = Flow_Turbulence(Finger_p)
    Finger_p = Plain_Bearing_Block_Looseness(Finger_p)
    Finger_p = Shaft_Misaligments(Finger_p)
    Finger_p = Pressure_Pulsations(Finger_p)
    Finger_p = Surge_Effect(Finger_p)
    Finger_p = Severe_Misaligment(Finger_p)
    Finger_p = Loose_Bedplate(Finger_p)
    return Finger_p
    

def Read_server(parameters,df_SH3bis,df_SH4bis):
    # ----------------------------construct API endpoint-----------------------
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

            if  std_SH3 > 0.001:
                array_SH3 = array_SH3-np.mean(array_SH3)
                array_SH3 = np.cumsum (G*1000*array_SH3/fs)    #---velocidad-     
                array_SH3 = signal.filtfilt(b, a, array_SH3) 
                array_SH3 = 1.63 * np.fft.fft(array_SH3* hann/l)
                df_SH3bis.loc[seg_SH3] = array_SH3

            else:
                print('maquina parada')
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

            if std_SH4 > 0.001:
                array_SH4 = array_SH4-np.mean(array_SH4)
                array_SH4 = np.cumsum (G*1000*array_SH4/fs)    #---velocidad-     
                array_SH4 = signal.filtfilt(b, a, array_SH4) 
                array_SH4 = 1.63 * np.fft.fft(array_SH4* hann/l)
                df_SH4bis.loc[seg_SH4] = array_SH4

            else:
                print('maquina parada')
        else:
            print('lectura corrupta')
            
    ulti_fecha      = Seconds_JsonTime(min(seg_SH3,seg_SH4))
    
    return df_SH3bis,df_SH4bis,ulti_fecha

def Read_serverbis(parameters):
    # ----------------------------construct API endpoint-----------------------
    api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
    '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
    '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])

    # -------------------------------------make GET request to API endpoint
    response          = requests.get(api_endpoint)
    # -------------------------------------convert response from server into json object
    response_json     = response.json()        
    
    l_data_SH3  = []
    l_fecha_SH3 = []
    l_data_SH4  = []
    l_fecha_SH4 = []
    
    seg_SH3     = -1
    seg_SH3     = -1
    print (response_json[0]['waveform'][0]['IdPosicion'])
    for lecture in response_json[0]['waveform'][0]['ValoresTrama']:
        if lecture != None:
            fecha_SH3    = lecture['ServerTimeStamp']
            
            seg_SH3      = JsonTime_Seconds (fecha_SH3)  
            cal_f        = np.float(lecture['Props'][4]['Value'])
            array_SH3    = np.asarray(lecture['Value']) *cal_f
            std_SH3      = np.std(array_SH3)

            if  std_SH3 > 0.001:
                array_SH3 = array_SH3-np.mean(array_SH3)
                array_SH3 = np.cumsum (G*1000*array_SH3/fs)    #---velocidad-     
                array_SH3 = signal.filtfilt(b, a, array_SH3) 
                array_SH3 = 1.63 * np.fft.fft(array_SH3* hann/l)
                l_fecha_SH3.append(seg_SH3)
                l_data_SH3.append(array_SH3)
            else:
                print('maquina parada')
        else:
            print('lectura corrupta')
    df_SPEED_SH3        = pd.DataFrame(data = l_data_SH3, index=l_fecha_SH3, columns=np.arange(l))
    Harm_SH3         = df_Harmonics(df_SPEED_SH3, fs,'blower')
    
    #plot_waterfall_lines(parameters,df_SH3bis,Harm_SH3,fs,0,400)
                
    print (response_json[0]['waveform'][1]['IdPosicion'])
    for lecture in response_json[0]['waveform'][1]['ValoresTrama']:
        if lecture != None:
            fecha_SH4    = lecture['ServerTimeStamp']
 
            seg_SH4      = JsonTime_Seconds (fecha_SH4)
            cal_f        = np.float(lecture['Props'][4]['Value'])
            array_SH4    = np.asarray(lecture['Value']) *cal_f
            std_SH4      = np.std(array_SH4)

            if std_SH4 > 0.001:
                array_SH4 = array_SH4-np.mean(array_SH4)
                array_SH4 = np.cumsum (G*1000*array_SH4/fs)    #---velocidad-     
                array_SH4 = signal.filtfilt(b, a, array_SH4) 
                array_SH4 = 1.63 * np.fft.fft(array_SH4* hann/l)
                l_fecha_SH4.append(seg_SH4)
                l_data_SH4.append(array_SH4)
            else:
                print('maquina parada')
        else:
            print('lectura corrupta')
    
    df_SPEED_SH4        = pd.DataFrame(data = l_data_SH4, index=l_fecha_SH4, columns=np.arange(l))
    Harm_SH4            = df_Harmonics(df_SPEED_SH4, fs,'blower')
    print ('los timepos----------',seg_SH3,seg_SH4)
    
    if seg_SH3 != -1 and seg_SH4 != -1: 
        ult_seg = min(seg_SH3,seg_SH4)
    if seg_SH3 == -1 and seg_SH4 != -1:
        ult_seg = seg_SH4
        print(seg_SH3)
    if seg_SH3 != -1 and seg_SH4 == -1:
        ult_seg = seg_SH3
        print(seg_SH4)
    if seg_SH3 == -1 and seg_SH4 == -1:
        ult_seg = JsonTime_Seconds(parameters['Fecha'])- 10*60*int(parameters['NumeroTramas'])
        print(seg_SH3,seg_SH3)

    ulti_fecha       = Seconds_JsonTime(ult_seg)
    
    return Harm_SH3,Harm_SH4,ulti_fecha

if __name__ == '__main__':
    
    start = time.time()
    l         = 16384
    fs        = 5120
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81


    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Localizacion' : 'SH4', #SH3/4
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-05-10T10:00:00.0Z',
        'FechaInicio'  : '',
        'NumeroTramas' : '100',
        'Fecha Start'  : '2019-05-10T00:00:00.0Z',
        'Fecha Stop'   : '2018-09-01T00:00:00.0Z',
        'Parametros'   : 'waveform'}
    
    parameters['Fecha']  = parameters['Fecha Start']
    df_FP_SH3 = pd.DataFrame()
    df_FP_SH4 = pd.DataFrame()
    while JsonTime_Seconds(parameters['Fecha Stop']) < JsonTime_Seconds(parameters['Fecha']) :
        print('--------Accediendo servidor Petronor-------------',parameters['Fecha'])
        #df_SH3,df_SH4,last_data = Read_server(parameters,df_SH3,df_SH4
        df_FP_SH3p,df_FP_SH4p,last_data = Read_serverbis(parameters)
        df_FP_SH3 = pd.concat([df_FP_SH3,df_FP_SH3p])
        df_FP_SH4 = pd.concat([df_FP_SH4,df_FP_SH4p])
        parameters['Fecha'] = last_data
    
    print('---------Fin conexion Server)------------------------------')
    print('-----Blower_Diagnostics(df_FP_SH3)-------------------------')
    df_FP_SH3 = Blower_Diagnostics(df_FP_SH3)
    print('-----Blower_Diagnostics(df_FP_SH4)-------------------------')
    df_FP_SH4 = Blower_Diagnostics(df_FP_SH4)
    df_FP_SH3 = STD_mean(df_FP_SH3)    
    df_FP_SH4 = STD_mean(df_FP_SH4)
    

    Save_File(parameters,df_FP_SH3,'SH3')
    Save_File(parameters,df_FP_SH4,'SH4')
    end  = time.time()
    print('tiempo---------------------',end - start)


