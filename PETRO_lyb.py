# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:56:32 2018
Librerias para el proyecto PETRONOR
@author: 106300
"""

import requests
import os
import datetime, time
import numpy as np


from scipy import signal

import pandas as pd
from numba import jit



pi        = np.pi
E1        = 0.15
fs        = 5120.0
l         = 16384
Path_out  = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------  

def save_files(parameters,df_speed,df_SPEED,df_freq_fingerprint):
    f_fin   = 300
    i_f_fin = int(l*f_fin/fs)
    
    if parameters['Source'] == 'Petronor Server': # ======>     Acceso servidor
        
        fecha_label        = parameters['Fecha'].split('-')[2].split('T')[0]+'_'+ parameters['Fecha'].split('-')[2].split('T')[0]+'_'+parameters['Fecha'].split('-')[0]
        print(fecha_label)
                                    #---------tengo que salvar los ficheros de velocity
        file_suffix        = parameters['IdAsset']+'_' + parameters['Localizacion'] + '_' + fecha_label 
#        fichero_speed      = 'speed_t_'+file_suffix + '.pkl'
#        fichero_SPEED      = 'SPEED_f_'+file_suffix + '.pkl'
#        df_speed.to_pickle (Path_out+fichero_speed)
#        df_SPEED.to_pickle (Path_out+fichero_SPEED)
    
    if parameters['Source'] == 'Local Database':  # ======>     Acceso Disco 
        
        month          = parameters['Month']
        day            = parameters['Day']
        hour           = parameters['Hour']
        
        file_suffix    = parameters['IdAsset']+'_' + parameters['Localizacion'] + '_' + month+'_'+day+'_'+hour 
        fichero_speed  = 'speed_t_'+file_suffix + '.pkl'
        fichero_SPEED  = 'SPEED_f_'+file_suffix + '.pkl'
        
        if find_file(fichero_speed,fichero_SPEED) == False:
            df_speed.to_pickle (Path_out+fichero_speed )
            df_SPEED.to_pickle (Path_out+fichero_SPEED )
            print('Velocity files (t&f) saved')
            
    #-------------aqui almaceno un pedazo del espectro para DAMMIKA--------
    writer             = pd.ExcelWriter(Path_out+'SPEED_f_'+file_suffix+'.xlsx')
    df_SPEED.take(np.arange(i_f_fin),axis= 1).to_excel(writer, 'DataFrame')
    writer.save()

    df_freq_fingerprint.to_pickle (Path_out+'Freq_FP_'+file_suffix + '.pkl')
    print('Spectral Finger Print Saved .pkl')
    writer      = pd.ExcelWriter(Path_out+'Freq_FP_'+file_suffix + '.xlsx')
    df_freq_fingerprint.to_excel(writer, 'DataFrame')
    writer.save()
    print('Spectral Finger Print Saved .xls')
    
    return
#------------------------------------------------------------------------------            
      
@jit
def find_file(file1,file2):
    
    found1 = False
    found2 = False
    files  = os.listdir(Path_out)
    
    for name in files:
        
        if file1 == name:
            found1  = True
        if file2 == name:
            found2  = True   
    found  = found1 and found2
    print ('Encontrado?--------------------------',found)
    return found

# --------- BLOWERS
# Load_Vibration_Data_Global
#     │___Load_Vibration_Data_From_Get
#     │___Load_Vibration_Data_From_DB
#
#--------- PUMPS (no leo de mi disco duro, pq no disponibles 2018)
# Load_Vibration_Data_Global_pumps
#     │___Load_Vibration_Data_From_Get_pumps
#

#------------------------------------------------------------------------------
#                        BLOWERS    
#------------------------------------------------------------------------------
def Load_Vibration(parameters):
    

    print('--------Accediendo servidor Petronor-------------',parameters['Fecha'])
    # -----------------------------------------------construct API endpoint
    api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
    '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
    '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])

    # -------------------------------------make GET request to API endpoint
    response          = requests.get(api_endpoint)
    # -------------------------------------convert response from server into json object
    response_json     = response.json()        
    df_accel, fecha   = Load_Vibration_Get(response_json, parameters)   
    df_speed,df_SPEED = velocity(df_accel)

   
 
    return df_speed,df_SPEED,fecha
#------------------------------------------------------------------------------    


def Load_Vibration_Get(input_data, Parameters):#MeasurePointId, num_tramas, assetId):

    data           = []
    date           = []
    MeasurePointId = Parameters['Localizacion']
    assetId        = Parameters['IdAsset']
    format = "%Y-%m-%dT%H:%M:%S"

    # get waveform data
    for counter,sensor in enumerate(input_data[0]['waveform']):
        if sensor['IdPosicion'] == Parameters['Localizacion']:
            flag_pos = counter    
            
    print('--------------------', input_data[0]['waveform'][flag_pos]['IdPosicion'])
    frames = []
    
    for trama in input_data[0]['waveform'][flag_pos]['ValoresTrama']:
        if trama != None:
            #print ('holeeeeeeeeeeeee')
            frames.append(trama)
    for values_trama in frames:

        res = pd.DataFrame.from_dict(values_trama, orient='index')
        res = res.transpose()
        
        if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:

            cal_factor = np.float(res.Props.iloc[0][4]['Value'])
            data.append(np.asarray(res.Value.values[0])*cal_factor)

            fecha = res.ServerTimeStamp.values[0]
            datetime_obj = datetime.datetime.strptime(fecha[0:19],format) # pierde la parte decimal de los segundos
                                                                          # sumo la patrte decimal de los segundos
            segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1])
            fecha = segundos
            date.append(fecha)
            # para recuperar (EXACTO) "YY MM DD hh mm ss" ==> datetime.datetime.fromtimestamp(segundos)

            print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )

    pp = str(datetime.datetime.fromtimestamp(segundos-60*5))
    pp = pp.replace(' ','T')+'Z'
    #print(pp+'Z','   esta')
            

    df_out     = pd.DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    fecha = pp
    return df_out, fecha 
#------------------------------------------------------------------------------
#                                   PUMPS    
#------------------------------------------------------------------------------

def Load_Vibration_Data_Global_Pumps(parameters):

    if parameters['Source'] == 'Petronor Server': # ======>     Acceso servidor
        print('------------------------------------Accediendo servidor Petronor')
        # -----------------------------------------------construct API endpoint
        api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
        '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
        '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])
    
        # -------------------------------------make GET request to API endpoint
        response = requests.get(api_endpoint)
        # -------------------------------------convert response from server into json object
        response_json = response.json()
        
        df_accel_BH3,df_accel_BA4,df_accel_BV4 = Load_Vibration_Data_From_Get_Pumps(response_json, parameters)   

        df_speed_BH3,df_SPEED_BH3  = velocity(df_accel_BH3)
        df_speed_BV4,df_SPEED_BV4  = velocity(df_accel_BV4)
        df_speed_BA4,df_SPEED_BA4  = velocity(df_accel_BA4)

    return df_speed_BH3,df_SPEED_BH3,df_speed_BV4,df_SPEED_BV4,df_speed_BA4,df_SPEED_BA4

#------------------------------------------------------------------------------

def Load_Vibration_Data_From_Get_Pumps(input_data, Parameters):#MeasurePointId, num_tramas, assetId):
    #BH3 (horizontal), BA4 (axial) y BV4 (radial)
    
    date           = []
    MeasurePointId = Parameters['Localizacion']    
    assetId        = Parameters['IdAsset']
    format = "%Y-%m-%dT%H:%M:%S"

    frames_BH3   = []
    frames_BV4   = []
    frames_BA4   = []
    lista_frames = (frames_BH3,frames_BV4,frames_BA4)
    data_BH3     = []
    data_BV4     = []
    data_BA4     = []
    lista_data   = (data_BH3,data_BV4,data_BA4)
    date_BH3     = []
    date_BV4     = []
    date_BA4     = []
    lista_date   = (date_BH3,date_BV4,date_BA4)
    for flag_pos in range(3):
        print('--------------------', input_data[0]['waveform'][flag_pos]['IdPosicion'])
        for trama in input_data[0]['waveform'][flag_pos]['ValoresTrama']:
            if trama != None:
                lista_frames[flag_pos].append(trama)

        for values_trama in lista_frames[flag_pos]:
    
            res = pd.DataFrame.from_dict(values_trama, orient='index')
            res = res.transpose()
    
            if res.AssetId.values[0] == assetId:# and res.MeasurePointId.values[0] == MeasurePointId:
                #print(root)
                print(res.Props.iloc[0][0]['Value'])
                print(res.Props.iloc[0][4]['Value'])
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                #data.append(np.asarray(res.Value.values[0])*cal_factor)
                lista_data[flag_pos].append(np.asarray(res.Value.values[0])*cal_factor)
                fecha = res.ServerTimeStamp.values[0]
                datetime_obj = datetime.datetime.strptime(fecha[0:19],format) # pierde la parte decimal de los segundos
                                                                              # sumo la patrte decimal de los segundos
                segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1])
                # para recuperar (EXACTO) "YY MM DD hh mm ss" ==> datetime.datetime.fromtimestamp(segundos)
    
                print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )
    
                fecha = segundos
                #date.append(fecha)
                lista_date[flag_pos].append(fecha)

    df_out_BH3     = pd.DataFrame(data=lista_data[0], index=lista_date[0])
    df_out_BH3.sort_index(inplace=True)
    df_out_BV4     = pd.DataFrame(data=lista_data[1], index=lista_date[1])
    df_out_BV4.sort_index(inplace=True)
    df_out_BA4     = pd.DataFrame(data=lista_data[2], index=lista_date[2])
    df_out_BA4.sort_index(inplace=True)
    
    return df_out_BH3,df_out_BV4,df_out_BA4

#------------------------------------------------------------------------------
@jit
def velocity(df_in):
    #---- devuleve el espectro con ventana de hanning y corregida en potencia
    #---- es decir multiplicado por el factor 1.63    
    l         = df_in.shape[1]
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81
    
    df_speed  = pd.DataFrame(np.nan                                            ,index = df_in.index,columns = df_in.columns.values)
    df_cmplx  = pd.DataFrame(np.zeros([np.size(df_in.index), l], dtype=complex),index = df_in.index,columns = df_in.columns.values)
    
    for counter,indice in enumerate(df_in.index):
        trace                   = df_in.iloc[counter].values- np.mean(df_in.iloc[counter].values)
        trace                   = np.cumsum (G*1000*trace/fs)    #---velocidad-     
        trace_speed             = signal.filtfilt(b, a, trace) 
        df_speed.iloc[counter]  = trace_speed
                                    #------------------------------------------
        trace_SPEED             = 1.63 * np.fft.fft(trace_speed* hann/l)
        df_cmplx.iloc[counter]  =  trace_SPEED 
    
    return df_speed,df_cmplx

#------------------------------------------------------------------------------
