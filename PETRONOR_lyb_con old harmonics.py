# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:56:32 2018
Librerias para el proyecto PETRONOR
@author: 106300
"""

import requests

import datetime, time
import numpy as np

from scipy.signal import hilbert, chirp

from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates
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
Path_out  = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#------------------------------------------------------------------------------

class fingerprint:
    def __init__(self,        label,tipo,f1,f2,f_norm):
        self.label = label
        self.tipo  = tipo
        self.f1    = f1
        self.f2    = f2
        self.f_norm = f_norm
       

    def __str__(self):
        s = ''.join(['Label    : ', str(self.label),   '\n',
                     'Tipo     : ', str(self.tipo),    '\n',
                     'f1       : ', str(self.f1),      '\n',
                     'f2       : ', str(self.f2),      '\n',
                     'Relative : ', str(self.f_norm),'\n',
                     '\n'])
        return s

#------------------------------------------------------------------------------
def PK(a):
    if a > E1:
        out = True
    else:
        out = False
    return out

def PEAKS(*args):
    E1 = 0.15
    n_peaks = np.size(args)
    counter = 0
    for peak in args:
        if peak > E1:
            counter = counter+1
    if counter == n_peaks:
        out = True
    else:
        out = False
    return out
    
def NO_PEAKS(*args):
    E1 = 0.15
    out = True
    for peak in args:
        if peak > E1:
            out = False
    return out

def Truth_Table (A,B,C):
    out = 'None'
    if A:
        out = 'Green'
    if B:
        out = 'Yellow'
    if C:
        out = 'Red'
    return out
#------------------------------------------------------------------------------        
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

#------------------------------------------------------------------------------
def Load_Vibration_Data_Global(parameters):
    f_fin   = 300
    i_f_fin = int(l*f_fin/fs)
    f       = np.arange(l)/(l-1)*fs
    f       = f[0:i_f_fin]
    df_freq = pd.DataFrame([f],columns=np.arange(i_f_fin))
    
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
        
        df_accel           = Load_Vibration_Data_From_Get(response_json, parameters)   
        df_speed,df_SPEED  = velocity(df_accel)

        fecha_label        = parameters['Fecha'].split('-')[2].split('T')[0]+'_'+ parameters['Fecha'].split('-')[2].split('T')[0]+'_'+parameters['Fecha'].split('-')[0]
    
        writer             = pd.ExcelWriter(Path_out+'SPEED_f_Server_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+fecha_label+'.xlsx')
        df_SPEED.take(np.arange(i_f_fin),axis= 1).to_excel(writer, 'DataFrame')
        writer.save()
        
    if parameters['Source'] == 'Local Database':  # ======>     Acceso Disco  
        print('----------------------------------Accediendo Basedatos en  local')  
        month          = parameters['Month']
        day            = parameters['Day']
        hour           = parameters['Hour']
        
        fichero_speed  = 'speed_t_Local_DB_'+parameters['IdAsset']+'_' + parameters['Localizacion'] + '_' + month+'_'+day+'_'+hour + '.pkl'
        fichero_SPEED  = 'SPEED_f_Local_DB_'+parameters['IdAsset']+'_' + parameters['Localizacion'] + '_' + month+'_'+day+'_'+hour + '.pkl'
        
        Found          = find_file(fichero_speed,fichero_SPEED)
        #Found          = False
        #print (nombre_fichero,Found)
        if Found :
            print('-----------------------------------Accediendo fichero pickle  desde ficheros')
                                                           
            df_speed           = pd.read_pickle(Path_out + fichero_speed)
            df_SPEED           = pd.read_pickle(Path_out + fichero_SPEED)
                                                           
            
        else:
            print('------------------------------------Accediendo ficheros json desde BaseDatos')
            df_accel           = Load_Vibration_Data_From_DB(parameters['Path']+'\\'+month+'\\'+day+'\\'+hour,parameters['IdAsset'],parameters['Localizacion'])
            df_speed,df_SPEED  = velocity(df_accel)

            df_speed.to_pickle (Path_out+fichero_speed)
            df_SPEED.to_pickle (Path_out+fichero_SPEED)
                                         
            #-----we take just a dataframe portion 
#            df_save= df_SPEED.take(np.arange(i_f_fin),axis= 1)
#            df_save = df_save.append(df_freq)
#            
#            writer = pd.ExcelWriter(Path_out+'SPEED_f_Local_DB_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+month+'_'+day+'_'+hour+'.xlsx')
#            df_save.to_excel(writer, 'DataFrame')
#            writer.save()
            #-----we save the whole dataframe
#            writer = pd.ExcelWriter(Path_out+'SPEED_f_Local_DB_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+month+'_'+day+'_'+hour+'.xlsx')
#            df_SPEED.to_excel(writer, 'DataFrame')
#            writer.save()
 
    return df_speed,df_SPEED


#------------------------------------------------------------------------------

    
def Load_Vibration_Data_From_DB(rootdir, assetId,MeasurePointId):
    data           = []
    date           = []
    lista_maquinas = []
    format = "%Y-%m-%dT%H:%M:%S"
    counter = 0
    for root, dirs, files in os.walk(rootdir):

        for filename in files:
            fullpath = os.path.join(root, filename) 
            #print(fullpath)
            # read the entire file into a python array
            with open(fullpath, 'rb') as f:
                file = f.read().decode("utf-8-sig").encode("utf-8")
            res = pd.read_json(file, lines=True)
            nfiles = 1
                                                         #--ver todo lo que hay
            #print (res.AssetId.values[0],res.MeasurePointId.values[0]) 
                                                         #---------------------            
            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                #print(root)
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                data.append(np.asarray(res.Value.values[0])*cal_factor)
                
                #data.append(res.Value.values[0])
                #print(res.MeasurePointId.values[0],res.MeasurePointName.values[0] )
                fecha = res.ServerTimeStamp.values[0]
                #print(fecha,np.float(fecha[19:len(fecha)-1]))
                datetime_obj = datetime.datetime.strptime(fecha[0:19],format) # pierde la parte decimal de los segundos
                                                                              # sumo la patrte decimal de los segundos
                segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1]) 
                # para recuperar (EXACTO) "YY MM DD hh mm ss" ==> datetime.datetime.fromtimestamp(segundos)
                
                        
                print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )
                fecha = segundos
                date.append(fecha)
#                if nfiles == 6: #----files per day per day
#                    break 
#                nfiles = nfiles +1

        counter = counter +1
    df_out     = DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out
#------------------------------------------------------------------------------

def Load_Vibration_Data_From_Get(input_data, Parameters):#MeasurePointId, num_tramas, assetId):

    data           = []
    date           = []
    lista_maquinas = []
    MeasurePointId = Parameters['Localizacion']
    num_tramas     = Parameters['NumeroTramas']
    assetId        = Parameters['IdAsset']
    format = "%Y-%m-%dT%H:%M:%S"

    # get waveform data
#    
#    if input_data[0]['waveform'][0]['IdPosicion'] == Parameters['Localizacion']:
#            flag_pos = 0
#    if input_data[0]['waveform'][1]['IdPosicion'] == Parameters['Localizacion']:
#            flag_pos = 1
            
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
            #print(root)
            print(res.Props.iloc[0][0]['Value'])
            print(res.Props.iloc[0][4]['Value'])
            cal_factor = np.float(res.Props.iloc[0][4]['Value'])
            data.append(np.asarray(res.Value.values[0])*cal_factor)

            fecha = res.ServerTimeStamp.values[0]
            datetime_obj = datetime.datetime.strptime(fecha[0:19],format) # pierde la parte decimal de los segundos
                                                                          # sumo la patrte decimal de los segundos
            segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1])
            # para recuperar (EXACTO) "YY MM DD hh mm ss" ==> datetime.datetime.fromtimestamp(segundos)

            print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )

            fecha = segundos
            date.append(fecha)

    df_out     = DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out

#------------------------------------------------------------------------------

def Load_Vibration_Data_Global_Pumps(parameters):
    f_fin   = 300
    i_f_fin = int(l*f_fin/fs)
    f       = np.arange(l)/(l-1)*fs
    f       = f[0:i_f_fin]
    
    
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
        df_speed_BA4,df_SPEED_BA4  = velocity(df_accel_BA4)
        df_speed_BV4,df_SPEED_BV4  = velocity(df_accel_BV4)
        
    #return df_accel_BH3,df_accel_BA4,df_accel_BV4
    return df_speed_BH3,df_SPEED_BH3,df_speed_BA4,df_SPEED_BA4,df_speed_BV4,df_SPEED_BV4

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def Load_Vibration_Data_From_Get_Pumps(input_data, Parameters):#MeasurePointId, num_tramas, assetId):
    #BH3 (horizontal), BA4 (axial) y BV4 (radial)
    
    date           = []
    MeasurePointId = Parameters['Localizacion']    
    assetId        = Parameters['IdAsset']
    format = "%Y-%m-%dT%H:%M:%S"

   
    
    frames_BH3   = []
    frames_BA4   = []
    frames_BV4   = []
    lista_frames = (frames_BH3,frames_BA4,frames_BV4)
    data_BH3     = []
    data_BA4     = []
    data_BV4     = []
    lista_data   = (data_BH3,data_BA4,data_BV4)
    date_BH3     = []
    date_BA4     = []
    date_BV4     = []
    lista_date   = (date_BH3,date_BA4,date_BV4)
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

    df_out_BH3     = DataFrame(data=lista_data[0], index=lista_date[0])
    df_out_BH3.sort_index(inplace=True)
    df_out_BA4     = DataFrame(data=lista_data[1], index=lista_date[1])
    df_out_BA4.sort_index(inplace=True)
    df_out_BV4     = DataFrame(data=lista_data[2], index=lista_date[2])
    df_out_BV4.sort_index(inplace=True)
    
    return df_out_BH3,df_out_BA4,df_out_BV4

#------------------------------------------------------------------------------
def velocity(df_in):
    #---- devuleve el espectro con ventana de hanning y corregida en potencia
    #---- es decir multiplicado por el factor 1.63    
    l         = df_in.shape[1]
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81
    
    df_speed  = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    df_abs    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    df_angle  = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    
    for counter,indice in enumerate(df_in.index):
        trace                   = df_in.iloc[counter].values- np.mean(df_in.iloc[counter].values)
        trace                   = np.cumsum (G*1000*trace/fs)    #---velocidad-     
        trace_speed             = signal.filtfilt(b, a, trace) 
        df_speed.iloc[counter]  = trace_speed
                                    #------------------------------------------
        trace_SPEED             = 1.63 * np.fft.fft(trace_speed* hann/l)
        df_abs.iloc[counter]    = np.abs  ( trace_SPEED )
        df_angle.iloc[counter]  = np.angle( trace_SPEED )
        #print(np.std())
    return df_speed,df_abs#,df_angle
#------------------------------------------------------------------------------
def Plain_Bearing_Clearance(df_in):
    print('-------------------------Clearance Failure--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Plain Bearing Clearance Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Plain Bearing Clearance Failure'] = 'No vibration detected'
        else:
            v_1x   = df_in.iloc[i]['RMS 1.0']
            v_2x   = df_in.iloc[i]['RMS 2.0']
            v_3x   = df_in.iloc[i]['RMS 3.0']
    
            v_0_5x = df_in.iloc[i]['RMS 1/2']
            v_1_5x = df_in.iloc[i]['RMS 3/2']
            v_2_5x = df_in.iloc[i]['RMS 5/2']
                                                #-------1.0x 2.0x 3.0x decreciente
            a1 = PEAKS(v_1x,v_2x,v_3x)            and v_1x >v_2x > v_3x
                                                # --2.0x >2% 1.0x and 3.0x >2% 1.0x
            a2 = PEAKS(v_1x,v_2x,v_3x)            and (v_2x > 0.02 * v_1x) and (v_3x > 0.02 * v_1x)
            A  = a1 and a2
                                                #-------0.5x 1.5x 2.5x decreciente
            b1 = PEAKS(v_0_5x,v_1_5x,v_2_5x)      and v_0_5x > v_1_5x > v_2_5x
                                                # ------0.5x >2% 1.0x and 1.5x > 2% 1.0x and 2.5x > 2% 1.0x
            b2 = PEAKS(v_0_5x,v_1x,v_1_5x,v_2_5x) and (v_0_5x > 0.02 * v_1x) and (v_1_5x > 0.02 * v_1x) and (v_2_5x > 0.02 * v_1x)
            B  = b1 and b2
            
            df_in.loc[df_in.index[i],'$Plain Bearing Clearance Failure'] = Truth_Table( not(A) and not(B) , A or B , A and B)
    return df_in
#------------------------------------------------------------------------------
def Blower_Wheel_Unbalance(df_in):
    print('--------------------Blower Wheel Unbalance Failure------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Blower Wheel Unbalance Failure'] = none_list

    for i in range (n_traces):
        #print(df_in.iloc[i].values[1:8])
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Blower Wheel Unbalance Failure'] = 'No vibration detected'
        else:
                                                #---1X meno que el umbral
            A_peaks = PK(df_in.iloc[i]['RMS 1.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 1.0'] < 4
                                                #---El 15% 1x < resto armonicos.
                                                #   es decir 1X no es dominante
            B_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2nd Highest'])
            B       =  B_peaks and df_in.iloc[i]['RMS 1.0'] * 0.15 < df_in.iloc[i]['RMS 2nd Highest']
    
                                                #--------------------------Green
            if A and B:
                df_in.loc[df_in.index[i],'$Blower Wheel Unbalance Failure'] = 'Green'
                                                #--------------------------yellow
                                                #   Xor = cualquiera de ellas
                                                #        pero no ambas
            if (A == False) ^   (B == False):
                df_in.loc[df_in.index[i],'$Blower Wheel Unbalance Failure'] = 'Yellow'
                                                #--------------------------Red
                                                # las dos falsas
            if (A == False) and (B == False):
                df_in.loc[df_in.index[i],'$Blower Wheel Unbalance Failure'] = 'Green'
    return df_in
#------------------------------------------------------------------------------
def Oil_Whirl(df_in):
    print('---------------------------Oil Whirl Failure-----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Oil Whirl Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Oil Whirl Failure'] = 'No vibration detected'
        else:
                                                #-----------green-----------------
                                                # no detected Peak in '0.38-0.48'
            A = df_in.iloc[i]['RMS Oil Whirl'] > E1
                                                        #-----------yellow-----------------
                                                # Detected Peak in '0.38-0.48'
                                                #         but
                                                # Peak in '0.38-0.48' < 2% 1.0x
            B_peaks = PEAKS(df_in.iloc[i]['RMS Oil Whirl'],df_in.iloc[i]['RMS 1.0']) 
            B       = B_peaks and df_in.iloc[i]['RMS Oil Whirl'] > 0.02 * df_in.iloc[i]['RMS 1.0']
            
            df_in.loc[df_in.index[i],'$Oil Whirl Failure'] = Truth_Table( not(A) , A and (not B) , A and B)

            if df_in.loc[df_in.index[i],'$Oil Whirl Failure'] == 'None':
                print ('Fallo en:', i)
                print (df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS Oil Whirl'],A,B)
    return df_in
#------------------------------------------------------------------------------
def Oil_Whip(df_in):
    print('---------------------------Oil Whip Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
  

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Oil Whip Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Oil Whip Failure'] = 'No vibration detected'
        else:
            A       = (df_in.iloc[i]['RMS 1/2'] >= E1 and df_in.iloc[i]['BW 1/2'] >= 4)  and  ((df_in.iloc[i]['RMS 5/2'] >= E1) and df_in.iloc[i]['BW 2.5'] >= 4)
            B_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 1/2']) 
            B       = B_peaks and df_in.iloc[i]['RMS 1/2' ] > 0.02 *  df_in.iloc[i]['RMS 1.0']
            C_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 5/2'])
            C       = C_peaks and df_in.iloc[i]['RMS 5/2' ] > 0.02 *  df_in.iloc[i]['RMS 1.0']
            #print(A,B,C)
                                                 #  Tabla de verdad progresiva
                                                 #  puede empezar siendo verde,
                                                 #  acabar siendo rojo
    
                                                 #-----------green-----------------
                                                 # 2H BW at 0.5 = 0 and 2H BW at 2.5 = 0
    
            if A == False and ( (B and C) == False ):
                df_in.loc[df_in.index[i],'$Oil Whip Failure'] = 'Green'
                                                 #---------yellow------------------
                                                 # 2H BW at 0.5 > 0
                                                 # 2H BW at 2.5 > 0
                                                 # 2H BW at 0.5 >2% 1.0x
                                                 # 2H BW at 2.5 >2% 1.0x
            if A ^ ((B ^ C)) :
                df_in.loc[df_in.index[i],'$Oil Whip Failure'] = 'Yellow'
                                                 #-----------red-------------------
                                                 #     2H BW at 0.5 >2% 1.0x
                                                 #           AND
                                                 #     2H BW at 2.5 >2% 1.0x
            if A and B and C:
                df_in.loc[df_in.index[i],'$Oil Whip Failure'] = 'Red'
    return df_in
#------------------------------------------------------------------------------
def Blade_Faults(df_in):
    print('-----------------------Blade Faults Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
   
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Blade Faults Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Blade Faults Failure'] = 'No vibration detected'
        else:
            A = PK(df_in.iloc[i]['RMS 12.0'])
            B = PEAKS(df_in.iloc[i]['RMS 12.0'],df_in.iloc[i]['RMS 24.0']) 
            C = PK(df_in.iloc[i]['RMS 12.0'])       and ( PK(df_in.iloc[i]['RMS 11.0']) or PK(df_in.iloc[i]['RMS 13.0']) )
            D = C and PK(df_in.iloc[i]['RMS 24.0']) 
            E = C and PK(df_in.iloc[i]['RMS 24.0']) and ( PK(df_in.iloc[i]['RMS 23.0']) or PK(df_in.iloc[i]['RMS 25.0']) )
            F = df_in.iloc[i]['RMS 12.0'] < E1 and df_in.iloc[i]['RMS 24.0'] < E1
            #print('Blade Faults         ',A,B,C,D,E,F)
                                                 #  Tabla de verdad progresiva
                                                 #  puede empezar siendo verde,
                                                 #  acabar siendo rojo
            df_in.loc[df_in.index[i],'$Blade Faults Failure'] = Truth_Table( A or B or F , C or D , E)

    return df_in
#------------------------------------------------------------------------------
def Flow_Turbulence(df_in):
    print('---------------------Flow Turbulence Failure-----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Flow Turbulence Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Flow Turbulence Failure'] = 'No vibration detected'
        else:
            A       = df_in.iloc[i]['RMS Flow T.'] <= 0.2
            B_peaks = PK(df_in.iloc[i]['RMS 1.0'])
            B       = B_peaks and (0.2 <= df_in.iloc[i]['RMS Flow T.'] <= df_in.iloc[i]['RMS 1.0'])
            C_peaks = PK(df_in.iloc[i]['RMS 1.0'])
            C       = C_peaks and (df_in.iloc[i]['RMS Flow T.'] >  df_in.iloc[i]['RMS 1.0'])
            #print('Flow Tur.           ',A,B,C)
            
            df_in.loc[df_in.index[i],'$Flow Turbulence Failure'] = Truth_Table(A,B,C)
           
    return df_in
#------------------------------------------------------------------------------
def Plain_Bearing_Block_Looseness(df_in):
    print('-----------------------PBB looseness Failure-----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$PBB looseness Failure'] = none_list
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$PBB looseness Failure'] = 'No vibration detected'
        else:
            A_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 1.0'] <df_in.iloc[i]['RMS 2.0'] > df_in.iloc[i]['RMS 3.0']
            B       = PEAKS(df_in.iloc[i]['RMS 1/2'],df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 1/4'])
            #print('Plain Bearin block   ',A,B)
            df_in.loc[df_in.index[i],'$PBB looseness Failure'] = Truth_Table(not A and not B , A or B , A and B)

    return df_in
#------------------------------------------------------------------------------
def Shaft_Misaligments(df_in):
    print('--------------------------Shaft Mis. Failure-----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Shaft Mis. Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Shaft Mis. Failure'] = 'No vibration detected'
        else:
            A_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 2.0'] < 0.5 *  df_in.iloc[i]['RMS 1.0']
            B_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            B       = B_peaks and 1.5 *df_in.iloc[i]['RMS 1.0'] >        df_in.iloc[i]['RMS 2.0'] > 0.5 *df_in.iloc[i]['RMS 1.0']
            C_peaks = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            C       = C_peaks and 1.5 *df_in.iloc[i]['RMS 1.0'] <        df_in.iloc[i]['RMS 2.0']
            D       = PEAKS(df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'],df_in.iloc[i]['RMS 5.0'])
            #print('Shaft miss          ',A,B,C,D)
            df_in.loc[df_in.index[i],'$Shaft Mis. Failure'] = Truth_Table(A or not D , B and D , C and D)
    return df_in
#------------------------------------------------------------------------------
def Pressure_Pulsations(df_in):
    print('-------------------------Pressure P. Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Pressure P. Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Pressure P. Failure'] = 'No vibration detected'
        else:
            A       = PEAKS(df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 2/3'],df_in.iloc[i]['RMS 4/3'],df_in.iloc[i]['RMS 5/3'])
            B_peaks = PEAKS(df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 4/3'],df_in.iloc[i]['RMS 8/3'],df_in.iloc[i]['RMS 4.0']) 
            B       = B_peaks and (df_in.iloc[i]['RMS 4/3'] > df_in.iloc[i]['RMS 1/3']) and (df_in.iloc[i]['RMS 8/3'] > df_in.iloc[i]['RMS 1/3']) and (df_in.iloc[i]['RMS 4.0'] > df_in.iloc[i]['RMS 1/3'])
            #print('Shaft miss          ',A,B,C,D)
            df_in.loc[df_in.index[i],'$Pressure P. Failure'] = Truth_Table(not A , A , A and B)
       
    return df_in
#------------------------------------------------------------------------------
def Surge_Effect(df_in):
    print('------------------------Surge E. Failure---------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Surge E. Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Surge E. Failure'] = 'No vibration detected'
        else:
            A = PK(df_in.iloc[i]['RMS Surge E. 0.33x 0.5x'])
            B = PK(df_in.iloc[i]['RMS Surge E. 12/20k'])
            
            df_in.loc[df_in.index[i],'$Surge E. Failure'] = Truth_Table(not A , A , A and B)
    return df_in
#------------------------------------------------------------------------------
def Severe_Misaligment(df_in):
    print('----------------------------Severe Mis. Failure--------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Severe Mis. Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Severe Mis. Failure'] = 'No vibration detected'
        else:
            counter_A = 0
            for m in [2,3,4,5,6,7,8,9,10]:
                if (df_in.iloc[i]['RMS '+str(m)+'.0'] > 0.02*df_in.iloc[i]['RMS 1.0']) and PK(df_in.iloc[i]['RMS 1.0']):
                    counter_A = counter_A+1
            A         = counter_A >= 3
    
            counter_B = 0
            for m in ['5/2','7/2','9/2','11/2','13/2','15/2','17/2','19/2']:
                if (df_in.iloc[i]['RMS '          +m] > 0.02*df_in.iloc[i]['RMS 1.0']) and PK(df_in.iloc[i]['RMS 1.0']):
                    counter_B = counter_B+1
            B         = counter_B >= 3
    
            C_peaks   = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            C         = C_peaks and df_in.iloc[i]['RMS 2.0'] > df_in.iloc[i]['RMS 1.0']
            
            if not A:
                df_in.loc[df_in.index[i],'$Severe Mis. Failure'] = 'Green'
            if A or B:               #---27/2/19 Dammika por tlfn cambio Xor por or
                df_in.loc[df_in.index[i],'$Severe Mis. Failure'] = 'Yellow'
            if A and B and C:
                df_in.loc[df_in.index[i],'$Severe Mis. Failure'] = 'Red'
    #        print(df_in.loc[df_in.index[i],'$Severe Mis. Failure'])
    return df_in
#------------------------------------------------------------------------------
def Loose_Bedplate(df_in):
    print('------------------Loose Bedplate Failure---------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Loose Bedplate Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Loose Bedplate Failure'] = 'No vibration detected'
        else:
            A       =                                  2.5 > df_in.iloc[i]['RMS 1.0'] > 0
            B       = PK(df_in.iloc[i]['RMS 1.0']) and 2.5 < df_in.iloc[i]['RMS 1.0'] < 6.00 
            C       = PK(df_in.iloc[i]['RMS 1.0']) and 6.0 < df_in.iloc[i]['RMS 1.0']
            D_peaks = PEAKS(df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'])
            D       = D_peaks and df_in.iloc[i]['RMS 3.0'] > df_in.iloc[i]['RMS 2.0']
            #print ('Loose Bedplate',A,B,C,D)
            df_in.loc[df_in.index[i],'$Loose Bedplate Failure'] = Truth_Table(A , B ^ C , C and D)              
            if df_in.loc[df_in.index[i],'$Loose Bedplate Failure'] == 'None':
                print ('Fallo en:', i)
                print ('Loose Bedplate',df_in.iloc[i]['RMS 1.0'],A,B,C,D)
    return df_in

#------------------------------------------------------------------------------
def Pillow_Block_Loseness(df_in):
    print('------------------------PB Loseness Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$PB Loseness Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$PB Loseness Failure'] = 'No vibration detected'
        else:

            A1 = PEAKS(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'])
            A2 = df_in.iloc[i]['RMS 2.0'] < df_in.iloc[i]['RMS 1.0'] > df_in.iloc[i]['RMS 3.0'] 
            A  = A1 and A2
            B  = PEAKS(df_in.iloc[i]['RMS 1/2'],df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 1/4'])
            
            df_in.loc[df_in.index[i],'$PB Loseness Failure'] = Truth_Table( (not A) and (not B) , A ^ B , A and B)
    return df_in
#==============================================================================
def Ball_Bearing_Outer_Race_Defects_22217C(df_in):
    print('-------------------Ball B. O. Race D. Failure_22217C---------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball B. O. Race D. Failure_22217C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. O. Race D. Failure_22217C'] = none_list
    else:
        print('Columna previemente creada')
        
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22217C'] = 'No vibration detected'
        else:

            a1 = (df_in.iloc[i]['RMS BPFO1'] < E1) 
            a2 = (df_in.iloc[i]['RMS BPFO1'] > E1) and (df_in.iloc[i]['RMS 2*BPFO1'] < E1) and (df_in.iloc[i]['RMS 3*BPFO1'] < E1) and (df_in.iloc[i]['RMS 4*BPFO1'] < E1)
            A  = a1 or a2
            B  = PEAKS(df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'])
            C  = PEAKS(df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'],df_in.iloc[i]['RMS 3*BPFO1'])
            D  = PEAKS(df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'],df_in.iloc[i]['RMS 3*BPFO1'],df_in.iloc[i]['RMS 4*BPFO1'])
            
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22217C']    = Truth_Table(A,B ^ C,D)
            
            if df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22217C'] == 'None':
                print ('Fallo en:', i)
    return df_in
#------------------------------------------------------------------------------    
def Ball_Bearing_Outer_Race_Defects_22219C(df_in):
    print('-------------------Ball B. O. Race D. Failure_22219C---------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball B. O. Race D. Failure_22219C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. O. Race D. Failure_22219C'] = none_list
    else:
        print('Columna previemente creada')
    

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22219C'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BPFO2'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFO2'] > E1) and (df_in.iloc[i]['RMS 2*BPFO2'] < E1) and (df_in.iloc[i]['RMS 3*BPFO2'] < E1) and (df_in.iloc[i]['RMS 4*BPFO2'] < E1)
            A = a1 or a2
            B = PEAKS(df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'])
            C = PEAKS(df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'],df_in.iloc[i]['RMS 3*BPFO2'])
            D = PEAKS(df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'],df_in.iloc[i]['RMS 3*BPFO2'],df_in.iloc[i]['RMS 4*BPFO2'])
            
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22219C'] = Truth_Table(A,B ^ C,D)
            
            if df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_22219C'] == 'None':
                print ('Fallo en:', i)
    return df_in
#==============================================================================
def Ball_Bearing_Inner_Race_Defects_22217C(df_in):
    print('-------------Ball B. I. Race D. Failure_22217C-------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. I. Race D. Failure_22217C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. I. Race D. Failure_22217C'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22217C'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BPFI1'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFI1'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1'] < E1)
            A = a1 or a2 
            B = (df_in.iloc[i]['RMS BPFI1']   > E1) and (df_in.iloc[i]['RMS 2*BPFI1']  > E1)
            C = (df_in.iloc[i]['RMS BPFI1']   > E1) and (df_in.iloc[i]['RMS BPFI1+f']   > E1) and (df_in.iloc[i]['RMS BPFI1-f']   > E1)
            D = (df_in.iloc[i]['RMS 2*BPFI1'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1+f'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1-f'] > E1)      
    
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22217C'] = Truth_Table(A,B ^ C,D)
            
            if df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22217C'] == 'None':
                print ('Fallo en:', i)
    return df_in
#------------------------------------------------------------------------------
def Ball_Bearing_Inner_Race_Defects_22219C(df_in):
    print('-------------Ball B. I. Race D. Failure_22219C-------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. I. Race D. Failure_22219C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. I. Race D. Failure_22219C'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22219C'] = 'No vibration detected'
        else:         
            a1 = (df_in.iloc[i]['RMS BPFI2'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFI2'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2'] < E1)
            A = a1 or a2 
            B = (df_in.iloc[i]['RMS BPFI2']   > E1) and (df_in.iloc[i]['RMS 2*BPFI2']   > E1)
            C = (df_in.iloc[i]['RMS BPFI2']   > E1) and (df_in.iloc[i]['RMS BPFI2+f']   > E1) and (df_in.iloc[i]['RMS BPFI2-f']   > E1)
            D = (df_in.iloc[i]['RMS 2*BPFI2'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2+f'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2-f'] > E1)  
    
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22219C'] = Truth_Table(A,B ^ C,D)
    
            if df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_22219C'] == 'None':
                print ('Fallo en:', i, df_in.iloc[i]['RMS BPFI2'] ,df_in.iloc[i]['RMS 2*BPFI2'] )
                print (df_in.iloc[i]['f BPFI2'] ,df_in.iloc[i]['RMS BPFI2'] )
                print (df_in.iloc[i]['f 2*BPFI2'] ,df_in.iloc[i]['RMS 2*BPFI2'] )
                print('A=',A)
                print('B=',B)
                print('C=',C)
                print('D=',D)
    return df_in
#==============================================================================
def Ball_Bearing_Defect_22217C(df_in):
    print('--------------------------Ball B D. Failure_22217C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball B D. Failure_22217C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B D. Failure_22217C'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B D. Failure_22217C'] = 'No vibration detected'
        else:
    
            a1 = (df_in.iloc[i]['RMS BSF1']   < E1)
            a2 = (df_in.iloc[i]['RMS BSF1']   > E1) and (df_in.iloc[i]['RMS 2*BSF1'] < E1)
            A  = a1 or a2 
            B  = PEAKS(df_in.iloc[i]['RMS BSF1'],df_in.iloc[i]['RMS 2*BSF1'])
            C  = PEAKS(df_in.iloc[i]['RMS BSF1']  ,df_in.iloc[i]['RMS BSF1+FTF1']  ,df_in.iloc[i]['RMS BSF1-FTF1'])
            D  = PEAKS(df_in.iloc[i]['RMS 2*BSF1'],df_in.iloc[i]['RMS 2*BSF1+FTF1'],df_in.iloc[i]['RMS 2*BSF1-FTF1'])
            
            df_in.loc[df_in.index[i],'$Ball B D. Failure_22217C'] = Truth_Table(A,B ^ C,D)
    
            if df_in.loc[df_in.index[i],'$Ball B D. Failure_22217C'] == 'None':
                print ('Fallo en:', i)
                print ('No peak at BSF1           ',a1,df_in.iloc[i]['f BSF1']   ,df_in.iloc[i]['RMS BSF1'] )
                print ('Peak at BSF1, no harmonics',a2,df_in.iloc[i]['f 2*BSF1'] ,df_in.iloc[i]['RMS 2*BSF1'] )
                print('A=',A)
                print('B=',B)
                print('C=',C)
                print('D=',D)
            #print('________________',df_in.loc[df_in.index[i],'$Ball B D. Failure_22217C'])
    return df_in
#------------------------------------------------------------------------------
def Ball_Bearing_Defect_22219C(df_in):
    print('--------------------------Ball B D. Failure_22219C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    if ('$Ball B D. Failure_22219C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B D. Failure_22219C'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces): 
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B D. Failure_22219C'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BSF2']   < E1)
            a2 = (df_in.iloc[i]['RMS BSF2']   > E1) and (df_in.iloc[i]['RMS 2*BSF2'] < E1)
            A = a1 or a2 
            B = PEAKS(df_in.iloc[i]['RMS BSF2']  ,df_in.iloc[i]['RMS 2*BSF2'])
            C = PEAKS(df_in.iloc[i]['RMS BSF2']  ,df_in.iloc[i]['RMS BSF2+FTF2']  ,df_in.iloc[i]['RMS BSF2-FTF2'])
            D = PEAKS(df_in.iloc[i]['RMS 2*BSF2'],df_in.iloc[i]['RMS 2*BSF2+FTF2'],df_in.iloc[i]['RMS 2*BSF2-FTF2'])
            
            df_in.loc[df_in.index[i],'$Ball B D. Failure_22219C'] = Truth_Table(A,B ^ C,D)
            if  df_in.loc[df_in.index[i],'$Ball B D. Failure_22219C'] == 'None':
                print ('Fallo en:', i)
            #print('________________',df_in.loc[df_in.index[i],'$Ball B D. Failure_22219C'])
    return df_in
#==============================================================================
def Ball_Bearing_Cage_Defect_22217C(df_in):
    print('------------Ball B. Cage D. Failure_22217C-----------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. Cage D. Failure_22217C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. Cage D. Failure_22217C'] = none_list
    else:
        print('Columna previemente creada')


    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22217C'] = 'No vibration detected'
        else:
            a1 = PK(df_in.iloc[i]['RMS 1.0'])  
            a2 = df_in.iloc[i]['RMS FTF1'] < df_in.iloc[i]['RMS 1.0']  
            b2 = df_in.iloc[i]['RMS FTF1'] > df_in.iloc[i]['RMS 2*FTF1'] < df_in.iloc[i]['RMS 1.0'] 
            c2 = df_in.iloc[i]['RMS FTF1'] > df_in.iloc[i]['RMS 2*FTF1'] > df_in.iloc[i]['RMS 1.0'] > df_in.iloc[i]['RMS 3*FTF1'] > df_in.iloc[i]['RMS 4*FTF1']
            
            A  = a1 and a2
            B  = a1 and b2
            C  = a1 and c2
            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22217C'] = Truth_Table(A,B,C)
            if df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22217C'] == 'None':
                print ('Fallo en:', i)
            #print('________________',df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22217C'])
    return df_in

#------------------------------------------------------------------------------
def Ball_Bearing_Cage_Defect_22219C(df_in):
    print('------------Ball B. Cage D. Failure_22219C-----------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. Cage D. Failure_22219C' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. Cage D. Failure_22219C'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'] = 'No vibration detected'
        else:
            a1 = PK(df_in.iloc[i]['RMS 1.0'])  
            a2 = df_in.iloc[i]['RMS FTF2'] < df_in.iloc[i]['RMS 1.0']  
            b2 = df_in.iloc[i]['RMS FTF2'] > df_in.iloc[i]['RMS 2*FTF2'] < df_in.iloc[i]['RMS 1.0'] 
            c2 = df_in.iloc[i]['RMS FTF2'] > df_in.iloc[i]['RMS 2*FTF2'] > df_in.iloc[i]['RMS 1.0'] > df_in.iloc[i]['RMS 3*FTF2'] > df_in.iloc[i]['RMS 4*FTF2']
            A  = a1 and a2
            B  = a1 and b2
            C  = a1 and c2
            
            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'] = Truth_Table(A,B,C)
            if df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'] == 'None':
                print ('Fallo en:', i)
            #print('________________',df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'])
    return df_in
#------------------------------------------------------------------------------
def Recirculation_in_pump(df_in):
    print('-----------------Recirculation_in_pump-----------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Recirculation_in_pump' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Recirculation_in_pump'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Recirculation_in_pump'] = 'No vibration detected'
        else:
   
            A  = df_in.iloc[i]['RMS R. Pump'] <= 0.3
            B  = 0.3 < df_in.iloc[i]['RMS R. Pump'] <= df_in.iloc[i]['RMS 1.0']
            C  = df_in.iloc[i]['RMS R. Pump'] > df_in.iloc[i]['RMS 1.0']
            
            df_in.loc[df_in.index[i],'$Recirculation_in_pump'] = Truth_Table(A,B,C)
            if df_in.loc[df_in.index[i],'$Recirculation_in_pump'] == 'None':
                print ('Fallo en:', i)
    return df_in

#------------------------------------------------------------------------------

def Impeller_Rotor_Unbalance(df_in):
    print('-----------------Impeller_Rotor_Unbalance--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Impeller_Rotor_Unbalance' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Impeller_Rotor_Unbalance'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Impeller_Rotor_Unbalance'] = 'No vibration detected'
        else:
      
            A  = 0.15 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2nd Highest']
            B  = df_in.iloc[i]['RMS 1.0'] > 4
            
            df_in.loc[df_in.index[i],'$Impeller_Rotor_Unbalance'] = Truth_Table( A and B,A ^ B,(not A) and (not B))            
            if df_in.loc[df_in.index[i],'$Impeller_Rotor_Unbalance'] == 'None':
                print ('Fallo en:', i)
            #print('________________',df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'])
    return df_in
  
#------------------------------------------------------------------------------
def Shaft_misaligment_Radial(df_in):
    print('-----------------Shaft_misaligment_Radial--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Shaft_misaligment_Radial' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Shaft_misaligment_Radial'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Shaft_misaligment_Radial'] = 'No vibration detected'
        else:       
            
            No_Peaks = NO_PEAKS(df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'])
            A        = PK(df_in.iloc[i]['RMS 1.0']) and (0.5 * df_in.iloc[i]['RMS 1.0'] > df_in.iloc[i]['RMS 2.0']) or No_Peaks
            B        = PK(df_in.iloc[i]['RMS 1.0']) and (0.5 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2.0'] < 1.5 * df_in.iloc[i]['RMS 1.0'])
            C        = PK(df_in.iloc[i]['RMS 1.0']) and (1.5 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2.0'] )
            D        = PEAKS(df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'])
    
            #print (A,B,C,D)
            df_in.loc[df_in.index[i],'$Shaft_misaligment_Radial'] = Truth_Table(A and (not D),B or D,C and D)
            if df_in.loc[df_in.index[i],'$Shaft_misaligment_Radial'] == 'None':
                print ('Fallo en:', i)
                print (df_in.iloc[i]['RMS 1.0'])
                print (df_in.iloc[i]['RMS 2.0'], )
                print (df_in.iloc[i]['RMS 3.0'])
                print (df_in.iloc[i]['RMS 4.0'])
               
            #print('________________',df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_22219C'])
    return df_in    

#------------------------------------------------------------------------------
def extract_index(instante,df_in):
    
    indice_captura =-1
    
    if np.size(df_in) != 0:  #----------------------- hay capturas
        indices = np.zeros(np.size(df_in.index))
        for counter,k in enumerate(df_in.index):
            indices[counter] = time.mktime(k.timetuple())
            
        value = np.min(np.abs(indices-instante))
        #print('time diference in minutes:',value/60)
        if value <= 30*60:
            indice_captura = np.argmin(np.abs(indices-instante))
            
    return indice_captura

def Shaft_misaligment_Axial(df_in_H,df_in_V,df_in_A):
    print('-----------------Shaft_misaligment_Axial--------------------------')
    n_traces   = df_in_A.shape[0]
    none_list  = []

    if ('$Shaft_misaligment_Axial' in df_in_A.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in_A['$Shaft_misaligment_Axial'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        instante = time.mktime((df_in_A.index[i].timetuple()))
        i_H      =extract_index(instante,df_in_H)
        i_V      =extract_index(instante,df_in_V)
        
        if df_in_A.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in_A.loc[df_in_A.index[i],'$Shaft_misaligment_Axial'] = 'No vibration detected'
        else:     
            A  = ( E1 < df_in_A.iloc[i]['RMS 1.0'] < 2.5) and PK(df_in_A.iloc[i]['RMS 2.0']) or  (df_in_A.iloc[i]['RMS 2.0'] < E1 )
            B  = (2.4 < df_in_A.iloc[i]['RMS 1.0'] < 4  ) and PK(df_in_A.iloc[i]['RMS 2.0'])
            C  = (  4 < df_in_A.iloc[i]['RMS 1.0']      ) and PEAKS(df_in_A.iloc[i]['RMS 2.0'],df_in_A.iloc[i]['RMS 3.0'])
            
            if i_H >= 0 and i_V >= 0 :
                D = PEAKS( df_in_H.iloc[i_H]['RMS 1.0'],df_in_H.iloc[i_H]['RMS 2.0']) or PEAKS(df_in_V.iloc[i_V]['RMS 1.0'],df_in_V.iloc[i_V]['RMS 2.0'])
            else:
                #print('no hya dtos radiales')
                D = False
                
            df_in_A.loc[df_in_A.index[i],'$Shaft_misaligment_Axial'] = Truth_Table(A and (not B),A or B,C or D)
            if df_in_A.loc[df_in_A.index[i],'$Shaft_misaligment_Axial'] == 'None':
                print ('Fallo en:', i)
    return df_in_A    

#------------------------------------------------------------------------------
def Hydraulic_Instability(df_in):
    print('-----------------Hydraulic_Instability--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Hydraulic_Instability' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Hydraulic_Instability'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Hydraulic_Instability'] = 'No vibration detected'
        else:     
            A  =  df_in.iloc[i]['RMS Hydr. Inst.'] < E1
            B  = PK(df_in.iloc[i]['RMS 1.0']) and (E1 < df_in.iloc[i]['RMS Hydr. Inst.'] < 0.5 * df_in.iloc[i]['RMS 1.0']) 
            C  = PK(df_in.iloc[i]['RMS 1.0']) and (     df_in.iloc[i]['RMS Hydr. Inst.'] > 0.5 * df_in.iloc[i]['RMS 1.0']) 
            
            df_in.loc[df_in.index[i],'$Hydraulic_Instability'] = Truth_Table(A,B,C)
            
            if df_in.loc[df_in.index[i],'$Hydraulic_Instability'] == 'None':
                print ('Fallo en:', i)
                print(A,B,C)
                print(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS Hydr. Inst.'])
    return df_in    
#==============================================================================
    #-------ATENCION: df_FFT ya viene multiplicada por la ventana de hanning 
    #-------y corregido en potencia, es decir multiplicado por 1.63
def df_Harmonics_old(df_speed,df_FFT,fs):
    print('------------------------------------Extracting fingerprint')
    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]
    
    BPFI1 = 295.080
    BPFO1 = 222.920
    FTF1  = 10.615
    BSF1  = 85.864
    
    BPFI2 = 282.527
    BPFO2 = 210.806
    FTF2  = 10.540
    BSF2  = 82.008
    
    fignerprint_list = [
                        fingerprint('1/2' ,'Peak',1/2 ,0,'relativo'),
                        fingerprint('1.0' ,'Peak',1   ,0,'relativo'),
                        fingerprint('3/2' ,'Peak',1.5 ,0,'relativo'),
                        fingerprint('2.0' ,'Peak',2   ,0,'relativo'),
                        fingerprint('5/2' ,'Peak',2.5 ,0,'relativo'),
                        fingerprint('3.0' ,'Peak',3   ,0,'relativo'),
                        fingerprint('7/2' ,'Peak',3.5 ,0,'relativo'),
                        fingerprint('4.0' ,'Peak',4   ,0,'relativo'),
                        fingerprint('9/2' ,'Peak',4.5 ,0,'relativo'),
                        fingerprint('5.0' ,'Peak',5   ,0,'relativo'),
                        fingerprint('11/2','Peak',5.5 ,0,'relativo'),
                        fingerprint('6.0' ,'Peak',6   ,0,'relativo'),
                        fingerprint('13/2','Peak',6.5 ,0,'relativo'),
                        fingerprint('7.0' ,'Peak',7   ,0,'relativo'),
                        fingerprint('15/2','Peak',7.5 ,0,'relativo'),
                        fingerprint('8.0' ,'Peak',8   ,0,'relativo'),
                        fingerprint('17/2','Peak',8.5 ,0,'relativo'),
                        fingerprint('9.0' ,'Peak',9   ,0,'relativo'),
                        fingerprint('19/2','Peak',9.5 ,0,'relativo'),
                        fingerprint('10.0','Peak',10  ,0,'relativo'),
                        fingerprint('21/2','Peak',10.5,0,'relativo'),

                        fingerprint('2/3' ,'Peak',2/3 ,0,'relativo'),
                        fingerprint('4/3' ,'Peak',4/3 ,0,'relativo'),
                        fingerprint('5/3' ,'Peak',5/3 ,0,'relativo'),
                        fingerprint('8/3' ,'Peak',8/3 ,0,'relativo'),

                        fingerprint('1/3' ,'Peak',1/3 ,0,'relativo'),
                        fingerprint('1/4' ,'Peak',1/4 ,0,'relativo'),

                        fingerprint('11.0','Peak',11  ,0,'relativo'),
                        fingerprint('12.0','Peak',12  ,0,'relativo'),
                        fingerprint('13.0','Peak',13  ,0,'relativo'),
                        fingerprint('23.0','Peak',23  ,0,'relativo'),
                        fingerprint('24.0','Peak',24  ,0,'relativo'),
                        fingerprint('25.0','Peak',25  ,0,'relativo'),

                        fingerprint('2nd Highest'        ,'Peak',0    ,fs/2 ,'absoluto'),

                        fingerprint('Oil Whirl'          ,'Span',0.38 ,0.48 ,'relativo'),
                        fingerprint('500rpm'             ,'Span',0.38 ,0.48 ,'relativo'),
                        fingerprint('Flow T.'            ,'Span',12   ,24   ,'absoluto'),
                        fingerprint('Surge E. 0.33x 0.5x','Span',0.33 ,0.5  ,'relativo'),
                        fingerprint('Surge E. 12/20k'    ,'Span',12000,20000,'absoluto'),
                        
                        fingerprint('BPFO1'              ,'Peak',1*BPFO1     ,0 ,'absoluto'),
                        fingerprint('2*BPFO1'            ,'Peak',2*BPFO1     ,0 ,'absoluto'),
                        fingerprint('3*BPFO1'            ,'Peak',3*BPFO1     ,0 ,'absoluto'),
                        fingerprint('4*BPFO1'            ,'Peak',4*BPFO1     ,0 ,'absoluto'),
                        fingerprint('BPFO2'              ,'Peak',1*BPFO2     ,0 ,'absoluto'),
                        fingerprint('2*BPFO2'            ,'Peak',2*BPFO2     ,0 ,'absoluto'),
                        fingerprint('3*BPFO2'            ,'Peak',3*BPFO2     ,0 ,'absoluto'),
                        fingerprint('4*BPFO2'            ,'Peak',4*BPFO2     ,0 ,'absoluto'),

                        fingerprint('BPFI1'              ,'Peak',BPFI1       ,0 ,'absoluto'),
                        fingerprint('BPFI1+f'            ,'Peak',BPFI1       ,1 ,'mixto'),
                        fingerprint('BPFI1-f'            ,'Peak',BPFI1       ,-1,'mixto'),
                        fingerprint('2*BPFI1'            ,'Peak',2*BPFI1     ,0 ,'absoluto'),
                        fingerprint('2*BPFI1+f'          ,'Peak',2*BPFI1     ,1 ,'mixto'),
                        fingerprint('2*BPFI1-f'          ,'Peak',2*BPFI1     ,-1,'mixto'),
                        
                        fingerprint('BPFI2'              ,'Peak',BPFI2       ,0 ,'absoluto'),
                        fingerprint('BPFI2+f'            ,'Peak',BPFI2       ,1 ,'mixto'),
                        fingerprint('BPFI2-f'            ,'Peak',BPFI2       ,-1,'mixto'),
                        fingerprint('2*BPFI2'            ,'Peak',2*BPFI2     ,0 ,'absoluto'),
                        fingerprint('2*BPFI2+f'          ,'Peak',2*BPFI2     ,1 ,'mixto'),
                        fingerprint('2*BPFI2-f'          ,'Peak',2*BPFI2     ,-1,'mixto'),
                        
                        fingerprint('BSF1'               ,'Peak',BSF1        ,0 ,'absoluto'),
                        fingerprint('BSF1-FTF1'          ,'Peak',BSF1-FTF1   ,0 ,'absoluto'),
                        fingerprint('BSF1+FTF1'          ,'Peak',BSF1+FTF1   ,0 ,'absoluto'),
                        fingerprint('2*BSF1'             ,'Peak',2*BSF1      ,0 ,'absoluto'),
                        fingerprint('2*BSF1-FTF1'        ,'Peak',2*BSF1-FTF1 ,0 ,'absoluto'),
                        fingerprint('2*BSF1+FTF1'        ,'Peak',2*BSF1+FTF1 ,0 ,'absoluto'),
                        
                        fingerprint('BSF2'               ,'Peak',BSF2        ,0 ,'absoluto'),
                        fingerprint('BSF2-FTF2'          ,'Peak',BSF2-FTF2   ,0 ,'absoluto'),
                        fingerprint('BSF2+FTF2'          ,'Peak',BSF2+FTF2   ,0 ,'absoluto'),
                        fingerprint('2*BSF2'             ,'Peak',2*BSF2      ,0 ,'absoluto'),
                        fingerprint('2*BSF2-FTF2'        ,'Peak',2*BSF2-FTF2 ,0 ,'absoluto'),
                        fingerprint('2*BSF2+FTF2'        ,'Peak',2*BSF2+FTF2 ,0 ,'absoluto'),
                        
                        fingerprint('FTF1'               ,'Peak',FTF1        ,0 ,'absoluto'),
                        fingerprint('2*FTF1'             ,'Peak',2*FTF1      ,0 ,'absoluto'),
                        fingerprint('3*FTF1'             ,'Peak',3*FTF1      ,0 ,'absoluto'),
                        fingerprint('4*FTF1'             ,'Peak',4*FTF1      ,0 ,'absoluto'),
                        fingerprint('FTF2'               ,'Peak',FTF2        ,0 ,'absoluto'),
                        fingerprint('2*FTF2'             ,'Peak',2*FTF2      ,0 ,'absoluto'),
                        fingerprint('3*FTF2'             ,'Peak',3*FTF2      ,0 ,'absoluto'),
                        fingerprint('4*FTF2'             ,'Peak',4*FTF2      ,0 ,'absoluto'),
                        
                        fingerprint('R. Pump'            ,'Span',0   ,10   ,'absoluto'),
                        
                        ]
    
    

    fecha = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))

    columnas = []
    word =''
    for k in fignerprint_list:
        word = k.label
        columnas.append('i '+word)
        columnas.append('Point '+word)
        columnas.append('RMS '+word)
        columnas.append('f '+word)
        columnas.append('BW '+word)
        columnas.append('n_s '+word)
        columnas.append('n_e '+word)
    
    columnas    = ['RMS (mm/s) t','RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))

    #f_1x        = 1480/60
    #f_1x        = 24.7
    f_1x_low    = 24
    f_1x_high   = 26
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    
    i_f_1x_low  = int(np.round(f_1x_low*l/fs))
    i_f_1x_high = int(np.round(f_1x_high*l/fs))


    for medida in range(n_traces):
        #print ('Medida numero',df_harm.index[medida])
        
        RMS_freq                             = np.sqrt(np.sum( df_FFT.iloc[medida].values**2 ) )
        RMS_time                             = np.std (df_speed.iloc[medida].values)
        df_harm.iloc[medida]['RMS (mm/s) t'] = RMS_time
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        
        sptrm_C                              = df_FFT.iloc[medida].values * np.sqrt(2) 
                                            # integramos en 1º z Nyquist por eso el voltaje x raiz(2)         
        
#        RMS_freq2                            = np.sqrt(np.sum( sptrm_C[0:l_mitad]**2))
#        print('Diferencia en RMS', RMS_freq,RMS_time,RMS_freq2)
#        if 100*np.abs(RMS2-RMS1)/RMS1 > 5 or 100*np.abs(RMS_teorica-RMS1)/RMS1 > 5:
#            print('Diferencia en RMS', RMS1,RMS2,RMS_teorica)
#            print(100*np.abs(RMS2-RMS1)/RMS1,100*np.abs(RMS_teorica-RMS1)/RMS1)
        
        #print( np.sqrt(np.sum(sptrm_C[0:l_mitad]**2) ))
        #n_maxi_C                            = np.argmax(sptrm_C[0:l_mitad])
        indexes, properties                 = find_peaks(sptrm_C[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
        array_peaks                         = sptrm_C[indexes]

        indexes_f_1x, properties_f_1x       = find_peaks(sptrm_C[i_f_1x_low:i_f_1x_high],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
        if np.size(indexes_f_1x) > 0:
            values_f_1x                     = sptrm_C[i_f_1x_low + indexes_f_1x]
            offset                          = np.argmax(values_f_1x)
            offset_index                    = indexes_f_1x[offset]
        else:
            offset_index                    = 0
        
        
        #print(values_f_1x,f[i_f_1x_low+indexes_f_1x[offset]],np.max(sptrm_C[i_f_1x_low:i_f_1x_high]))
        #if 24<f[n_maxi_C]<25.5:
            #f_1x       = f[n_maxi_C]
        if 24<f[i_f_1x_low + offset_index]<25.5 and np.size(array_peaks) > 0 :
            f_1x             = f[i_f_1x_low + offset_index]
            #----------------------------------------------------------------------------------------- El segundopico mas grande
            i_L              = np.argmax(array_peaks)
            array_peaks[i_L] = 0
            i_L2             = np.argmax(array_peaks)

            df_harm.iloc[medida]['i 2nd Highest']     = indexes[i_L2]
            df_harm.iloc[medida]['Point 2nd Highest'] = sptrm_C[indexes[i_L2]]
            df_harm.iloc[medida]['RMS 2nd Highest']   = np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 ))
            df_harm.iloc[medida]['f 2nd Highest']     = f[indexes[i_L2]]
            df_harm.iloc[medida]['BW 2nd Highest']    = properties["widths"][i_L2]
            #print(np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 )),sptrm_C[indexes[i_L2]])

                                             #----------------------------------
            delta = 0.75 #-----------en Hz
            fa = 0
            fb = 0
            for counter,k in enumerate(indexes) :
                for h in fignerprint_list:
                    if h.tipo !='2nd Highest' :
                        fa = 0
                        fb = 0
                        if h.f_norm == 'absoluto':   #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 - delta) / f_1x
                                fb = (h.f1 + delta) / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 / f_1x
                                fb = h.f2 / f_1x
                            
                        if h.f_norm == 'relativo':  #-------------------------
                            if h.tipo == 'Peak':
                                fa = h.f1 - delta / f_1x
                                fb = h.f1 + delta / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 
                                fb = h.f2   
                            
                        if h.f_norm == 'mixto':     #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 + h.f2*f_1x - delta) / f_1x
                                fb = (h.f1 + h.f2*f_1x + delta) / f_1x
                        
                        #print (h.label,fa*f_1x,fb*f_1x)

                        if fa  <= f[k]/f_1x <= fb:
                            word = h.label
                            init = int(np.round(properties["left_ips" ][counter]))
                            end  = int(np.round(properties["right_ips"][counter]))
                            piko = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) 
                                                    #--estos valores son RMS mm/s
                            
                            #print (word,'   ','piko=',piko,'f=',f[k],'fa=',fa,'f[k]/f_1x =',f[k]/f_1x,'fb=',fb)
                            if piko > df_harm.iloc[medida]['RMS '+word]:
                                df_harm.iloc[medida]['i '     + word] = k
                                df_harm.iloc[medida]['Point ' + word] = sptrm_C[k]
                                df_harm.iloc[medida]['RMS '   + word] = piko #/ Max_value
                                df_harm.iloc[medida]['f '     + word] = f[k]
                                df_harm.iloc[medida]['BW '    + word] = f[int(properties["widths"][counter]) ]
                                df_harm.iloc[medida]['n_s '   + word] = init
                                df_harm.iloc[medida]['n_e '   + word] = end

    print('-------------------------------------Fingerprints extracted')
    print()
    return df_harm
#------------------------------------------------------------------------------
    #==============================================================================
    #-------ATENCION: df_FFT ya viene multiplicada por la ventana de hanning 
    #-------y corregido en potencia, es decir multiplicado por 1.63
def df_Harmonics(df_speed,df_FFT,fs):
    print('------------------------------------Extracting fingerprint')
    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]
    
    BPFI1 = 295.080
    BPFO1 = 222.920
    FTF1  = 10.615
    BSF1  = 85.864
    
    BPFI2 = 282.527
    BPFO2 = 210.806
    FTF2  = 10.540
    BSF2  = 82.008
    
    fignerprint_list = [
                        fingerprint('1/2' ,'Peak',1/2 ,0,'relativo'),
                        fingerprint('1.0' ,'Peak',1   ,0,'relativo'),
                        fingerprint('3/2' ,'Peak',1.5 ,0,'relativo'),
                        fingerprint('2.0' ,'Peak',2   ,0,'relativo'),
                        fingerprint('5/2' ,'Peak',2.5 ,0,'relativo'),
                        fingerprint('3.0' ,'Peak',3   ,0,'relativo'),
                        fingerprint('7/2' ,'Peak',3.5 ,0,'relativo'),
                        fingerprint('4.0' ,'Peak',4   ,0,'relativo'),
                        fingerprint('9/2' ,'Peak',4.5 ,0,'relativo'),
                        fingerprint('5.0' ,'Peak',5   ,0,'relativo'),
                        fingerprint('11/2','Peak',5.5 ,0,'relativo'),
                        fingerprint('6.0' ,'Peak',6   ,0,'relativo'),
                        fingerprint('13/2','Peak',6.5 ,0,'relativo'),
                        fingerprint('7.0' ,'Peak',7   ,0,'relativo'),
                        fingerprint('15/2','Peak',7.5 ,0,'relativo'),
                        fingerprint('8.0' ,'Peak',8   ,0,'relativo'),
                        fingerprint('17/2','Peak',8.5 ,0,'relativo'),
                        fingerprint('9.0' ,'Peak',9   ,0,'relativo'),
                        fingerprint('19/2','Peak',9.5 ,0,'relativo'),
                        fingerprint('10.0','Peak',10  ,0,'relativo'),
                        fingerprint('21/2','Peak',10.5,0,'relativo'),

                        fingerprint('2/3' ,'Peak',2/3 ,0,'relativo'),
                        fingerprint('4/3' ,'Peak',4/3 ,0,'relativo'),
                        fingerprint('5/3' ,'Peak',5/3 ,0,'relativo'),
                        fingerprint('8/3' ,'Peak',8/3 ,0,'relativo'),

                        fingerprint('1/3' ,'Peak',1/3 ,0,'relativo'),
                        fingerprint('1/4' ,'Peak',1/4 ,0,'relativo'),

                        fingerprint('11.0','Peak',11  ,0,'relativo'),
                        fingerprint('12.0','Peak',12  ,0,'relativo'),
                        fingerprint('13.0','Peak',13  ,0,'relativo'),
                        fingerprint('23.0','Peak',23  ,0,'relativo'),
                        fingerprint('24.0','Peak',24  ,0,'relativo'),
                        fingerprint('25.0','Peak',25  ,0,'relativo'),

                        fingerprint('2nd Highest'        ,'Peak',0    ,fs/2 ,'absoluto'),

                        fingerprint('Oil Whirl'          ,'Span',0.38 ,0.48 ,'relativo'),
                        fingerprint('500rpm'             ,'Span',0.38 ,0.48 ,'relativo'),
                        fingerprint('Flow T.'            ,'Span',12   ,24   ,'absoluto'),
                        fingerprint('Surge E. 0.33x 0.5x','Span',0.33 ,0.5  ,'relativo'),
                        fingerprint('Surge E. 12/20k'    ,'Span',12000,20000,'absoluto'),
                        
                        fingerprint('BPFO1'              ,'Peak',1*BPFO1     ,0 ,'absoluto'),
                        fingerprint('2*BPFO1'            ,'Peak',2*BPFO1     ,0 ,'absoluto'),
                        fingerprint('3*BPFO1'            ,'Peak',3*BPFO1     ,0 ,'absoluto'),
                        fingerprint('4*BPFO1'            ,'Peak',4*BPFO1     ,0 ,'absoluto'),
                        fingerprint('BPFO2'              ,'Peak',1*BPFO2     ,0 ,'absoluto'),
                        fingerprint('2*BPFO2'            ,'Peak',2*BPFO2     ,0 ,'absoluto'),
                        fingerprint('3*BPFO2'            ,'Peak',3*BPFO2     ,0 ,'absoluto'),
                        fingerprint('4*BPFO2'            ,'Peak',4*BPFO2     ,0 ,'absoluto'),

                        fingerprint('BPFI1'              ,'Peak',BPFI1       ,0 ,'absoluto'),
                        fingerprint('BPFI1+f'            ,'Peak',BPFI1       ,1 ,'mixto'),
                        fingerprint('BPFI1-f'            ,'Peak',BPFI1       ,-1,'mixto'),
                        fingerprint('2*BPFI1'            ,'Peak',2*BPFI1     ,0 ,'absoluto'),
                        fingerprint('2*BPFI1+f'          ,'Peak',2*BPFI1     ,1 ,'mixto'),
                        fingerprint('2*BPFI1-f'          ,'Peak',2*BPFI1     ,-1,'mixto'),
                        
                        fingerprint('BPFI2'              ,'Peak',BPFI2       ,0 ,'absoluto'),
                        fingerprint('BPFI2+f'            ,'Peak',BPFI2       ,1 ,'mixto'),
                        fingerprint('BPFI2-f'            ,'Peak',BPFI2       ,-1,'mixto'),
                        fingerprint('2*BPFI2'            ,'Peak',2*BPFI2     ,0 ,'absoluto'),
                        fingerprint('2*BPFI2+f'          ,'Peak',2*BPFI2     ,1 ,'mixto'),
                        fingerprint('2*BPFI2-f'          ,'Peak',2*BPFI2     ,-1,'mixto'),
                        
                        fingerprint('BSF1'               ,'Peak',BSF1        ,0 ,'absoluto'),
                        fingerprint('BSF1-FTF1'          ,'Peak',BSF1-FTF1   ,0 ,'absoluto'),
                        fingerprint('BSF1+FTF1'          ,'Peak',BSF1+FTF1   ,0 ,'absoluto'),
                        fingerprint('2*BSF1'             ,'Peak',2*BSF1      ,0 ,'absoluto'),
                        fingerprint('2*BSF1-FTF1'        ,'Peak',2*BSF1-FTF1 ,0 ,'absoluto'),
                        fingerprint('2*BSF1+FTF1'        ,'Peak',2*BSF1+FTF1 ,0 ,'absoluto'),
                        
                        fingerprint('BSF2'               ,'Peak',BSF2        ,0 ,'absoluto'),
                        fingerprint('BSF2-FTF2'          ,'Peak',BSF2-FTF2   ,0 ,'absoluto'),
                        fingerprint('BSF2+FTF2'          ,'Peak',BSF2+FTF2   ,0 ,'absoluto'),
                        fingerprint('2*BSF2'             ,'Peak',2*BSF2      ,0 ,'absoluto'),
                        fingerprint('2*BSF2-FTF2'        ,'Peak',2*BSF2-FTF2 ,0 ,'absoluto'),
                        fingerprint('2*BSF2+FTF2'        ,'Peak',2*BSF2+FTF2 ,0 ,'absoluto'),
                        
                        fingerprint('FTF1'               ,'Peak',FTF1        ,0 ,'absoluto'),
                        fingerprint('2*FTF1'             ,'Peak',2*FTF1      ,0 ,'absoluto'),
                        fingerprint('3*FTF1'             ,'Peak',3*FTF1      ,0 ,'absoluto'),
                        fingerprint('4*FTF1'             ,'Peak',4*FTF1      ,0 ,'absoluto'),
                        fingerprint('FTF2'               ,'Peak',FTF2        ,0 ,'absoluto'),
                        fingerprint('2*FTF2'             ,'Peak',2*FTF2      ,0 ,'absoluto'),
                        fingerprint('3*FTF2'             ,'Peak',3*FTF2      ,0 ,'absoluto'),
                        fingerprint('4*FTF2'             ,'Peak',4*FTF2      ,0 ,'absoluto'),
                        
                        fingerprint('R. Pump'            ,'Span',0   ,10   ,'absoluto'),
                        
                        ]

    fecha = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))

    columnas = []
    word =''
    for k in fignerprint_list:
        word = k.label
        columnas.append('i '+word)
        columnas.append('Point '+word)
        columnas.append('RMS '+word)
        columnas.append('f '+word)
        columnas.append('BW '+word)
        columnas.append('n_s '+word)
        columnas.append('n_e '+word)
    
    columnas    = ['RMS (mm/s) t','RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))

    f_1x_low    = 20
    f_1x_high   = 30
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    
    i_f_1x_low  = int(np.round(f_1x_low*l/fs))
    i_f_1x_high = int(np.round(f_1x_high*l/fs))


    for medida in range(n_traces):
        #print ('Medida numero',df_harm.index[medida])
        
        RMS_freq                             = np.sqrt(np.sum( df_FFT.iloc[medida].values**2 ) )
        RMS_time                             = np.std (df_speed.iloc[medida].values)
        df_harm.iloc[medida]['RMS (mm/s) t'] = RMS_time
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        
        sptrm_C                              = df_FFT.iloc[medida].values * np.sqrt(2) 
                                            # integramos en 1º z Nyquist por eso el voltaje x raiz(2)         
        
        indexes, properties                  = find_peaks(sptrm_C[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
        array_peaks                          = sptrm_C[indexes]

        indexes_f_1x, properties_f_1x        = find_peaks(sptrm_C[i_f_1x_low:i_f_1x_high],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
        
        f_1x_exist                           = False
        max_value_f_1x                       = 0.1
                # miramos si hay algun pico entre 48 y 51Hz y nos quedamos con el mas grande
                # y si lo encontramos entonces "f_1x_exist= True"
        for k in indexes_f_1x:    
            if 24 < f[i_f_1x_low + k] < 25.5 and sptrm_C[i_f_1x_low + k] > max_value_f_1x:
                f_1x           = f[i_f_1x_low + k]
                max_value_f_1x = sptrm_C[i_f_1x_low + k]
                f_1x_exist     = True
                
        if f_1x_exist :
            #----------------------------------------------------------------------------------------- El segundopico mas grande
            i_L              = np.argmax(array_peaks)
            array_peaks[i_L] = 0
            i_L2             = np.argmax(array_peaks)

            df_harm.iloc[medida]['i 2nd Highest']     = indexes[i_L2]
            df_harm.iloc[medida]['Point 2nd Highest'] = sptrm_C[indexes[i_L2]]
            df_harm.iloc[medida]['RMS 2nd Highest']   = np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 ))
            df_harm.iloc[medida]['f 2nd Highest']     = f[indexes[i_L2]]
            df_harm.iloc[medida]['BW 2nd Highest']    = properties["widths"][i_L2]

            delta = 0.75 #-----------en Hz
            fa = 0
            fb = 0
            for counter,k in enumerate(indexes) :
                for h in fignerprint_list:
                    if h.tipo !='2nd Highest' :
                        fa = 0
                        fb = 0
                        if h.f_norm == 'absoluto':   #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 - delta) / f_1x
                                fb = (h.f1 + delta) / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 / f_1x
                                fb = h.f2 / f_1x
                            
                        if h.f_norm == 'relativo':  #-------------------------
                            if h.tipo == 'Peak':
                                fa = h.f1 - delta / f_1x
                                fb = h.f1 + delta / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 
                                fb = h.f2   
                            
                        if h.f_norm == 'mixto':     #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 + h.f2*f_1x - delta) / f_1x
                                fb = (h.f1 + h.f2*f_1x + delta) / f_1x

                        if fa  <= f[k]/f_1x <= fb:
                            word = h.label
                            init = int(np.round(properties["left_ips" ][counter]))
                            end  = int(np.round(properties["right_ips"][counter]))
                            piko = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) 
                                                    #--estos valores son RMS mm/s

                            if piko > df_harm.iloc[medida]['RMS '+word]:
                                df_harm.iloc[medida]['i '     + word] = k
                                df_harm.iloc[medida]['Point ' + word] = sptrm_C[k]
                                df_harm.iloc[medida]['RMS '   + word] = piko #/ Max_value
                                df_harm.iloc[medida]['f '     + word] = f[k]
                                df_harm.iloc[medida]['BW '    + word] = f[int(properties["widths"][counter]) ]
                                df_harm.iloc[medida]['n_s '   + word] = init
                                df_harm.iloc[medida]['n_e '   + word] = end

    print('-------------------------------------Fingerprints extracted')
    print()
    return df_harm
#------------------------------------------------------------------------------
#==============================================================================
    #-------ATENCION: df_FFT ya viene multiplicada por la ventana de hanning 
    #-------y corregido en potencia, es decir multiplicado por 1.63
def df_Harmonics_pumps(df_speed,df_FFT,fs):
    print('------------------------------------Extracting fingerprint')
    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]
    

    
    fignerprint_list = [
                        fingerprint('1.0'         ,'Peak',1   ,0    ,'relativo'),
                        fingerprint('2.0'         ,'Peak',2   ,0    ,'relativo'),
                        fingerprint('3.0'         ,'Peak',3   ,0    ,'relativo'),
                        fingerprint('4.0'         ,'Peak',4   ,0    ,'relativo'),
                        fingerprint('5.0'         ,'Peak',5   ,0    ,'relativo'),
                        fingerprint('2nd Highest' ,'Peak',0   ,fs/2 ,'absoluto'),
                        fingerprint('R. Pump'     ,'Span',0   ,10   ,'absoluto'),
                        fingerprint('Hydr. Inst.' ,'Span',0.7 ,0.85 ,'relativo'),
                        ]

    fecha = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))

    columnas = []
    word =''
    for k in fignerprint_list:
        word = k.label
        columnas.append('i '+word)
        columnas.append('Point '+word)
        columnas.append('RMS '+word)
        columnas.append('f '+word)
        columnas.append('BW '+word)
        columnas.append('n_s '+word)
        columnas.append('n_e '+word)
    
    columnas    = ['RMS (mm/s) t','RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))

    f_1x_low    = 39
    f_1x_high   = 59
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    
    i_f_1x_low  = int(np.round(f_1x_low*l/fs))
    i_f_1x_high = int(np.round(f_1x_high*l/fs))
    
    #print (i_f_1x_low,f[i_f_1x_low])

    for medida in range(n_traces):
        print ('>>>>>>>>>>>>>>>>>>>>>>>>>>>>Medida numero',df_harm.index[medida])
        
        RMS_freq                             = np.sqrt(np.sum( df_FFT.iloc[medida].values**2 ) )
        RMS_time                             = np.std (df_speed.iloc[medida].values)
        df_harm.iloc[medida]['RMS (mm/s) t'] = RMS_time
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        
        sptrm_C                              = df_FFT.iloc[medida].values * np.sqrt(2) 
                                            # integramos en 1º z Nyquist por eso el voltaje x raiz(2)         
        #np.save('pump',sptrm_C)

        indexes, properties                 = find_peaks(sptrm_C[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
        array_peaks                         = sptrm_C[indexes]

        indexes_f_1x, properties_f_1x       = find_peaks(sptrm_C[i_f_1x_low:i_f_1x_high],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
                                            # ------miro todos los picos entre f_1x_low y f_1x_high
#        print(indexes_f_1x)
#        print('frecuencias de los indices',f[i_f_1x_low+indexes_f_1x])
#        print('modulo de los indices',sptrm_C[i_f_1x_low+indexes_f_1x])
        f_1x_exist                          = False
        max_value_f_1x                      = 0.03
                # miramos si hay algun pico entre 48 y 51Hz y nos quedamos con el mas grande
                # y si lo encontramos entonces "f_1x_exist= True"
        for counter,k in enumerate(indexes_f_1x):    
            
            if 48.5 < f[i_f_1x_low + k] < 49 and sptrm_C[i_f_1x_low + k] > max_value_f_1x:
                f_1x           = f[i_f_1x_low + k]
                max_value_f_1x = sptrm_C[i_f_1x_low + k]
                f_1x_exist     = True
#                print (medida,counter,f_1x,max_value_f_1x)
                

        if f_1x_exist :
            
            #----------------------------------------------------------------------------------------- El segundopico mas grande
            i_L              = np.argmax(array_peaks)
            array_peaks[i_L] = 0
            i_L2             = np.argmax(array_peaks)

            df_harm.iloc[medida]['i 2nd Highest']     = indexes[i_L2]
            df_harm.iloc[medida]['Point 2nd Highest'] = sptrm_C[indexes[i_L2]]
            df_harm.iloc[medida]['RMS 2nd Highest']   = np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 ))
            df_harm.iloc[medida]['f 2nd Highest']     = f[indexes[i_L2]]
            df_harm.iloc[medida]['BW 2nd Highest']    = properties["widths"][i_L2]
            #print(np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 )),sptrm_C[indexes[i_L2]])

                                             #----------------------------------
            delta = 0.75 #-----------en Hz
            fa = 0
            fb = 0
            for counter,k in enumerate(indexes) :
                for h in fignerprint_list:
                    if h.tipo !='2nd Highest' :
                        fa = 0
                        fb = 0
                        if h.f_norm == 'absoluto':   #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 - delta) / f_1x
                                fb = (h.f1 + delta) / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 / f_1x
                                fb = h.f2 / f_1x
                            
                        if h.f_norm == 'relativo':  #-------------------------
                            if h.tipo == 'Peak':
                                fa = h.f1 - delta / f_1x
                                fb = h.f1 + delta / f_1x
                            if h.tipo == 'Span':
                                fa = h.f1 
                                fb = h.f2   
                            
                        if h.f_norm == 'mixto':     #-------------------------
                            if h.tipo == 'Peak':
                                fa = (h.f1 + h.f2*f_1x - delta) / f_1x
                                fb = (h.f1 + h.f2*f_1x + delta) / f_1x
                        
                        #print (h.label,fa*f_1x,fb*f_1x)

                        if fa  <= f[k]/f_1x <= fb:
                            word = h.label
                            init = int(np.round(properties["left_ips" ][counter]))
                            end  = int(np.round(properties["right_ips"][counter]))
                            piko = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) 
                                                    #--estos valores son RMS mm/s
                            
                            #print (word,'   ','piko=',piko,'f=',f[k],'fa=',fa,'f[k]/f_1x =',f[k]/f_1x,'fb=',fb)
                            if piko > df_harm.iloc[medida]['RMS '+word]:
                                df_harm.iloc[medida]['i '     + word] = k
                                df_harm.iloc[medida]['Point ' + word] = sptrm_C[k]
                                df_harm.iloc[medida]['RMS '   + word] = piko #/ Max_value
                                df_harm.iloc[medida]['f '     + word] = f[k]
                                df_harm.iloc[medida]['BW '    + word] = f[int(properties["widths"][counter]) ]
                                df_harm.iloc[medida]['n_s '   + word] = init
                                df_harm.iloc[medida]['n_e '   + word] = end

    print('-------------------------------------Fingerprints extracted')
    print()
    return df_harm
#------------------------------------------------------------------------------

def plot_waterfall(df_in,harm,fs,fmin,fmax):
    alfa      = 0.7
    cc        = lambda arg: colorConverter.to_rgba(arg, alpha=alfa)
    color1    = cc('b')
    color2    = cc('y')

    col_list  = [color1,color2]
    n_traces  = df_in.shape[0]
    l         = df_in.shape[1]
    #f         = np.arange(l)/l*fs

    escala ='RMS' #escalado para PLOTEAR
    if escala == 'Peak':
        cte   = 2 * 2                       # 2 => hanning , 2 => 1º Z. Niquist 
        label = 'Peak'
    else:
        cte   = 2* np.sqrt(2)               # 2 => hanning , np.sqrt(2) => 1º Z. Niquist
        label = 'RMS'

    #fmax      = 55
    n_fmin    = np.int(l*fmin/(fs))
    n_fmax    = np.int(l*fmax/(fs))
    f_portion = np.arange(n_fmin,n_fmax)/n_fmax*fmax
    verts     = []
    color     = np.ones((n_fmax-n_fmin,n_traces))
    traces    = df_in.index.values
    traces    = (traces-traces[0])/24/3600

    #fig       = plt.figure()
    fig=plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax        = fig.gca(projection='3d')

    inic_day  = datetime.datetime.fromtimestamp(df_in.index[0]).day
    facecol   = []
    color_counter = 0
    for counter,indice in enumerate(df_in.index):

        if datetime.datetime.fromtimestamp(df_in.index[counter]).day != inic_day:
            color_counter = color_counter+1

        curva            = cte *  df_in.iloc[counter].values
        curva_p          = curva[n_fmin:n_fmax]
        minimo           = np.min(curva_p)
        #print (minimo,np.argmin(curva_p))
        curva_p[0]       = minimo
        curva_p[-1]      = minimo
        color[:,counter] = curva_p
        inic_day         = datetime.datetime.fromtimestamp(df_in.index[counter]).day
        verts.append(list(zip(f_portion, curva[n_fmin:n_fmax])))
        facecol.append(col_list[np.mod(color_counter,2)])

    #poly = PolyCollection(verts, facecolors=[cc('g')])
    poly = PolyCollection(verts, facecolors=facecol)
    ax.add_collection3d(poly, zs=traces, zdir='y')

    ax.view_init(40, -90)
    ax.set_xlabel('Hertz')
    ax.set_xlim3d(fmin, fmax)
    ax.set_ylabel('3edays')
    ax.set_ylim3d(0,    traces[np.size(traces)-1])
    ax.set_zlabel(label+'mm/s')
    ax.set_zlim3d(np.min(color), np.max(color))
    ax.set_title('mm/sg '+ label)

    second_plot = plt.axes([.05, .75, .2, .2], facecolor='w')
    #fig, ax1 = plt.subplots()

    plt.plot(traces,harm.loc[:,'RMS (mm/s) f'].values)
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('RMS')
    plt.tight_layout()
    #plt.show()

    return color , verts
#------------------------------------------------------------------------------

def plot_waterfall2(Parameters,df_in,df_harm,fs,fmin,fmax):

    #col_list  = ['b','r']c='xkcd:baby poop green'
    col_list  = ['xkcd:light mustard','xkcd:maize',
                 'xkcd:golden','xkcd:sand',
                 'xkcd:wheat',
                 'xkcd:yellow ochre','xkcd:mud']
    l         = df_in.shape[1]
    #df_in.sort_index(ascending=False,inplace=True)

    n_fmin        = np.int(l*fmin/(fs))
    n_fmax        = np.int(l*fmax/(fs))
    f_portion     = np.arange(n_fmin,n_fmax)/n_fmax*fmax

    y_portion     = np.ones(np.size(f_portion)) #---array para el eje Y

    t_traces      = df_in.index.values
    t_traces      = (t_traces-t_traces[0])/24/3600

    #fig       = plt.figure()
    fig           = plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax            = fig.gca(projection='3d')
    inic_day      = datetime.datetime.fromtimestamp(df_in.index[0]).day
    color_counter = 0
    for counter,indice in enumerate(df_in.index):

        if datetime.datetime.fromtimestamp(df_in.index[counter]).day != inic_day:
            color_counter = color_counter+1

        curva            = 2 * df_in.iloc[counter].values/1.63 *np.sqrt(2)
        curva_p          = (curva[n_fmin:n_fmax])
        inic_day         = datetime.datetime.fromtimestamp(df_in.index[counter]).day
        #print(t_traces[counter])
        #ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[np.mod(color_counter,7)], linewidth = 0.2)
        ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[datetime.datetime.fromtimestamp(df_in.index[counter]).weekday()], linewidth = 0.2)

    ax.view_init(40, 90)
    ax.set_xlabel('Hertz')
    ax.set_xlim3d(fmin, fmax)
    ax.set_ylabel('days')
    ax.set_ylim3d(0, t_traces[np.size(t_traces)-1])
    ax.set_zlabel('RMS mm/s')
    #ax.set_zlim3d(np.min(color), np.max(color))
    ax.set_title(Parameters['IdAsset']+' '+Parameters['Localizacion']+' mm/sg RMS')

    second_plot = plt.axes([.05, .75, .2, .2], facecolor='w')
    plt.plot(t_traces,df_harm.loc[:,'RMS (mm/s) f'].values)
    plt.legend(('Total RMS', 'Peak @1x'),loc='upper right')
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('RMS')
    plt.tight_layout()

    return

#------------------------------------------------------------------------------
def find_closest(date,df_VELOCITY,df_harm):
    segundos = time.mktime(date.timetuple())
    captura = np.argmin( np.abs(df_VELOCITY.index.values-segundos) )
    print('Hora de la captura', datetime.datetime.fromtimestamp(df_VELOCITY.index[captura]))
    print('Diferencia en minutos',(df_VELOCITY.index[captura]-segundos)/60)
    print(captura)
    waveform = df_VELOCITY.iloc[captura].values

    date_exact = datetime.datetime.fromtimestamp(df_VELOCITY.index[captura])

    l          = np.size(waveform)
    l_mitad    = int(l/2)
    f          = np.arange(l)/l*fs
    #----------quitamos los 2 HZ del principio

    label = 'Peak'
    SPTRM_P             = 2 * df_VELOCITY.iloc[captura].values * 2

    indexes, properties = find_peaks( SPTRM_P[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)

    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')

    #plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    #ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)

    fig, ax1 = plt.subplots(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(f[0:l_mitad] , SPTRM_P[0:l_mitad],'b')
    #plt.plot(f[indexes]   , SPTRM_P[indexes]  ,'o')

    for counter,k in enumerate(df_harm.columns):

        if k[0] == 'i':
            #print (counter,k)
            index = int(df_harm.iloc[captura][counter])
            if f[index] !=0:
                offset = 0
                plt.plot(f[index] , SPTRM_P[index]  ,'o')
                if k == 'i 2nd Highest':
                    offset = 0.1
                #plt.text(f[index] , SPTRM_P[index],str(k))
                plt.text(f[index] , SPTRM_P[index]+offset,str(df_harm.columns[counter+1])+' '+str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+1])],'.02f'  ) ) )
                #print( str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+1])],'.02f'  ) )    )

    plt.ylabel('mm/s '+label)
    plt.title(str(date_exact))
    plt.xlabel('Hz')
    plt.grid(True)

    plt.vlines(x=f[indexes], ymin=SPTRM_P[indexes] - properties["prominences"],ymax = SPTRM_P[indexes], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=f[np.round(properties["left_ips"]).astype(int)],xmax=f[np.round(properties["right_ips"]).astype(int)], color = "C1")

    ax1.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    return

#------------------------------------------------------------------------------
#def Plot_Spectrum(date,df_VELOCITY,df_harm):
#    segundos = time.mktime(date.timetuple())
#    captura = np.argmin( np.abs(df_VELOCITY.index.values-segundos) )
#    print('Hora de la captura', datetime.datetime.fromtimestamp(df_VELOCITY.index[captura]))
#    print('Diferencia en minutos',(df_VELOCITY.index[captura]-segundos)/60)
#    print(captura)
    
def Plot_Spectrum(captura,df_VELOCITY,df_harm):
    
    waveform = df_VELOCITY.iloc[captura].values
    
    date_exact = datetime.datetime.fromtimestamp(df_VELOCITY.index[captura])

    l          = np.size(waveform)
    l_mitad    = int(l/2)
    f          = np.arange(l)/(l-1)*fs
    #----------quitamos los 2 HZ del principio

    
    SPTRM_P             = 2 * df_VELOCITY.iloc[captura].values * np.sqrt(2) /1.63

    indexes, properties = find_peaks( SPTRM_P[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)

    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')

    #plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    #ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)

    fig, ax1 = plt.subplots(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(f[0:l_mitad] , SPTRM_P[0:l_mitad],'b')
    #plt.plot(f[indexes]   , SPTRM_P[indexes]  ,'o')

    for counter,k in enumerate(df_harm.columns):

        if k[0] == 'i':
            #print (counter,k)
            index = int(df_harm.iloc[captura][counter])
            if f[index] !=0:
                offset = 0
                #plt.plot(f[index] , df_harm.iloc[captura][str(df_harm.columns[counter+2])]  ,'o')
                #plt.vlines(x=f[index], ymin=SPTRM_P[index] - properties["prominences"],ymax = df_harm.iloc[captura][str(df_harm.columns[counter+2])], color = "C1")
                plt.plot(f[index] , SPTRM_P[index]  ,'o')
                if k == 'i 2nd Highest':
                    offset = 0.1
                #plt.text(f[index] , SPTRM_P[index],str(k))
                #plt.text(f[index] ,df_harm.iloc[captura][str(df_harm.columns[counter+2])]+offset,str(df_harm.columns[counter+2])+' '+str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+2])],'.02f'  ) ) )
                plt.text(f[index] ,SPTRM_P[index]+offset,str(df_harm.columns[counter+2])+' '+str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+2])],'.02f'  ) ) )
                #print( str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+1])],'.02f'  ) )    )

    plt.ylabel('mm/s RMS')
    plt.title('Date: '+str(date_exact)+'           RMS: '+str(df_harm.iloc[captura]['RMS (mm/s) f'])+'mm/s' )
    plt.xlabel('Hz')
    plt.grid(True)

    plt.vlines(x=f[indexes], ymin=SPTRM_P[indexes] - properties["prominences"],ymax = SPTRM_P[indexes], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=f[np.round(properties["left_ips"]).astype(int)],xmax=f[np.round(properties["right_ips"]).astype(int)], color = "C1")

    ax1.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    
    return
#------------------------------------------------------------------------------
def check_test_results(harm):
    for columna in harm.columns:
        if columna[0]=='$':
            print(columna,'==>',harm.index[harm[str(columna)] == 'None'])
            print          
#------------------------------------------------------------------------------
def plot_waterfall27(Parameters,df_in,df_harm,fs,fmin,fmax):

    #col_list  = ['b','r']c='xkcd:baby poop green'
    col_list  = ['xkcd:light mustard','xkcd:maize',
                 'xkcd:golden','xkcd:sand',
                 'xkcd:wheat',
                 'xkcd:yellow ochre','xkcd:mud']
    l         = df_in.shape[1]
    #df_in.sort_index(ascending=False,inplace=True)

    n_fmin        = np.int(l*fmin/(fs))
    n_fmax        = np.int(l*fmax/(fs))
    f_portion     = np.arange(n_fmin,n_fmax)/n_fmax*fmax

    y_portion     = np.ones(np.size(f_portion)) #---array para el eje Y

    t_traces      = df_in.index.values
    t_traces      = (t_traces-t_traces[0])/24/3600

    #fig       = plt.figure()
    fig           = plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax            = fig.gca(projection='3d')
    inic_day      = datetime.datetime.fromtimestamp(df_in.index[0]).day
    color_counter = 0
    for counter,indice in enumerate(df_in.index):

        if datetime.datetime.fromtimestamp(df_in.index[counter]).day != inic_day:
            color_counter = color_counter+1

        curva            = 2 * df_in.iloc[counter].values/1.63 *np.sqrt(2)
        curva_p          = (curva[n_fmin:n_fmax])
        inic_day         = datetime.datetime.fromtimestamp(df_in.index[counter]).day
        #print(t_traces[counter])
        ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[np.mod(color_counter,7)], linewidth = 0.2)
    ax.set_yticks(df_in.index)
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.view_init(40, 90)
    ax.set_xlabel('Hertz')
    ax.set_xlim3d(fmin, fmax)
    ax.set_ylabel('days')
    #ax.set_ylim3d(0, t_traces[np.size(t_traces)-1])
    ax.set_zlabel('RMS mm/s')
    #ax.set_zlim3d(np.min(color), np.max(color))
    ax.set_title(Parameters['IdAsset']+' '+Parameters['Localizacion']+' mm/sg RMS')

    second_plot = plt.axes([.05, .75, .2, .2], facecolor='w')
    plt.plot(t_traces,df_harm.loc[:,'RMS (mm/s) f'].values)
    plt.legend(('Total RMS', 'Peak @1x'),loc='upper right')
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('RMS')
    plt.tight_layout()

    return