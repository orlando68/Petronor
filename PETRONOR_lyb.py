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

from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy import signal
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter

import pandas as pd
from numba import jit

from itertools import combinations


pi        = np.pi
E1        = 0.10
E2        = 0.07
fs        = 5120.0
l         = 16384
Path_out  ='C://OPG106300//TRABAJO//Proyectos//Petronor-075879.1 T 20000//Trabajo//data//outputs//'
Path_out  ='//home//instalador//Mantenimiento//data//outputs//' 
#------------------------------------------------------------------------------


class fingerprint:
    def __init__(self,        label,tipo,f1,f2,f_norm,order):
        self.label  = label
        self.tipo   = tipo
        self.f1     = f1
        self.f2     = f2
        self.f_norm = f_norm
        self.order  = order
       

    def __str__(self):
        s = ''.join(['Label    : ', str(self.label), '\n',
                     'Tipo     : ', str(self.tipo),  '\n',
                     'f1       : ', str(self.f1),    '\n',
                     'f2       : ', str(self.f2),    '\n',
                     'Relative : ', str(self.f_norm),'\n',
                     '\n'])
        return s

#------------------------------------------------------------------------------

        
BPFI1 = 295.080
BPFO1 = 222.920
FTF1  = 10.615
BSF1  = 85.864

BPFI2 = 282.527
BPFO2 = 210.806
FTF2  = 10.540
BSF2  = 82.008

fprnt_list_blwrs = [
                    fingerprint('Max Value.'  ,'Span',0 ,fs/2 ,'absoluto',2),
                    fingerprint('1.0' ,'Peak',1   ,0,'relativo',1),
                    fingerprint('2.0' ,'Peak',2   ,0,'relativo',1),
                    fingerprint('3.0' ,'Peak',3   ,0,'relativo',1),
                    fingerprint('4.0' ,'Peak',4   ,0,'relativo',1),
                    fingerprint('5.0' ,'Peak',5   ,0,'relativo',1),
                    fingerprint('6.0' ,'Peak',6   ,0,'relativo',1),
                    fingerprint('7.0' ,'Peak',7   ,0,'relativo',1),
                    fingerprint('8.0' ,'Peak',8   ,0,'relativo',1),
                    fingerprint('9.0' ,'Peak',9   ,0,'relativo',1),
                    fingerprint('10.0','Peak',10  ,0,'relativo',1),
                    fingerprint('11.0','Peak',11  ,0,'relativo',1),
                    fingerprint('12.0','Peak',12  ,0,'relativo',1),
                    fingerprint('13.0','Peak',13  ,0,'relativo',1),
                    
                    fingerprint('1/2' ,'Peak',1/2 ,0,'relativo',1),
                    fingerprint('3/2' ,'Peak',1.5 ,0,'relativo',1),
                    fingerprint('5/2' ,'Peak',2.5 ,0,'relativo',1),
                    fingerprint('7/2' ,'Peak',3.5 ,0,'relativo',1),
                    fingerprint('9/2' ,'Peak',4.5 ,0,'relativo',1),
                    fingerprint('11/2','Peak',5.5 ,0,'relativo',1),
                    fingerprint('13/2','Peak',6.5 ,0,'relativo',1),
                    fingerprint('15/2','Peak',7.5 ,0,'relativo',1),
                    fingerprint('17/2','Peak',8.5 ,0,'relativo',1),
                    fingerprint('19/2','Peak',9.5 ,0,'relativo',1),
                    fingerprint('21/2','Peak',10.5,0,'relativo',1),

                    fingerprint('2/3' ,'Peak',2/3 ,0,'relativo',1),
                    fingerprint('4/3' ,'Peak',4/3 ,0,'relativo',1),
                    fingerprint('5/3' ,'Peak',5/3 ,0,'relativo',1),
                    fingerprint('8/3' ,'Peak',8/3 ,0,'relativo',1),

                    fingerprint('1/3' ,'Peak',1/3 ,0,'relativo',1),
                    fingerprint('1/4' ,'Peak',1/4 ,0,'relativo',1),
               
                    fingerprint('23.0','Peak',23  ,0,'relativo',1),
                    fingerprint('24.0','Peak',24  ,0,'relativo',1),
                    fingerprint('25.0','Peak',25  ,0,'relativo',1),

                    fingerprint('Oil Whirl'          ,'Span',0.38 ,0.48 ,'relativo',1),
                    fingerprint('500rpm'             ,'Span',0.38 ,0.48 ,'relativo',1),
                    fingerprint('Flow T.'            ,'Span',12   ,24   ,'absoluto',1),
                    fingerprint('Surge E. 0.33x 0.5x','Span',0.33 ,0.5  ,'relativo',1),
                    fingerprint('Surge E. 12/20k'    ,'Span',12000,20000,'absoluto',1),
                    
                    fingerprint('BPFO1'              ,'Peak',1*BPFO1     ,0 ,'absoluto',1),
                    fingerprint('2*BPFO1'            ,'Peak',2*BPFO1     ,0 ,'absoluto',1),
                    fingerprint('3*BPFO1'            ,'Peak',3*BPFO1     ,0 ,'absoluto',1),
                    fingerprint('4*BPFO1'            ,'Peak',4*BPFO1     ,0 ,'absoluto',1),
                    fingerprint('BPFO2'              ,'Peak',1*BPFO2     ,0 ,'absoluto',1),
                    fingerprint('2*BPFO2'            ,'Peak',2*BPFO2     ,0 ,'absoluto',1),
                    fingerprint('3*BPFO2'            ,'Peak',3*BPFO2     ,0 ,'absoluto',1),
                    fingerprint('4*BPFO2'            ,'Peak',4*BPFO2     ,0 ,'absoluto',1),

                    fingerprint('BPFI1'              ,'Peak',BPFI1       ,0 ,'absoluto',1),
                    fingerprint('BPFI1+f'            ,'Peak',BPFI1       ,1 ,'mixto',1),
                    fingerprint('BPFI1-f'            ,'Peak',BPFI1       ,-1,'mixto',1),
                    fingerprint('2*BPFI1'            ,'Peak',2*BPFI1     ,0 ,'absoluto',1),
                    fingerprint('2*BPFI1+f'          ,'Peak',2*BPFI1     ,1 ,'mixto',1),
                    fingerprint('2*BPFI1-f'          ,'Peak',2*BPFI1     ,-1,'mixto',1),
                    
                    fingerprint('BPFI2'              ,'Peak',BPFI2       ,0 ,'absoluto',1),
                    fingerprint('BPFI2+f'            ,'Peak',BPFI2       ,1 ,'mixto',1),
                    fingerprint('BPFI2-f'            ,'Peak',BPFI2       ,-1,'mixto',1),
                    fingerprint('2*BPFI2'            ,'Peak',2*BPFI2     ,0 ,'absoluto',1),
                    fingerprint('2*BPFI2+f'          ,'Peak',2*BPFI2     ,1 ,'mixto',1),
                    fingerprint('2*BPFI2-f'          ,'Peak',2*BPFI2     ,-1,'mixto',1),
                    
                    fingerprint('BSF1'               ,'Peak',BSF1        ,0 ,'absoluto',1),
                    fingerprint('BSF1-FTF1'          ,'Peak',BSF1-FTF1   ,0 ,'absoluto',1),
                    fingerprint('BSF1+FTF1'          ,'Peak',BSF1+FTF1   ,0 ,'absoluto',1),
                    fingerprint('2*BSF1'             ,'Peak',2*BSF1      ,0 ,'absoluto',1),
                    fingerprint('2*BSF1-FTF1'        ,'Peak',2*BSF1-FTF1 ,0 ,'absoluto',1),
                    fingerprint('2*BSF1+FTF1'        ,'Peak',2*BSF1+FTF1 ,0 ,'absoluto',1),
                    
                    fingerprint('BSF2'               ,'Peak',BSF2        ,0 ,'absoluto',1),
                    fingerprint('BSF2-FTF2'          ,'Peak',BSF2-FTF2   ,0 ,'absoluto',1),
                    fingerprint('BSF2+FTF2'          ,'Peak',BSF2+FTF2   ,0 ,'absoluto',1),
                    fingerprint('2*BSF2'             ,'Peak',2*BSF2      ,0 ,'absoluto',1),
                    fingerprint('2*BSF2-FTF2'        ,'Peak',2*BSF2-FTF2 ,0 ,'absoluto',1),
                    fingerprint('2*BSF2+FTF2'        ,'Peak',2*BSF2+FTF2 ,0 ,'absoluto',1),
                    
                    fingerprint('FTF1'               ,'Peak',FTF1        ,0 ,'absoluto',1),
                    fingerprint('2*FTF1'             ,'Peak',2*FTF1      ,0 ,'absoluto',1),
                    fingerprint('3*FTF1'             ,'Peak',3*FTF1      ,0 ,'absoluto',1),
                    fingerprint('4*FTF1'             ,'Peak',4*FTF1      ,0 ,'absoluto',1),
                    fingerprint('FTF2'               ,'Peak',FTF2        ,0 ,'absoluto',1),
                    fingerprint('2*FTF2'             ,'Peak',2*FTF2      ,0 ,'absoluto',1),
                    fingerprint('3*FTF2'             ,'Peak',3*FTF2      ,0 ,'absoluto',1),
                    fingerprint('4*FTF2'             ,'Peak',4*FTF2      ,0 ,'absoluto',1),
                    
                    fingerprint('R. Pump'            ,'Span',0   ,10   ,'absoluto',1),
                    ]    
    
BPFI = 321.489 
BPFO = 219.345 
FTF  = 19.940 
BSF  = 96.155 
VPF  = 245.85
fprnt_list_pumps = [
                    fingerprint('Max Value.'  ,'Span',0 ,fs/2 ,'absoluto',2),            
                    fingerprint('1.0'         ,'Peak',1   ,0    ,'relativo',1),
                    fingerprint('2.0'         ,'Peak',2   ,0    ,'relativo',1),
                    
                    fingerprint('Pip Vib.'    ,'Span',3,15,'absoluto',3),
                    fingerprint('Auto Osc.'   ,'Span',15 ,20 ,'absoluto',3),
                    
                    
                    fingerprint('3.0'         ,'Peak',3   ,0    ,'relativo',1),
                    fingerprint('4.0'         ,'Peak',4   ,0    ,'relativo',1),
                    fingerprint('5.0'         ,'Peak',5   ,0    ,'relativo',1),
                    fingerprint('10.0'        ,'Peak',10  ,0    ,'relativo',1),
                    
                    fingerprint('1/2'         ,'Peak',1/2 ,0,'relativo',1),
                    fingerprint('3/2'         ,'Peak',1.5 ,0,'relativo',1),
                    fingerprint('5/2'         ,'Peak',2.5 ,0,'relativo',1),
                    
                    fingerprint('Oil Whirl'   ,'Span',0.38 ,0.48 ,'relativo',1),
                    
                    fingerprint('R. Pump'     ,'Span',0   ,10   ,'absoluto',1),
                    fingerprint('Hydr. Inst.' ,'Span',0.7 ,0.85 ,'relativo',1),
                    fingerprint('Cavit. Noise','Span',1000,2500,'absoluto',1),
                    fingerprint('Rotat. Cavit','Span',1.1 ,1.25 ,'relativo',1),
                    fingerprint('R. R. Stall' ,'Span',0.5 ,0.75 ,'relativo',1),
                    
                     
                    
                    fingerprint('BPFO'        ,'Peak',1*BPFO     ,0 ,'absoluto',1),
                    fingerprint('2*BPFO'      ,'Peak',2*BPFO     ,0 ,'absoluto',1),
                    fingerprint('3*BPFO'      ,'Peak',3*BPFO     ,0 ,'absoluto',1),
                    fingerprint('4*BPFO'      ,'Peak',4*BPFO     ,0 ,'absoluto',1),           

                    fingerprint('BPFI'        ,'Peak',BPFI       ,0 ,'absoluto',1),
                    fingerprint('BPFI+f'      ,'Peak',BPFI       ,1 ,'mixto',1),
                    fingerprint('BPFI-f'      ,'Peak',BPFI       ,-1,'mixto',1),
                    fingerprint('2*BPFI'      ,'Peak',2*BPFI     ,0 ,'absoluto',1),
                    fingerprint('2*BPFI+f'    ,'Peak',2*BPFI     ,1 ,'mixto',1),
                    fingerprint('2*BPFI-f'    ,'Peak',2*BPFI     ,-1,'mixto',1),
                    
                    fingerprint('BSF'         ,'Peak',BSF        ,0 ,'absoluto',1),
                    fingerprint('BSF-FTF'     ,'Peak',BSF-FTF   ,0 ,'absoluto',1),
                    fingerprint('BSF+FTF'     ,'Peak',BSF+FTF   ,0 ,'absoluto',1),
                    fingerprint('2*BSF'       ,'Peak',2*BSF      ,0 ,'absoluto',1),
                    fingerprint('2*BSF-FTF'   ,'Peak',2*BSF-FTF ,0 ,'absoluto',1),
                    fingerprint('2*BSF+FTF'   ,'Peak',2*BSF+FTF ,0 ,'absoluto',1),
                                       
                    fingerprint('FTF'         ,'Peak',FTF        ,0 ,'absoluto',1),
                    fingerprint('2*FTF'       ,'Peak',2*FTF      ,0 ,'absoluto',1),
                    fingerprint('3*FTF'       ,'Peak',3*FTF      ,0 ,'absoluto',1),
                    fingerprint('4*FTF'       ,'Peak',4*FTF      ,0 ,'absoluto',1),
                    
                    fingerprint('VPF'         ,'Peak',VPF        ,0 ,'absoluto',1),
                    fingerprint('VPF+f'       ,'Peak',VPF        ,1 ,'mixto',1),
                    fingerprint('VPF-f'       ,'Peak',VPF        ,-1,'mixto',1),
                    
                    fingerprint('2*VPF'       ,'Peak',2*VPF     ,0 ,'absoluto',1),
                    fingerprint('2*VPF+f'     ,'Peak',2*VPF     ,1 ,'mixto',1),
                    fingerprint('2*VPF-f'     ,'Peak',2*VPF     ,-1,'mixto',1)
                    
                    
                    ]
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
 
def PK(threshold,a):
    if a > threshold:
        out = True
    else:
        out = False
    return out
 
def PEAKS(threshold,*args):
    #E1 = 0.15
    n_peaks = np.size(args)
    counter = 0
    for peak in args:
        if peak > threshold:
            counter = counter+1
    if counter == n_peaks:
        out = True
    else:
        out = False
    return out
     
def NO_PEAKS(threshold,*args):
    #E1 = 0.15
    out = True
    for peak in args:
        if peak > threshold:
            out = False
    return out

def Number_PEAKS(threshold,*args):
    #E1 = 0.15
    out = 0
    for peak in args:
        if peak > threshold:
            out = out + 1
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
# Python program to find N largest element from given list of integers  
def Nmaxelements(x, N):
    
    #print(pikos,'numero de pikos',np.size(pikos))
    pikos = np.copy(x)
    list_values = np.array([])
    list_index  = []
    if np.size(pikos) > 0:
        
        for i in range(0, N):
            max_value = 0
            indice    = -1
            for counter,valor in enumerate (pikos):
                if valor > max_value:
                    max_value = valor
                    indice    = counter
            
            if indice > -1:    
                list_index.append(indice)# = np.append(list_index,indice)
                list_values= np.append(list_values,max_value)
                
            pikos [indice] = 0
     
    #    print(final_list)
   # print(list_index)
    return list_index
#------------------------------------------------------------------------------        
def find_file(file1,file2):
    
    found1 = False
    found2 = False
    files  = os.listdir(Path_out)
#    print(files)
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
def Load_Vibration_Data_Global(parameters):
    
    if parameters['Source'] == 'Petronor Server': # ======>     Acceso servidor
        print('------------------------------------Accediendo servidor Petronor')
        # -----------------------------------------------construct API endpoint
        api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
        '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
        '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])
    
        # -------------------------------------make GET request to API endpoint
        response          = requests.get(api_endpoint)
        # -------------------------------------convert response from server into json object
        response_json     = response.json()        
        df_accel          = Load_Vibration_Data_From_Get(response_json, parameters)   
        df_speed,df_SPEED = velocity(df_accel)

    if parameters['Source'] == 'Local Database':  # ======>     Acceso Disco  
        print('----------------------------------Accediendo Basedatos en  local')  
        month          = parameters['Month']
        day            = parameters['Day']
        hour           = parameters['Hour']
        
        file_suffix    = parameters['IdAsset']+'_' + parameters['Localizacion'] + '_' + month+'_'+day+'_'+hour + '.pkl'
        fichero_speed  = 'speed_t_'+file_suffix
        fichero_SPEED  = 'SPEED_f_'+file_suffix
        Found          = find_file(fichero_speed,fichero_SPEED)
        #print (nombre_fichero,Found)
        if Found :
            print('-----------------------------------Accediendo fichero pickle  desde ficheros')                                                           
            df_speed           = pd.read_pickle(Path_out + fichero_speed)
            df_SPEED           = pd.read_pickle(Path_out + fichero_SPEED)
        else:
            print('------------------------------------Accediendo ficheros json desde BaseDatos')
            df_accel           = Load_Vibration_Data_From_DB(parameters['Path']+'//'+month+'//'+day+'//'+hour,parameters['IdAsset'],parameters['Localizacion'])
            df_speed,df_SPEED  = velocity(df_accel)
 
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
    df_out     = pd.DataFrame(data=data, index=date)
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
#            print(res.Props.iloc[0][0]['Value'])
#            print(res.Props.iloc[0][4]['Value'])
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

    df_out     = pd.DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out
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
#                print(res.Props.iloc[0][0]['Value'])
#                print(res.Props.iloc[0][4]['Value'])
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
 

#-----------------------------------------------------------------------------1
def Plain_Bearing_Clearance(df_in):
    print('-------------------------Clearance Failure--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Plain_Bearing_Clearance_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Plain_Bearing_Clearance_Failure'] = 'No vibration detected'
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
            
            df_in.loc[df_in.index[i],'$Plain_Bearing_Clearance_Failure'] = Truth_Table( not(A) and not(B) , A or B , A and B)
    return df_in
#-----------------------------------------------------------------------------2
def Plain_Bearing_Lubrication_Whirl(df_in):
    print('---------------------------Oil Whirl Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Plain_Bearing_Lubrication_Whirl_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Plain_Bearing_Lubrication_Whirl_Failure'] = 'No vibration detected'
        else:
                                                #-----------green-----------------
                                                # no detected Peak in '0.38-0.48'
            A = df_in.iloc[i]['RMS Oil Whirl'] > E1
                                                        #-----------yellow-----------------
                                                # Detected Peak in '0.38-0.48'
                                                #         but
                                                # Peak in '0.38-0.48' < 2% 1.0x
            B_peaks = PEAKS(E1,df_in.iloc[i]['RMS Oil Whirl'],df_in.iloc[i]['RMS 1.0']) 
            B       = B_peaks and df_in.iloc[i]['RMS Oil Whirl'] > 0.02 * df_in.iloc[i]['RMS 1.0']
            
            df_in.loc[df_in.index[i],'$Plain_Bearing_Lubrication_Whirl_Failure'] = Truth_Table( not(A) , A and (not B) , A and B)
            
            if df_in.iloc[i]['$Plain_Bearing_Lubrication_Whirl_Failure'] == 'None':
                print ('Fallo en:', i)
                print (df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS Oil Whirl'],A,B)
    return df_in
#-----------------------------------------------------------------------------3

def Plain_Bearing_Lubrication_Whip(df_in):
    print('---------------------------Oil Whip Failure-------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
  

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Plain_Bearing_Lubrication_Whip_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Plain_Bearing_Lubrication_Whip_Failure'] = 'No vibration detected'
        else:
            A       = (df_in.iloc[i]['RMS 1/2'] >= E1 and df_in.iloc[i]['BW 1/2'] >= 4)  and  ((df_in.iloc[i]['RMS 5/2'] >= E1) and df_in.iloc[i]['BW 5/2'] >= 4)
            B_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 1/2']) 
            B       = B_peaks and df_in.iloc[i]['RMS 1/2' ] > 0.02 *  df_in.iloc[i]['RMS 1.0']
            C_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 5/2'])
            C       = C_peaks and df_in.iloc[i]['RMS 5/2' ] > 0.02 *  df_in.iloc[i]['RMS 1.0']
            #print(A,B,C)
                                                 #  Tabla de verdad progresiva
                                                 #  puede empezar siendo verde,
                                                 #  acabar siendo rojo
    
                                                 #-----------green-----------------
                                                 # 2H BW at 0.5 = 0 and 2H BW at 2.5 = 0
    
            if A == False and ( (B and C) == False ):
                df_in.loc[df_in.index[i],'$Plain_Bearing_Lubrication_Whip_Failure'] = 'Green'
                                                 #---------yellow------------------
                                                 # 2H BW at 0.5 > 0
                                                 # 2H BW at 2.5 > 0
                                                 # 2H BW at 0.5 >2% 1.0x
                                                 # 2H BW at 2.5 >2% 1.0x
            if A ^ ((B ^ C)) :
                df_in.loc[df_in.index[i],'$Plain_Bearing_Lubrication_Whip_Failure'] = 'Yellow'
                                                 #-----------red-------------------
                                                 #     2H BW at 0.5 >2% 1.0x
                                                 #           AND
                                                 #     2H BW at 2.5 >2% 1.0x
            if A and B and C:
                df_in.loc[df_in.index[i],'Plain_Bearing_Lubrication_Whip_Failure'] = 'Red'
    return df_in
#-----------------------------------------------------------------------------4
def Plain_Bearing_Block_Looseness(df_in):
    print('-----------------------PBB looseness Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Plain_Bearing_Block_Looseness_Failure'] = none_list
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Plain_Bearing_Block_Looseness_Failure'] = 'No vibration detected'
        else:
            A_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 1.0'] <df_in.iloc[i]['RMS 2.0'] > df_in.iloc[i]['RMS 3.0']
            B       = PEAKS(E1,df_in.iloc[i]['RMS 1/2'],df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 1/4'])
            #print('Plain Bearin block   ',A,B)
            df_in.loc[df_in.index[i],'$Plain_Bearing_Block_Looseness_Failure'] = Truth_Table(not A and not B , A ^ B , A and B)
    return df_in
#-----------------------------------------------------------------------------5
def Centrifugal_Fan_Unbalance (df_in):
    print('--------------------Blower Wheel Unbalance Failure------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Centrifugal_Fan_Unbalance_Failure'] = none_list

    for i in range (n_traces):
        #print(df_in.iloc[i].values[1:8])
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Centrifugal_Fan_Unbalance_Failure'] = 'No vibration detected'
        else:
                                                #---1X meno que el umbral
            A_peaks = PK(E1,df_in.iloc[i]['RMS 1.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 1.0'] < 4
                                                #---El 15% 1x < resto armonicos.
                                                #   es decir 1X no es dominante
            B_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2th Max Value.'])
            B       =  B_peaks and df_in.iloc[i]['RMS 1.0'] * 0.15 < df_in.iloc[i]['RMS 2th Max Value.']
    
                                                #--------------------------Green
            if A and B:
                df_in.loc[df_in.index[i],'$Centrifugal_Fan_Unbalance_Failure'] = 'Green'
                                                #--------------------------yellow
                                                #   Xor = cualquiera de ellas
                                                #        pero no ambas
            if (A == False) ^   (B == False):
                df_in.loc[df_in.index[i],'$Centrifugal_Fan_Unbalance_Failure'] = 'Yellow'
                                                #--------------------------Red
                                                # las dos falsas
            if (A == False) and (B == False):
                df_in.loc[df_in.index[i],'$Centrifugal_Fan_Unbalance_Failure'] = 'Green'
    return df_in
#---------------------------------------------------------------------------6.1
def Severe_Misaligment_H4_FA_0001(df_in):
    print('----------------------------Severe Mis. Failure---------------------')
    n_traces  = df_in.shape[0]
    none_list = []

    threshold = 0.2
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Severe_Misaligment_H4_FA_0001_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0001_Failure'] = 'No vibration detected'
        else:
            counter_A = 0
            for m in [2,3,4,5,6,7,8,9,10]:
                if (df_in.iloc[i]['RMS '+str(m)+'.0'] > threshold*df_in.iloc[i]['RMS 1.0']) and PK(1,df_in.iloc[i]['RMS 1.0']):
                    counter_A = counter_A+1
            A       = counter_A >= 3
    
            counter_B = 0
            for m in ['5/2','7/2','9/2','11/2','13/2','15/2','17/2','19/2']:
                if (df_in.iloc[i]['RMS '          +m] > threshold*df_in.iloc[i]['RMS 1.0']) and PK(1,df_in.iloc[i]['RMS 1.0']):
                    counter_B = counter_B+1
            B       = counter_B >= 3
    
            C_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            C       = C_peaks and df_in.iloc[i]['RMS 2.0'] > df_in.iloc[i]['RMS 1.0']
            
            if not A:
                df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0001_Failure'] = 'Green'
            if A or B:               #---27/2/19 Dammika por tlfn cambio Xor por or
                df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0001_Failure'] = 'Yellow'
            if A and B and C:
                df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0001_Failure'] = 'Red'
    #        print(df_in.loc[df_in.index[i],'$Severe Mis. Failure'])
    return df_in
#---------------------------------------------------------------------------6.2
def Severe_Misaligment_H4_FA_0002(df_in):
    print('----------------------------Severe Mis. Failure---------------------')
    n_traces  = df_in.shape[0]
    none_list = []
    
    threshold = 0.1
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Severe_Misaligment_H4_FA_0002_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0002_Failure'] = 'No vibration detected'
        else:
            counter_A = 0
            for m in [2,3,4,5,6,7,8,9,10]:
                if (df_in.iloc[i]['RMS '+str(m)+'.0'] > threshold*df_in.iloc[i]['RMS 1.0']) and PK(1,df_in.iloc[i]['RMS 1.0']):
                    counter_A = counter_A+1
            A       = counter_A >= 3
    
            counter_B = 0
            for m in ['5/2','7/2','9/2','11/2','13/2','15/2','17/2','19/2']:
                if (df_in.iloc[i]['RMS '          +m] > threshold*df_in.iloc[i]['RMS 1.0']) and PK(1,df_in.iloc[i]['RMS 1.0']):
                    counter_B = counter_B+1
            B       = counter_B >= 3
    
            C_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            C       = C_peaks and df_in.iloc[i]['RMS 2.0'] > df_in.iloc[i]['RMS 1.0']
            
            if not A:
                df_in.loc[df_in.index[i],'$Severe_Misaligmen_H4_FA_0002t_Failure'] = 'Green'
            if A or B:               #---27/2/19 Dammika por tlfn cambio Xor por or
                df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0002_Failure'] = 'Yellow'
            if A and B and C:
                df_in.loc[df_in.index[i],'$Severe_Misaligment_H4_FA_0002_Failure'] = 'Red'
    #        print(df_in.loc[df_in.index[i],'$Severe Mis. Failure'])
    return df_in
#---------------------------------------------------------------------------7.1
def Blade_Faults(df_in):
    print('-----------------------Blade Faults Failure-------------------------')
    n_traces  = df_in.shape[0]
    none_list = []
   
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Blade_Faults_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Blade_Faults_Failure'] = 'No vibration detected'
        else:
            A = PK(E1,df_in.iloc[i]['RMS 12.0'])
            B = PEAKS(E1,df_in.iloc[i]['RMS 12.0'],df_in.iloc[i]['RMS 24.0']) 
            C = PK(E1,df_in.iloc[i]['RMS 12.0'])       and ( PK(E1,df_in.iloc[i]['RMS 11.0']) or PK(E1,df_in.iloc[i]['RMS 13.0']) )
            D = C and PK(E1,df_in.iloc[i]['RMS 24.0']) 
            E = C and PK(E1,df_in.iloc[i]['RMS 24.0']) and ( PK(E1,df_in.iloc[i]['RMS 23.0']) or PK(E1,df_in.iloc[i]['RMS 25.0']) )
            F = df_in.iloc[i]['RMS 12.0'] < E1 and df_in.iloc[i]['RMS 24.0'] < E1
            #print('Blade Faults         ',A,B,C,D,E,F)
                                                 #  Tabla de verdad progresiva
                                                 #  puede empezar siendo verde,
                                                 #  acabar siendo rojo
            df_in.loc[df_in.index[i],'$Blade_Faults_Failure'] = Truth_Table( A or B or F , C or D , E)

    return df_in
#---------------------------------------------------------------------------7.2
 
def Flow_Turbulence(df_in):
    print('---------------------Flow Turbulence Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Flow_Turbulence_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Flow_Turbulence_Failure'] = 'No vibration detected'
        else:
            A       = df_in.iloc[i]['RMS Flow T.'] <= 0.2
            B_peaks = PK(E1,df_in.iloc[i]['RMS 1.0'])
            B       = B_peaks and (0.2 <= df_in.iloc[i]['RMS Flow T.'] <= df_in.iloc[i]['RMS 1.0'])
            C_peaks = PK(E1,df_in.iloc[i]['RMS 1.0'])
            C       = C_peaks and (df_in.iloc[i]['RMS Flow T.'] >  df_in.iloc[i]['RMS 1.0'])
            #print('Flow Tur.           ',A,B,C)
            
            df_in.loc[df_in.index[i],'$Flow_Turbulence_Failure'] = Truth_Table(A,B,C)
           
    return df_in
#-----------------------------------------------------------------------------8 
def Pressure_Pulsations(df_in):
    print('-------------------------Pressure P. Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Pressure_Pulsations_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Pressure_Pulsations_Failure'] = 'No vibration detected'
        else:
            A       = PEAKS(E1,df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 2/3'],df_in.iloc[i]['RMS 4/3'],df_in.iloc[i]['RMS 5/3'])
            B_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1/3'],df_in.iloc[i]['RMS 4/3'],df_in.iloc[i]['RMS 8/3'],df_in.iloc[i]['RMS 4.0']) 
            B       = B_peaks and (df_in.iloc[i]['RMS 4/3'] > df_in.iloc[i]['RMS 1/3']) and (df_in.iloc[i]['RMS 8/3'] > df_in.iloc[i]['RMS 1/3']) and (df_in.iloc[i]['RMS 4.0'] > df_in.iloc[i]['RMS 1/3'])
            
            df_in.loc[df_in.index[i],'$Pressure_Pulsations_Failure'] = Truth_Table(not A , A , A and B)
       
    return df_in
#-----------------------------------------------------------------------------9 
def Shaft_Misaligments(df_in):
    print('--------------------------Shaft Mis. Failure------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Shaft_Misaligments_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Shaft_Misaligments_Failure'] = 'No vibration detected'
        else:
            A_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            A       = A_peaks and df_in.iloc[i]['RMS 2.0'] < 0.5 *  df_in.iloc[i]['RMS 1.0']
            B_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            B       = B_peaks and 1.5 *df_in.iloc[i]['RMS 1.0'] >        df_in.iloc[i]['RMS 2.0'] > 0.5 *df_in.iloc[i]['RMS 1.0']
            C_peaks = PEAKS(E1,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0'])
            C       = C_peaks and 1.5 *df_in.iloc[i]['RMS 1.0'] <        df_in.iloc[i]['RMS 2.0']
            D       = PEAKS(E1,df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'],df_in.iloc[i]['RMS 5.0'])
            
            df_in.loc[df_in.index[i],'$Shaft_Misaligments_Failure'] = Truth_Table(A or not D , B and D , C and D)
    return df_in

#----------------------------------------------------------------------------10
def Surge_Effect(df_in):
    print('-------------------------Surge E. Failure---------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    for i in range (n_traces):
        none_list.append('None')
    df_in['$Surge_Effect_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Surge_Effect_Failure'] = 'No vibration detected'
        else:
            A = PK(E1,df_in.iloc[i]['RMS Surge E. 0.33x 0.5x']) and df_in.iloc[i]['RMS Surge E. 0.33x 0.5x'] <= df_in.iloc[i]['RMS 1.0']
            B = PK(E1,df_in.iloc[i]['RMS Surge E. 12/20k'])
            C = PK(E1,df_in.iloc[i]['RMS Surge E. 0.33x 0.5x']) and df_in.iloc[i]['RMS Surge E. 0.33x 0.5x'] > df_in.iloc[i]['RMS 1.0']
            df_in.loc[df_in.index[i],'$Surge_Effect_Failure'] = Truth_Table(not A , A , C)
    return df_in
#----------------------------------------------------------------------------11
 
def Loose_Bedplate(df_in):
    print('---------------------Loose Bedplate Failure-------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Loose_Bedplate_Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Loose_Bedplate_Failure'] = 'No vibration detected'
        else:
            A       =                                     2.0 > df_in.iloc[i]['RMS 1.0'] > 0
            B       = PK(E1,df_in.iloc[i]['RMS 1.0']) and 2.0 < df_in.iloc[i]['RMS 1.0'] < 5.0 
            C       = PK(E1,df_in.iloc[i]['RMS 1.0']) and 5.0 < df_in.iloc[i]['RMS 1.0']
            #D_peaks = PEAKS(E1,df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'])
            #D       = D_peaks and df_in.iloc[i]['RMS 3.0'] > df_in.iloc[i]['RMS 2.0']
            D       = df_in.iloc[i]['RMS 3.0'] > df_in.iloc[i]['RMS 2.0']
            #print ('Loose Bedplate',A,B,C,D)
            df_in.loc[df_in.index[i],'$Loose_Bedplate_Failure'] = Truth_Table(A , B ^ C , C and D)              
            if df_in.iloc[i]['$Loose_Bedplate_Failure'] == 'None':
                print ('Fallo en:', i)
                print ('Loose Bedplate',df_in.iloc[i]['RMS 1.0'],A,B,C,D)
    return df_in


#==============================================================================
 
def Ball_Bearing_Outer_Race_Defects_22217C(df_in):
    print('-----------------Ball B. O. Race D. Failure_22217C------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball_Bearing_Outer_Race_Defects_22217C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Outer_Race_Defects_22217C_Failure'] = none_list
    else:
        print('Columna previemente creada')
        
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Outer_Race_Defects_22217C_Failure'] = 'No vibration detected'
        else:

            a1 = (df_in.iloc[i]['RMS BPFO1'] < E1) 
            a2 = (df_in.iloc[i]['RMS BPFO1'] > E1) and (df_in.iloc[i]['RMS 2*BPFO1'] < E1) and (df_in.iloc[i]['RMS 3*BPFO1'] < E1) and (df_in.iloc[i]['RMS 4*BPFO1'] < E1)
            A  = a1 or a2
            B  = PEAKS(E1,df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'])
            C  = PEAKS(E1,df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'],df_in.iloc[i]['RMS 3*BPFO1'])
            D  = PEAKS(E1,df_in.iloc[i]['RMS BPFO1'],df_in.iloc[i]['RMS 2*BPFO1'],df_in.iloc[i]['RMS 3*BPFO1'],df_in.iloc[i]['RMS 4*BPFO1'])
            
            df_in.loc[df_in.index[i],'$Ball_Bearing_Outer_Race_Defects_22217C_Failure']    = Truth_Table(A,B ^ C,D)
            
            if df_in.iloc[i]['$Ball_Bearing_Outer_Race_Defects_22217C_Failure'] == 'None':
                print ('Fallo en:', i)
                print(A,B,C)
    return df_in
#------------------------------------------------------------------------------    
 
def Ball_Bearing_Outer_Race_Defects_22219C(df_in):
    print('----------------Ball B. O. Race D. Failure_22219C-------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball_Bearing_Outer_Race_Defects_22219C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Outer_Race_Defects_22219C_Failure'] = none_list
    else:
        print('Columna previemente creada')
    
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$all_Bearing_Outer_Race_Defects_22219C_Failure'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BPFO2'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFO2'] > E1) and (df_in.iloc[i]['RMS 2*BPFO2'] < E1) and (df_in.iloc[i]['RMS 3*BPFO2'] < E1) and (df_in.iloc[i]['RMS 4*BPFO2'] < E1)
            A  = a1 or a2
            B  = PEAKS(E1,df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'])
            C  = PEAKS(E1,df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'],df_in.iloc[i]['RMS 3*BPFO2'])
            D  = PEAKS(E1,df_in.iloc[i]['RMS BPFO2'],df_in.iloc[i]['RMS 2*BPFO2'],df_in.iloc[i]['RMS 3*BPFO2'],df_in.iloc[i]['RMS 4*BPFO2'])
            
            df_in.loc[df_in.index[i],'$Ball_Bearing_Outer_Race_Defects_22219C_Failure'] = Truth_Table(A,B ^ C,D)
            
            if df_in.iloc[i]['$Ball_Bearing_Outer_Race_Defects_22219C_Failure'] == 'None':
                print ('Fallo en:', i)
    return df_in
#==============================================================================
 
def Ball_Bearing_Inner_Race_Defects_22217C(df_in):
    print('---------------Ball B. I. Race D. Failure_22217C--------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball_Bearing_Inner_Race_Defects_22217C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Inner_Race_Defects_22217C_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Inner_Race_Defects_22217C_Failure'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BPFI1'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFI1'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1'] < E1)
            A  = a1 or a2 
            B  = (df_in.iloc[i]['RMS BPFI1']   > E1) and (df_in.iloc[i]['RMS 2*BPFI1']  > E1)
            C  = (df_in.iloc[i]['RMS BPFI1']   > E1) and (df_in.iloc[i]['RMS BPFI1+f']   > E1) and (df_in.iloc[i]['RMS BPFI1-f']   > E1)
            D  = (df_in.iloc[i]['RMS 2*BPFI1'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1+f'] > E1) and (df_in.iloc[i]['RMS 2*BPFI1-f'] > E1)      
    
            df_in.loc[df_in.index[i],'$Ball_Bearing_Inner_Race_Defects_22217C_Failure'] = Truth_Table(A,B ^ C,D)
            
            if df_in.iloc[i]['$Ball_Bearing_Inner_Race_Defects_22217C_Failure'] == 'None':
                print ('Fallo en:', i)
    return df_in
#------------------------------------------------------------------------------ 
def Ball_Bearing_Inner_Race_Defects_22219C(df_in):
    print('------------------Ball B. I. Race D. Failure_22219C-----------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball_Bearing_Inner_Race_Defects_22219C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Inner_Race_Defects_22219C_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Inner_Race_Defects_22219C_Failure'] = 'No vibration detected'
        else:         
            a1 = (df_in.iloc[i]['RMS BPFI2'] < E1)
            a2 = (df_in.iloc[i]['RMS BPFI2'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2'] < E1)
            A  = a1 or a2 
            B  = (df_in.iloc[i]['RMS BPFI2']   > E1) and (df_in.iloc[i]['RMS 2*BPFI2']   > E1)
            C  = (df_in.iloc[i]['RMS BPFI2']   > E1) and (df_in.iloc[i]['RMS BPFI2+f']   > E1) and (df_in.iloc[i]['RMS BPFI2-f']   > E1)
            D  = (df_in.iloc[i]['RMS 2*BPFI2'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2+f'] > E1) and (df_in.iloc[i]['RMS 2*BPFI2-f'] > E1)  
    
            df_in.loc[df_in.index[i],'$Ball_Bearing_Inner_Race_Defects_22219C_Failure'] = Truth_Table(A,B ^ C,D)
    
            if df_in.iloc[i]['$Ball_Bearing_Inner_Race_Defects_22219C_Failure'] == 'None':
                print('Fallo en:', i, df_in.iloc[i]['RMS BPFI2'] ,df_in.iloc[i]['RMS 2*BPFI2'])
                print(df_in.iloc[i]['f BPFI2'] ,df_in.iloc[i]['RMS BPFI2'])
                print(df_in.iloc[i]['f 2*BPFI2'] ,df_in.iloc[i]['RMS 2*BPFI2'])
                print('A=',A)
                print('B=',B)
                print('C=',C)
                print('D=',D)
    return df_in
#==============================================================================
 
def Ball_Bearing_Ball_Defect_22217C(df_in):
    print('----------------------Ball B D. Failure_22217C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball_Bearing_Ball_Defect_22217C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Ball_Defect_22217C_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Ball_Defect_22217C_Failure'] = 'No vibration detected'
        else:
    
            a1 = (df_in.iloc[i]['RMS BSF1']   < E1)
            a2 = (df_in.iloc[i]['RMS BSF1']   > E1) and (df_in.iloc[i]['RMS 2*BSF1'] < E1)
            A  = a1 or a2 
            B  = PEAKS(E1,df_in.iloc[i]['RMS BSF1'],df_in.iloc[i]['RMS 2*BSF1'])
            C  = PEAKS(E1,df_in.iloc[i]['RMS BSF1']  ,df_in.iloc[i]['RMS BSF1+FTF1']  ,df_in.iloc[i]['RMS BSF1-FTF1'])
            D  = PEAKS(E1,df_in.iloc[i]['RMS 2*BSF1'],df_in.iloc[i]['RMS 2*BSF1+FTF1'],df_in.iloc[i]['RMS 2*BSF1-FTF1'])
            
            df_in.loc[df_in.index[i],'$Ball_Bearing_Ball_Defect_22217C_Failure'] = Truth_Table(A,B ^ C,D)
    
            if df_in.iloc[i]['$Ball_Bearing_Ball_Defect_22217C_Failure'] == 'None':
                print ('Fallo en:', i)
                print ('No peak at BSF1           ',a1,df_in.iloc[i]['f BSF1']   ,df_in.iloc[i]['RMS BSF1'] )
                print ('Peak at BSF1, no harmonics',a2,df_in.iloc[i]['f 2*BSF1'] ,df_in.iloc[i]['RMS 2*BSF1'] )
                print('A=',A)
                print('B=',B)
                print('C=',C)
                print('D=',D)
            
    return df_in
#------------------------------------------------------------------------------
 
def Ball_Bearing_Ball_Defect_22219C(df_in):
    print('----------------------Ball B D. Failure_22219C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    if ('$Ball_Bearing_Ball_Defect_22219C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Ball_Defect_22219C_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces): 
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Ball_Defect_22219C_Failure'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BSF2']   < E1)
            a2 = (df_in.iloc[i]['RMS BSF2']   > E1) and (df_in.iloc[i]['RMS 2*BSF2'] < E1)
            A = a1 or a2 
            B = PEAKS(E1,df_in.iloc[i]['RMS BSF2']  ,df_in.iloc[i]['RMS 2*BSF2'])
            C = PEAKS(E1,df_in.iloc[i]['RMS BSF2']  ,df_in.iloc[i]['RMS BSF2+FTF2']  ,df_in.iloc[i]['RMS BSF2-FTF2'])
            D = PEAKS(E1,df_in.iloc[i]['RMS 2*BSF2'],df_in.iloc[i]['RMS 2*BSF2+FTF2'],df_in.iloc[i]['RMS 2*BSF2-FTF2'])
            
            df_in.loc[df_in.index[i],'$Ball_Bearing_Ball_Defect_22219C_Failure'] = Truth_Table(A,B ^ C,D)
            if  df_in.iloc[i]['$Ball_Bearing_Ball_Defect_22219C_Failure'] == 'None':
                print ('Fallo en:', i)
            
    return df_in
#==============================================================================
 
def Ball_Bearing_Cage_Defect_22217C(df_in):
    print('----------------Ball B. Cage D. Failure_22217C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball_Bearing_Cage_Defect_22217C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Cage_Defect_22217C_Failure'] = none_list
    else:
        print('Columna previemente creada')


    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Cage_Defect_22217C_Failure'] = 'No vibration detected'
        else:
            A = PK(E1,df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS FTF1'] < df_in.iloc[i]['RMS 1.0']  )
            B = (df_in.iloc[i]['RMS FTF1'] > df_in.iloc[i]['RMS 2*FTF1']) and (df_in.iloc[i]['RMS FTF1'] < df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS 2*FTF1'] < df_in.iloc[i]['RMS 1.0'])
            C = (df_in.iloc[i]['RMS FTF1'] > df_in.iloc[i]['RMS 2*FTF1'] > df_in.iloc[i]['RMS 1.0'] ) and PEAKS(E1, df_in.iloc[i]['RMS 3*FTF1'] > df_in.iloc[i]['RMS 4*FTF1'])
            

            df_in.loc[df_in.index[i],'$Ball_Bearing_Cage_Defect_22217C_Failure'] = Truth_Table(A,B,C)
            if df_in.iloc[i]['$Ball_Bearing_Cage_Defect_22217C_Failure'] == 'None':
                print ('Fallo en:', i)
            
    return df_in

#------------------------------------------------------------------------------
 
def Ball_Bearing_Cage_Defect_22219C(df_in):
    print('----------------Ball B. Cage D. Failure_22219C----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball_Bearing_Cage_Defect_22219C_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball_Bearing_Cage_Defect_22219C_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball_Bearing_Cage_Defect_22219C_Failure'] = 'No vibration detected'
        else:
            
            A = PK(E1,df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS FTF2'] < df_in.iloc[i]['RMS 1.0']  )
            B = (df_in.iloc[i]['RMS FTF2'] > df_in.iloc[i]['RMS 2*FTF2']) and (df_in.iloc[i]['RMS FTF2'] < df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS 2*FTF2'] < df_in.iloc[i]['RMS 1.0'])
            C = (df_in.iloc[i]['RMS FTF2'] > df_in.iloc[i]['RMS 2*FTF2'] > df_in.iloc[i]['RMS 1.0'] ) and PEAKS(E1, df_in.iloc[i]['RMS 3*FTF2'] > df_in.iloc[i]['RMS 4*FTF2'])
            
            df_in.loc[df_in.index[i],'$Ball_Bearing_Cage_Defect_22219C_Failure'] = Truth_Table(A,B,C)
            if df_in.iloc[i]['$Ball_Bearing_Cage_Defect_22219C_Failure'] == 'None':
                print ('Fallo en:', i)
            
    return df_in

#----------------------------PUMPS---------------------------------------------
#-----------------------------------------------------------------------------1
 
def Recirculation_in_pump(df_in):
    print('---------------------Recirculation_in_pump-------------------------')
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
            if df_in.iloc[i]['$Recirculation_in_pump'] == 'None':
                print ('Fallo en:', i)
    return df_in

#-----------------------------------------------------------------------------2

 
def Impeller_Rotor_Unbalance(df_in):
    print('----------------------Impeller_Rotor_Unbalance----------------------')
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
      
            A  = 0.15 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2th Max Value.']
            B  = df_in.iloc[i]['RMS 1.0'] > 4
            
            df_in.loc[df_in.index[i],'$Impeller_Rotor_Unbalance'] = Truth_Table( A and B,A ^ B,(not A) and (not B))            
            if df_in.iloc[i]['$Impeller_Rotor_Unbalance'] == 'None':
                print ('Fallo en:', i)
            
    return df_in
  
#-----------------------------------------------------------------------------3
 
def Shaft_misaligment_Radial(df_in):
    print('---------------------Shaft_misaligment_Radial-----------------------')
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
            
            
            a1       = PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0']) and (0.5 * df_in.iloc[i]['RMS 1.0'] > df_in.iloc[i]['RMS 2.0'])    
            a2       = PK(E2,df_in.iloc[i]['RMS 1.0']) and NO_PEAKS(E2,df_in.iloc[i]['RMS 2.0'],df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'])
            A        = a1 or a2
            B        = PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0']) and (0.5 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2.0'] < 1.5 * df_in.iloc[i]['RMS 1.0'])
            C        = PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0']) and (1.5 * df_in.iloc[i]['RMS 1.0'] < df_in.iloc[i]['RMS 2.0'] )
            D        = PEAKS(E2,df_in.iloc[i]['RMS 3.0'],df_in.iloc[i]['RMS 4.0'])
    
            #print (A,B,C,D)
            df_in.loc[df_in.index[i],'$Shaft_misaligment_Radial'] = Truth_Table(A and (not D),B or D,C )
            if df_in.iloc[i]['$Shaft_misaligment_Radial'] == 'None':
                print ('Fallo en:', i)
                print (A,B,C,D)
                print(PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 2.0']))
                print (df_in.iloc[i]['RMS 1.0'])
                print (df_in.iloc[i]['RMS 2.0'], )
                print (df_in.iloc[i]['RMS 3.0'])
                print (df_in.iloc[i]['RMS 4.0'])
               
    return df_in    

#-----------------------------------------------------------------------------4
 
def extract_index(instante,df_in):
    # -----------Devuelve el indice de la caotura más próxima en tiempo--------
    indice_captura = -1
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
    print('-------------------Shaft_misaligment_Axial--------------------------')
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
            A  = ( E2 < df_in_A.iloc[i]['RMS 1.0'] < 2.5) and PK(E2,df_in_A.iloc[i]['RMS 2.0']) or  (df_in_A.iloc[i]['RMS 2.0'] < E2 )
            B  = (2.4 < df_in_A.iloc[i]['RMS 1.0'] < 4  ) and PK(E2,df_in_A.iloc[i]['RMS 2.0'])
            C  = (  4 < df_in_A.iloc[i]['RMS 1.0']      ) and PEAKS(E2,df_in_A.iloc[i]['RMS 2.0'],df_in_A.iloc[i]['RMS 3.0'])
            
            if i_H >= 0 and i_V >= 0 :
                D = PEAKS(E2, df_in_H.iloc[i_H]['RMS 1.0'],df_in_H.iloc[i_H]['RMS 2.0']) or PEAKS(df_in_V.iloc[i_V]['RMS 1.0'],df_in_V.iloc[i_V]['RMS 2.0'])
            else:
                #print('no hya dtos radiales')
                D = False
                
            df_in_A.loc[df_in_A.index[i],'$Shaft_misaligment_Axial'] = Truth_Table(A and (not B),A or B,C or D)
            if df_in_A.iloc[i]['$Shaft_misaligment_Axial'] == 'None':
                print ('Fallo en:', i)
    return df_in_A    

#-----------------------------------------------------------------------------5
 
def Hydraulic_Instability(df_in):
    print('------------------------Hydraulic_Instability-----------------------')
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
            A  =  df_in.iloc[i]['RMS Hydr. Inst.'] < E2
            B  = PK(E2,df_in.iloc[i]['RMS 1.0']) and (E2 < df_in.iloc[i]['RMS Hydr. Inst.'] < 0.5 * df_in.iloc[i]['RMS 1.0']) 
            C  = PK(E2,df_in.iloc[i]['RMS 1.0']) and (     df_in.iloc[i]['RMS Hydr. Inst.'] > 0.5 * df_in.iloc[i]['RMS 1.0']) 
            
            df_in.loc[df_in.index[i],'$Hydraulic_Instability'] = Truth_Table(A,B,C)
            
            if df_in.iloc[i]['$Hydraulic_Instability'] == 'None':
                print ('Fallo en:', i)
                print(A,B,C)
                print(df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS Hydr. Inst.'])
    return df_in    
#-----------------------------------------------------------------------------6
 
def Ball_Bearing_Outer_Race_Defects_7310BEP(df_in):
    print('------------------Ball B. O. Race D. Failure_7310BEP----------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball B. O. Race D. Failure_7310BEP' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. O. Race D. Failure_7310BEP'] = none_list
    else:
        print('Columna previemente creada')
        
    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_7310BEP'] = 'No vibration detected'
        else:

            a1 = (df_in.iloc[i]['RMS BPFO'] < E2) 
            a2 = (df_in.iloc[i]['RMS BPFO'] > E2) and (df_in.iloc[i]['RMS 2*BPFO'] < E2) and (df_in.iloc[i]['RMS 3*BPFO'] < E2) and (df_in.iloc[i]['RMS 4*BPFO'] < E2)
            A  = a1 or a2
            B  = PEAKS(E2,df_in.iloc[i]['RMS BPFO'],df_in.iloc[i]['RMS 2*BPFO'])
            C  = PEAKS(E2,df_in.iloc[i]['RMS BPFO'],df_in.iloc[i]['RMS 2*BPFO'],df_in.iloc[i]['RMS 3*BPFO'])
            D  = PEAKS(E2,df_in.iloc[i]['RMS BPFO'],df_in.iloc[i]['RMS 2*BPFO'],df_in.iloc[i]['RMS 3*BPFO'],df_in.iloc[i]['RMS 4*BPFO'])
            
            df_in.loc[df_in.index[i],'$Ball B. O. Race D. Failure_7310BEP']    = Truth_Table(A,B ^ C,D)
            
            if df_in.iloc[i]['$Ball B. O. Race D. Failure_7310BEP'] == 'None':
                print ('Fallo en:', i)
    return df_in
#-----------------------------------------------------------------------------7
 
def Ball_Bearing_Inner_Race_Defects_7310BEP(df_in):
    print('---------------Ball B. I. Race D. Failure_7310BEP-------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. I. Race D. Failure_7310BEP' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. I. Race D. Failure_7310BEP'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_7310BEP'] = 'No vibration detected'
        else:
            a1 = (df_in.iloc[i]['RMS BPFI'] < E2)
            a2 = (df_in.iloc[i]['RMS BPFI'] > E2) and (df_in.iloc[i]['RMS 2*BPFI'] < E2)
            A  = a1 or a2 
            B  = (df_in.iloc[i]['RMS BPFI']   > E2) and (df_in.iloc[i]['RMS 2*BPFI']  > E2)
            C  = (df_in.iloc[i]['RMS BPFI']   > E2) and (df_in.iloc[i]['RMS BPFI+f']   > E2) and (df_in.iloc[i]['RMS BPFI-f']   > E2)
            D  = (df_in.iloc[i]['RMS 2*BPFI'] > E2) and (df_in.iloc[i]['RMS 2*BPFI+f'] > E2) and (df_in.iloc[i]['RMS 2*BPFI-f'] > E2)      
    
            df_in.loc[df_in.index[i],'$Ball B. I. Race D. Failure_7310BEP'] = Truth_Table(A,B ^ C,D)
            
            if df_in.iloc[i]['$Ball B. I. Race D. Failure_7310BEP'] == 'None':
                print ('Fallo en:', i)
    return df_in
#-----------------------------------------------------------------------------8    
 
def Ball_Bearing_Defect_7310BEP(df_in):
    print('---------------------Ball B D. Failure_7310BEP----------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
    
    if ('$Ball B D. Failure_7310BEP' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B D. Failure_7310BEP'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B D. Failure_7310BEP'] = 'No vibration detected'
        else:
    
            a1 = (df_in.iloc[i]['RMS BSF']   < E2)
            a2 = (df_in.iloc[i]['RMS BSF']   > E2) and (df_in.iloc[i]['RMS 2*BSF'] < E2)
            A  = a1 or a2 
            B  = PEAKS(E2,df_in.iloc[i]['RMS BSF'],df_in.iloc[i]['RMS 2*BSF'])
            C  = PEAKS(E2,df_in.iloc[i]['RMS BSF']  ,df_in.iloc[i]['RMS BSF+FTF']  ,df_in.iloc[i]['RMS BSF-FTF'])
            D  = PEAKS(E2,df_in.iloc[i]['RMS 2*BSF'],df_in.iloc[i]['RMS 2*BSF+FTF'],df_in.iloc[i]['RMS 2*BSF-FTF'])
            
            df_in.loc[df_in.index[i],'$Ball B D. Failure_7310BEP'] = Truth_Table(A,B or  C,D)
    
            if df_in.iloc[i]['$Ball B D. Failure_7310BEP'] == 'None':
                print ('Fallo en:', i,df_in.index[i])
                print ('No peak at BSF           ',a1,df_in.iloc[i]['f BSF']   ,df_in.iloc[i]['RMS BSF'] )
                print ('Peak at BSF, no harmonics',a2,df_in.iloc[i]['f 2*BSF'] ,df_in.iloc[i]['RMS 2*BSF'] )
                print('A=',A)
                print('B=',B)
                print('C=',C)
                print('D=',D)
            
    return df_in
#-----------------------------------------------------------------------------9
 
def Ball_Bearing_Cage_Defect_7310BEP(df_in):
    print('-----------------Ball B. Cage D. Failure_7310BEP--------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Ball B. Cage D. Failure_7310BEP' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Ball B. Cage D. Failure_7310BEP'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_7310BEP'] = 'No vibration detected'
        else:
            
            A =  df_in.iloc[i]['RMS FTF'] < df_in.iloc[i]['RMS 1.0']  
            B =  df_in.iloc[i]['RMS FTF'] > df_in.iloc[i]['RMS 2*FTF'] and (df_in.iloc[i]['RMS FTF'] < df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS 2*FTF'] < df_in.iloc[i]['RMS 1.0']) 
            C = (df_in.iloc[i]['RMS FTF'] > df_in.iloc[i]['RMS 2*FTF'] > df_in.iloc[i]['RMS 1.0']) and (df_in.iloc[i]['RMS 3*FTF'] >0) and (df_in.iloc[i]['RMS 4*FTF'] >0)
            

            df_in.loc[df_in.index[i],'$Ball B. Cage D. Failure_7310BEP'] = Truth_Table(A,B,C)
            if df_in.iloc[i]['$Ball B. Cage D. Failure_7310BEP'] == 'None':
                print(df_in.iloc[i]['RMS 1.0'])
                print ('Fallo en:', i)
                
                print(A,B,C)
    return df_in

#----------------------------------------------------------------------------10
 
def R_Rotating_Stall(df_in):
    print('----------------------R_Rotating_Stall------------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$R_Rotating_Stall' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$R_Rotating_Stall'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$R_Rotating_Stall'] = 'No vibration detected'
        else:     
            A  =       df_in.iloc[i]['RMS R. R. Stall'] < E2
            B  = E2  < df_in.iloc[i]['RMS R. R. Stall'] < 0.2
            C  = 0.2 < df_in.iloc[i]['RMS R. R. Stall']
            
            df_in.loc[df_in.index[i],'$R_Rotating_Stall'] = Truth_Table(A,B,C)
            
            if df_in.iloc[i]['$R_Rotating_Stall'] == 'None':
                print ('Fallo en:', i)
                
    return df_in 

#----------------------------------------------------------------------------11
 
def Rotating_Cavitation(df_in):
    print('----------------------Rotating_Cavitation---------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Rotating_Cavitation' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Rotating_Cavitation'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Rotating_Cavitation'] = 'No vibration detected'
        else:     
            A  =       df_in.iloc[i]['RMS Rotat. Cavit'] < E2
            B  = E2  < df_in.iloc[i]['RMS Rotat. Cavit'] < 0.3
            C  = 0.3 < df_in.iloc[i]['RMS Rotat. Cavit']
            
            df_in.loc[df_in.index[i],'$Rotating_Cavitation'] = Truth_Table(A,B,C)
            
            if df_in.iloc[i]['$Rotating_Cavitation'] == 'None':
                print ('Fallo en:', i)               
    return df_in 

#----------------------------------------------------------------------------12    
 
def Cavitation_Noise(df_in):
    print('---------------------Cavitation_Noise-------------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Cavitation_Noise' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Cavitation_Noise'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Cavitation_Noise'] = 'No vibration detected'
        else:     
            A  =       df_in.iloc[i]['RMS Cavit. Noise'] < E2
            B  = E2  < df_in.iloc[i]['RMS Cavit. Noise'] < 0.5
            C  = 0.5 < df_in.iloc[i]['RMS Cavit. Noise']
            
            df_in.loc[df_in.index[i],'$Cavitation_Noise'] = Truth_Table(A,B,C)
            
            if df_in.iloc[i]['$Cavitation_Noise'] == 'None':
                print ('Fallo en:', i)
                
    return df_in    
#----------------------------------------------------------------------------13    
 
def Piping_Vibration(df_in):
    print('--------------------------Piping_Vibration--------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Piping_Vibration' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Piping_Vibration'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Piping_Vibration'] = 'No vibration detected'
        else:     
            A  = NO_PEAKS(E2,df_in.iloc[i]['RMS 1th Pip Vib.'],df_in.iloc[i]['RMS 2th Pip Vib.'],df_in.iloc[i]['RMS 3th Pip Vib.'])
            
            b0 =       df_in.iloc[i]['RMS 1th Pip Vib.'] < 2.8 and df_in.iloc[i]['RMS 2th Pip Vib.'] < 2.8 # ninguna mayor de 2.8
            b1 = PK(E2,df_in.iloc[i]['RMS 1th Pip Vib.'])      and df_in.iloc[i]['RMS 1th Pip Vib.'] < 2.8
            b2 = PK(E2,df_in.iloc[i]['RMS 2th Pip Vib.'])      and df_in.iloc[i]['RMS 2th Pip Vib.'] < 2.8
            B  = b0 and (b1 ^ b2)
            
            c1 = df_in.iloc[i]['RMS 1th Pip Vib.'] > 2.8
            c2 = df_in.iloc[i]['RMS 2th Pip Vib.'] > 2.8
            C  = c1 ^ c2
            
            d1 = PK(E2,df_in.iloc[i]['RMS 1th Pip Vib.']) and df_in.iloc[i]['RMS 1th Pip Vib.'] < 2.8
            d2 = PK(E2,df_in.iloc[i]['RMS 2th Pip Vib.']) and df_in.iloc[i]['RMS 2th Pip Vib.'] < 2.8 
            D  = d1 and d2
            
            e1 = df_in.iloc[i]['RMS 1th Pip Vib.'] > 2.8
            e2 = df_in.iloc[i]['RMS 2th Pip Vib.'] > 2.8
            E  = e1 ^ e2
            
            df_in.loc[df_in.index[i],'$Piping_Vibration'] = Truth_Table(A^B,C^D,E)
            
            if df_in.iloc[i]['$Piping_Vibration'] == 'None':
                print ('Fallo en:', i)
                
    return df_in    

#----------------------------------------------------------------------------14
 
def Plain_Bearing_Clearance_pumps(df_in):
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
            a1 = PEAKS(E2,v_1x,v_2x,v_3x)            and v_1x >v_2x > v_3x
                                                # --2.0x >2% 1.0x and 3.0x >2% 1.0x
            a2 = PEAKS(E2,v_1x,v_2x,v_3x)            and (v_2x > 0.02 * v_1x) and (v_3x > 0.02 * v_1x)
            A  = a1 and a2
                                                #-------0.5x 1.5x 2.5x decreciente
            b1 = PEAKS(E2,v_0_5x,v_1_5x,v_2_5x)      and v_0_5x > v_1_5x > v_2_5x
                                                # ------0.5x >2% 1.0x and 1.5x > 2% 1.0x and 2.5x > 2% 1.0x
            b2 = PEAKS(E2,v_0_5x,v_1x,v_1_5x,v_2_5x) and (v_0_5x > 0.02 * v_1x) and (v_1_5x > 0.02 * v_1x) and (v_2_5x > 0.02 * v_1x)
            B  = b1 and b2
            
            df_in.loc[df_in.index[i],'$Plain Bearing Clearance Failure'] = Truth_Table( not(A) and not(B) , A or B , A and B)
    return df_in

#----------------------------------------------------------------------------15
 
def Vane_Failure(df_in):
    print('---------------------------Vane_Failure-----------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Vane_Failure' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Vane_Failure'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Vane_Failure'] = 'No vibration detected'
        else:     
            A  = PK   (E2,df_in.iloc[i]['RMS VPF'] )
            B  = PEAKS(E2,df_in.iloc[i]['RMS VPF'],df_in.iloc[i]['RMS 2*VPF']) 
            C  = PEAKS(E2,df_in.iloc[i]['RMS VPF'],df_in.iloc[i]['RMS VPF-f'],df_in.iloc[i]['RMS VPF+f'])
            D  = C and PEAKS(E2,df_in.iloc[i]['RMS 2*VPF'],df_in.iloc[i]['RMS 2*VPF-f'],df_in.iloc[i]['RMS 2*VPF+f'])
            E  = NO_PEAKS(E2,df_in.iloc[i]['RMS VPF'],df_in.iloc[i]['RMS 2*VPF'])
            
            df_in.loc[df_in.index[i],'$Vane_Failure'] = Truth_Table(A or B or C,C or D, E)
            
            if df_in.iloc[i]['$Vane_Failure'] == 'None':
                print ('Fallo en:', i)
                
    return df_in 

#----------------------------------------------------------------------------16
 
def Oil_Whirl_pumps(df_in):
    print('---------------------------Oil Whirl Failure------------------------')
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
            A = df_in.iloc[i]['RMS Oil Whirl'] > E2
                                                        #-----------yellow-----------------
                                                # Detected Peak in '0.38-0.48'
                                                #         but
                                                # Peak in '0.38-0.48' < 2% 1.0x
            B_peaks = PEAKS(E2,df_in.iloc[i]['RMS Oil Whirl'],df_in.iloc[i]['RMS 1.0']) 
            B       = B_peaks and df_in.iloc[i]['RMS Oil Whirl'] > 0.02 * df_in.iloc[i]['RMS 1.0']
            
            df_in.loc[df_in.index[i],'$Oil Whirl Failure'] = Truth_Table( not(A) , A and (not B) , A and B)

            if df_in.iloc[i]['$Oil Whirl Failure'] == 'None':
                print ('Fallo en:', i)
                print (df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS Oil Whirl'],A,B)
    return df_in
#----------------------------------------------------------------------------17
 
def Oil_Whip_pumps(df_in):
    print('---------------------------Oil Whip Failure-------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []
  

    for i in range (n_traces):
        none_list.append('None')
    df_in['$Oil Whip Failure'] = none_list

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Oil Whip Failure'] = 'No vibration detected'
        else:
            A       = (df_in.iloc[i]['RMS 1/2'] >= E2 and df_in.iloc[i]['BW 1/2'] >= 4)  and  ((df_in.iloc[i]['RMS 5/2'] >= E2) and df_in.iloc[i]['BW 2.5'] >= 4)
            B_peaks = PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 1/2']) 
            B       = B_peaks and df_in.iloc[i]['RMS 1/2' ] > 0.02 *  df_in.iloc[i]['RMS 1.0']
            C_peaks = PEAKS(E2,df_in.iloc[i]['RMS 1.0'],df_in.iloc[i]['RMS 5/2'])
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
#----------------------------------------------------------------------------18
 
def Loosness_pumps(df_in_H,df_in_A):
    print('---------------------Loosness_pumps---------------------------------')
    n_traces_c   = np.min([df_in_H.shape[0],df_in_A.shape[0]])
    n_traces_A   = df_in_A.shape[0]
    
    none_list  = []

    if ('$Loosness_pumps' in df_in_A.columns) == False: 
        for i in range (n_traces_A):
            none_list.append('None')
        df_in_A['$Loosness_pumps'] = none_list
        if n_traces_A > n_traces_c:
            for k in range(n_traces_c,n_traces_A):
                df_in_A.loc[df_in_A.index[k],'$Loosness_pumps'] = 'No data'
    else:
        print('Columna previemente creada')

    for i in range (n_traces_c):
        instante = time.mktime((df_in_A.index[i].timetuple()))
        i_H      = extract_index(instante,df_in_H)

        
        if df_in_A.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in_A.loc[df_in_A.index[i],'$Loosness_pumps'] = 'No vibration detected'
        else:     
#            print(i)
#            print (df_in_H.iloc[i]['RMS 1.0'],df_in_A.iloc[i]['RMS 1.0'])
#            print ()
            A  = ( E2  < df_in_H.iloc[i]['RMS 1.0'] < 2.5) or  ( E2  < df_in_A.iloc[i]['RMS 1.0'] < 2.5 )
            B  = ( 2.5 < df_in_H.iloc[i]['RMS 1.0'] < 6  ) or  ( 2.5 < df_in_A.iloc[i]['RMS 1.0'] < 6   )
            C  = ( 6   < df_in_H.iloc[i]['RMS 1.0']      ) or  ( 6   < df_in_A.iloc[i]['RMS 1.0']       )
                        #---------HORIZONTAL SENSOR
            if i_H >= 0 :
                D = PEAKS(E2, df_in_H.iloc[i_H]['RMS 2.0'],df_in_H.iloc[i_H]['RMS 3.0']) 
            else:
                #print('no hya dtos radiales')
                D = False
                
            df_in_A.loc[df_in_A.index[i],'$Loosness_pumps'] = Truth_Table(A,A ^ B,C and D)
            if df_in_A.loc[df_in_A.index[i],'$Loosness_pumps'] == 'None':
                print ('Fallo en:', i)
    return df_in_A        

#----------------------------------------------------------------------------19
 
def Dynamic_instability(df_in_H,df_in_A):
    print('---------------------dynamic instability----------------------------')
    n_traces_c   = np.min([df_in_H.shape[0],df_in_A.shape[0]])
    n_traces_A   = df_in_A.shape[0]
    
    none_list  = []

    if ('$Dynamic_instability' in df_in_A.columns) == False: 
        for i in range (n_traces_A):
            none_list.append('None')
        df_in_A['$Dynamic_instability'] = none_list
        if n_traces_A > n_traces_c:
            for k in range(n_traces_c,n_traces_A):
                df_in_A.loc[df_in_A.index[k],'$Dynamic_instability'] = 'No data'
    else:
        print('Columna previemente creada')

    for i in range (n_traces_c):
        instante = time.mktime((df_in_A.index[i].timetuple()))
        i_H      =extract_index(instante,df_in_H)
        
        if df_in_A.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in_A.loc[df_in_A.index[i],'$Dynamic_instability'] = 'No vibration detected'
        else:     
#            print(i)
#            print (df_in_H.iloc[i]['RMS 1.0'],df_in_A.iloc[i]['RMS 1.0'])
#            print ()
            A  = PEAKS(E2,df_in_A.iloc[i]['RMS 1.0'], df_in_H.iloc[i]['RMS 1.0']) and (df_in_A.iloc[i]['RMS 1.0'] > df_in_H.iloc[i]['RMS 1.0'])
            B  = df_in_H.iloc[i]['RMS VPF'] > df_in_A.iloc[i]['RMS VPF'] >4.0
            C  = df_in_H.iloc[i]['RMS VPF'] < df_in_A.iloc[i]['RMS VPF']
            d1 = df_in_A.iloc[i]['RMS 1.0'] > df_in_A.iloc[i]['RMS 2.0'] > df_in_A.iloc[i]['RMS 3.0']
            d2 = df_in_A.iloc[i]['RMS 2.0'] > 0.02*df_in_A.iloc[i]['RMS 1.0']
            d3 = df_in_A.iloc[i]['RMS 3.0'] > 0.02*df_in_A.iloc[i]['RMS 1.0']
            D = d1 and d2 and d3
            
            df_in_A.loc[df_in_A.index[i],'$Dynamic_instability'] = Truth_Table(not A,A or B,C or D)
            if df_in_A.iloc[i]['$Dynamic_instability'] == 'None':
                print ('Fallo en:', i)
    return df_in_A  





#----------------------------------------------------------------------------20
def Auto_Oscillation(df_in):
    print('------------------------Auto_Oscillation----------------------------')
    n_traces   = df_in.shape[0]
    none_list  = []

    if ('$Auto_Oscillation' in df_in.columns) == False: 
        for i in range (n_traces):
            none_list.append('None')
        df_in['$Auto_Oscillation'] = none_list
    else:
        print('Columna previemente creada')

    for i in range (n_traces):
        if df_in.iloc[i]['RMS (mm/s) f'] < 0.3:
            df_in.loc[df_in.index[i],'$Auto_Oscillation'] = 'No vibration detected'
        else:     
            A  = NO_PEAKS(E2,df_in.iloc[i]['RMS 1th Auto Osc.'],df_in.iloc[i]['RMS 2th Auto Osc.'],df_in.iloc[i]['RMS 3th Auto Osc.'])
#            b0 =       df_in.iloc[i]['RMS 1th Auto Osc.'] < 2.8 and df_in.iloc[i]['RMS 2th Auto Osc.'] < 2.8 # ninguna mayor de 2.8
#            b1 = PK(E2,df_in.iloc[i]['RMS 1th Auto Osc.'])      and df_in.iloc[i]['RMS 1th Auto Osc.'] < 2.8
#            b2 = PK(E2,df_in.iloc[i]['RMS 2th Auto Osc.'])      and df_in.iloc[i]['RMS 2th Auto Osc.'] < 2.
#            B  = b0 and (b1 ^ b2)
            
#            c1 = df_in.iloc[i]['RMS 1th Auto Osc.'] > 2.8
#            c2 = df_in.iloc[i]['RMS 2th Auto Osc.'] > 2.8
#            C  = c1 ^ c2
            
#            d1 = PK(E2,df_in.iloc[i]['RMS 1th Auto Osc.']) and df_in.iloc[i]['RMS 1th Auto Osc.'] < 2.8
#            d2 = PK(E2,df_in.iloc[i]['RMS 2th Auto Osc.']) and df_in.iloc[i]['RMS 2th Auto Osc.'] < 2.8 
#            D  = d1 and d2
   
          
#            e1 = df_in.iloc[i]['RMS 1th Auto Osc.'] > 2.8
#            e2 = df_in.iloc[i]['RMS 2th Auto Osc.'] > 2.8
            
#            E  = e1 ^ e2
            counter1 = 0
            counter2 = 0
            for name in ['RMS 1th Auto Osc.','RMS 2th Auto Osc.','RMS 3th Auto Osc.']:
                if E2 < df_in.iloc[i][name] < 2.8:
                    counter1 = counter1 +1
                if      df_in.iloc[i][name] > 2.8:
                    counter2 = counter2 +1            
            B = False
            if counter1 == 1:
                B = True            
            C = False
            if counter2 == 1:
                C = True          
            D = False
            if counter1 > 1:
                D = True
            E = False
            if counter2 > 1:
                E = True

            df_in.loc[df_in.index[i],'$Auto_Oscillation'] = Truth_Table(A^B,C^D,E)
            
            if df_in.iloc[i]['$Auto_Oscillation'] == 'None':
                print ('Fallo en:', i,df_in.index[i])
                print( A,B,C,D,E)
    return df_in

#==============================================================================
    #==============================================================================
    #-------ATENCION: df_FFT ya viene multiplicada por la ventana de hanning 
    #-------y corregido en potencia, es decir multiplicado por 1.63


#def df_Harmonics(df_speed,df_FFT,fs,machine_type):
def df_Harmonics_old(df_FFT,fs,machine_type):
    
    print('------------------------------------Extracting fingerprint')
    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]
    
    if machine_type == 'blower':
        print('----------------------------BLOWER----------------------')
        fingerprint_list = fprnt_list_blwrs
        f_1x_low         = 20
        f_1x_high        = 30
        f_1x_LOW         = 24
        f_1x_HIGH        = 25.5
        max_value_f_1x_i = 0.1
        fnom             = 24.7

    if machine_type == 'pump':
        fingerprint_list = fprnt_list_pumps
        f_1x_low         = 39
        f_1x_high        = 59
        f_1x_LOW         = 48.5
        f_1x_HIGH        = 49.1
        max_value_f_1x_i = 0.03
        fnom             = 48.75
        
    fecha = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))

    columnas = []
    word =''
    for k in fingerprint_list:
        if k.order == 1:
            word = k.label
            columnas.append('i '    + word)
            columnas.append('Point '+ word)
            columnas.append('RMS '  + word)
            columnas.append('f '    + word)
            columnas.append('BW '   + word)
            columnas.append('n_s '  + word)
            columnas.append('n_e '  + word)
            columnas.append('S/N '  + word)
        if k.order > 1:
            word = k.label
            for i in np.arange(k.order)+1:
                word_bis = str(i)+'th '
                columnas.append('i '    + word_bis + word)
                columnas.append('Point '+ word_bis + word)
                columnas.append('RMS '  + word_bis + word)
                columnas.append('f '    + word_bis + word)
                columnas.append('BW '   + word_bis + word)
                columnas.append('n_s '  + word_bis + word)
                columnas.append('n_e '  + word_bis + word)
                columnas.append('S/N '  + word_bis + word)
    
    columnas    = ['RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    i_f_1x_low  = int(np.round(f_1x_low*l/fs))
    i_f_1x_high = int(np.round(f_1x_high*l/fs))

    for medida in range(n_traces):
        
                                #--------N peaks within a certain bandwithd
#        print ('Medida numero-------------------------------------------------',medida,df_harm.index[medida])
                                
        f_1x                                 = 0
        abs_line                             = np.abs(df_FFT.iloc[medida].values)
        RMS_freq                             = np.sqrt(np.sum( abs_line**2 ) )
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        sptrm_C                              = abs_line * np.sqrt(2) 
                                              # integramos en 1º z Nyquist por eso el voltaje x raiz(2)         
#        find_f1x(sptrm_C,fnom)
        if RMS_freq > 0.06:
            for h in fingerprint_list:        
                if  h.order >1:
                    word = h.label
                    fa   = 0
                    fb   = 0
                    if h.f_norm == 'absoluto':   #-------------------------
                        fa = h.f1
                        fb = h.f2

                    i_fa                           = int(np.round(fa*l/fs))
                    i_fb                           = int(np.round(fb*l/fs))
                    indexes_excp, properties_excp  = find_peaks(sptrm_C[i_fa:i_fb],height  = 0.01 ,prominence = 0.01 , width=1 , rel_height = 0.75)
                    N_max_positions                = Nmaxelements(sptrm_C[i_fa+indexes_excp], 3)
                    
                    for counter,position in enumerate(N_max_positions):
                        word = h.label
                        word_bis = str(counter+1)+'th '            
#                        if h.label == 'Max Value.': # 'Pip Vib.','Max Value.'    
#                            print('indices absolutos-----',indexes_excp[position],'=>',indexes_excp[position]+i_fa,sptrm_C[i_fa+ indexes_excp[position]])            
                        init                                            = int(i_fa + np.round(properties_excp["left_ips"][position]))                                                                     
                        end                                             = int(i_fa + np.round(properties_excp["right_ips"][position]))  
                        piko                                            = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) 
#                        print('i '    + word_bis + word)
                        df_harm.iloc[medida]['i '    + word_bis + word] =         int(i_fa + indexes_excp[position])
                        df_harm.iloc[medida]['Point '+ word_bis + word] = sptrm_C[int(i_fa + indexes_excp[position])]
                        df_harm.iloc[medida]['RMS '  + word_bis + word] = piko #/ Max_value
                        df_harm.iloc[medida]['f '    + word_bis + word] = f      [int(i_fa + indexes_excp[position])]
                        df_harm.iloc[medida]['BW '   + word_bis + word] = f      [int(properties_excp["widths"][position]) ]
                        df_harm.iloc[medida]['n_s '  + word_bis + word] = init
                        df_harm.iloc[medida]['n_e '  + word_bis + word] = end
            
            if machine_type == 'blower':
#                i_f_1x_LOW                    = int(np.round(f_1x_LOW*l/fs))
#                i_f_1x_HIGH                   = int(np.round(f_1x_HIGH*l/fs))
#                indexes_f_1x, properties_f_1x = find_peaks(sptrm_C[i_f_1x_LOW:i_f_1x_HIGH],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
#                posicion                      = np.argmax( sptrm_C[i_f_1x_LOW+ indexes_f_1x] )
#                f_1x                          = f[i_f_1x_LOW+ indexes_f_1x[posicion]]
                
                indexes_f_1x, properties_f_1x        = find_peaks(sptrm_C[i_f_1x_low:i_f_1x_high],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
                f_1x_exist                           = False
                max_value_f_1x                       = max_value_f_1x_i
                            # miramos si hay algun pico entre 48 y 51Hz y nos quedamos con el mas grande
                            # y si lo encontramos entonces "f_1x_exist= True"
                for k in indexes_f_1x:    
                    if f_1x_LOW < f[i_f_1x_low + k] < f_1x_HIGH and sptrm_C[i_f_1x_low + k] > max_value_f_1x:
                        f_1x           = f[i_f_1x_low + k]
                        max_value_f_1x = sptrm_C[i_f_1x_low + k]
                        f_1x_exist     = True

            if machine_type == 'pump':
                ratio    = df_harm.iloc[medida]['f 1th Max Value.'] /48.75
                dist = np.abs(np.round(ratio)-ratio)
                if dist < 0.09 and ratio >1.5: #-----miro a ver si el pico mas grande es un armonco
                    f_1x = df_harm.iloc[medida]['f 1th Max Value.'] /np.round(ratio)
                    print(np.round(ratio),'th Harmonic. f1x =',f_1x)
                else: # sino miro el 5th armonico
                    i_f_1x_LOW                    = int(np.round(5*f_1x_LOW*l/fs))
                    i_f_1x_HIGH                   = int(np.round(5*f_1x_HIGH*l/fs))
                    indexes_f_1x, properties_f_1x = find_peaks(sptrm_C[i_f_1x_LOW:i_f_1x_HIGH],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
                    if np.size(indexes_f_1x) >0 :
                        posicion                      = np.argmax( sptrm_C[i_f_1x_LOW+ indexes_f_1x] )
                        f_1x                          = f[i_f_1x_LOW+ indexes_f_1x[posicion]] /5
                        print('5th harmonic, f1x=',f_1x)
                    else: #---------si no existe el 5th armonico busco la fundamentla directamete
                        indexes_f_1x, properties_f_1x = find_peaks(sptrm_C[i_f_1x_low:i_f_1x_high],height= 0,prominence = 0.03,width=1,rel_height = 0.75)
                        posicion                      = np.argmax( sptrm_C[i_f_1x_LOW+ indexes_f_1x] )
                        f_1x                          = f[i_f_1x_LOW+ indexes_f_1x[posicion]]
                        print('directamente f1x=',f_1x)
                        
#                indexes_f_1x, properties_f_1x        = find_peaks(sptrm_C[5*i_f_1x_low:5*i_f_1x_high],height  = 0 ,prominence = 0.03 , width=1 , rel_height = 0.75)
#                f_1x_exist                           = False
#                max_value_f_1x                       = max_value_f_1x_i
#                            # miramos si hay algun pico entre 48 y 51Hz y nos quedamos con el mas grande
#                            # y si lo encontramos entonces "f_1x_exist= True"
#                for k in indexes_f_1x:    
#                    if 5*f_1x_LOW < f[5*i_f_1x_low + k] < 5*f_1x_HIGH and sptrm_C[5*i_f_1x_low + k] > max_value_f_1x:
#                        f_1x           = f[5*i_f_1x_low + k]
#                        f_1x           = f_1x  /5         
#                        max_value_f_1x = sptrm_C[i_f_1x_low + k]
#                        f_1x_exist     = True
                    
#            if f_1x_exist :  
            print('-----------------------------------',f_1x)
            if f_1x !=0:
                indexes, properties = find_peaks(sptrm_C[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
                delta               = 0.75 #-----------en Hz
                fa                  = 0
                fb                  = 0
                for counter,k in enumerate(indexes) :
                    for h in fingerprint_list:
                        if h.order ==1:
                            fa = 0
                            fb = 0
                            if h.f_norm == 'absoluto':   #-------------------------
                                if h.tipo == 'Peak':
                                    fa = (h.f1 - delta) / f_1x
                                    fb = (h.f1 + delta) / f_1x
                                if h.tipo == 'Span':
                                    fa = h.f1 / f_1x
                                    fb = h.f2 / f_1x                                
                            if h.f_norm == 'relativo':   #-------------------------
                                if h.tipo == 'Peak':
                                    fa = h.f1 - delta / f_1x
                                    fb = h.f1 + delta / f_1x
                                if h.tipo == 'Span':
                                    fa = h.f1 
                                    fb = h.f2                                   
                            if h.f_norm == 'mixto':      #-------------------------
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
                                    df_harm.iloc[medida]['S/N '   + word] = 10*np.log10(piko**2/(df_harm.iloc[medida]['RMS (mm/s) f']**2-piko**2))
                

    print('-------------------------------Fingerprints extracted---------------')
    print()

    return df_harm

#==============================================================================

def find_f1x_bis(sptrm,indexes, properties ,fnom,machine_type):
    if machine_type == 'blower':
        HarmonicList = [1,2,3]

    if machine_type == 'pump':
        HarmonicList = [1,5]
    
    df_harmonics= pd.DataFrame(np.nan,index = ['i-','Value','Hz','f1x'],columns = HarmonicList)
   
    Hz = f_in(1)
#    N_max_positions  = Nmaxelements(sptrm[indexes], 3)
#    found            = False
#    
#    for counter,k in enumerate(N_max_positions):      #---lo busco entre los 3 mayores
#        dist = Distan_Armonico(indexes[k],fnom)
#        if dist < 0.01:                               #----se trata de un armonico
#            if np.round( f(indexes[k]) / fnom ) == 5: #--quinto armocnico
#                f_found          = f(indexes[k])/np.round( f(indexes[k]) / fnom )
#                found = True
#                print ('El quinto armonico encontrado',f_found)
    found = False
    if not found:                               #---lo busco entre los 10 primeros armonicos
        highest_value = 0
        for i in  HarmonicList:
        
#            print(i*fnom,'===>',f_in(i*fnom)-Hz,f_in(i*fnom),f_in(i*fnom)+Hz)
            a = indexes[indexes >= f_in(i*fnom)-Hz]
            b = a [a <= f_in(i*fnom) + Hz ] 
            print('armonico',i,'valores de b',b,f(b))
            print( b[np.argmax(sptrm[b])])
            df_harmonics.loc['i-',i]        =  b[np.argmax(sptrm[b])]
            df_harmonics.loc['Value',i]     = sptrm[ b[np.argmax(sptrm[b])]]
            df_harmonics.loc['Hz',i]        = f( b[np.argmax(sptrm[b])])
            df_harmonics.loc['f1x',i]       = f(b[np.argmax(sptrm[b])]) /np.round( f(b[np.argmax(sptrm[b])]) / fnom )
            df_harmonics.loc['distancia',i] = Distan_Armonico(b[np.argmax(sptrm[b])],fnom)
            
            if np.size(b) == 1:
                dist = Distan_Armonico(b,fnom)
                if dist < 0.01:
                    
                    found = True
                    if sptrm[b] > highest_value:
                        highest_value = sptrm[b]
                        i_highest     = b
        if found:
            f_found       = f(i_highest) /np.round( f(i_highest) / fnom )       
            print('El',np.round( f(i_highest) / fnom) ,'th armonico el el mayor', f_found)
        else:
            f_found = 0
            print('Fundamental f1x not found')
    print(df_harmonics)
#        print('indices',indexes[f_in(i*fnom)-Hz:f_in(i*fnom)+Hz])
#        print('valores',sptrm[indexes[f_in(i*fnom)-Hz:f_in(i*fnom)+Hz]])
                        
    return f_found,found
#------------------------------------------------------------------------------
def find_f1x(sptrm,indexes, properties ,fnom,machine_type):
    if machine_type == 'blower':
        HarmonicList = [1,2,3]
    if machine_type == 'pump':
        HarmonicList = [1,5,10]
#    Hz    = f_in(.95)
    found = False
#    print(f(indexes))
                              #---lo busco entre los 10 primeros armonicos
    highest_value = 0
#    df_harmonics= pd.DataFrame(np.nan,index = HarmonicList ,columns = range(4))
    for i in  HarmonicList:   
        
        delta_Hz = f_in(.95*i)
#            print(i*fnom,'===>',f_in(i*fnom)-Hz,f_in(i*fnom),f_in(i*fnom)+Hz)
        a = indexes[indexes >= f_in(i*fnom)-delta_Hz]
        b = a [a <= f_in(i*fnom)+delta_Hz]
#        df_harmonics.loc[i][0:np.size(b)] = b
#        print (i,b,'=>',f(b),sptrm[b],np.size(b))
        if np.size(b) > 0:
            max_value = np.max(sptrm[b])
            indx      = np.argmax(sptrm[b])
            b_sel     = b[indx]
            if max_value > highest_value:
                highest_value = max_value
                i_highest     = b_sel
            
    if highest_value > 0:        
        if sptrm[i_highest-1] > sptrm[i_highest+1]:
            f1 = f(i_highest-1)
            f2 = f(i_highest)
            x1 = sptrm[i_highest-1]
            x2 = sptrm[i_highest]
        else:
            f1 = f(i_highest)
            f2 = f(i_highest+1)
            x1 = sptrm[i_highest]
            x2 = sptrm[i_highest+1]
            
        f_found = x1*f1/(x1+x2)  + x2*f2/(x1+x2)
        f_found = f(i_highest) /np.round( f(i_highest) / fnom )       
        print('El',np.round( f(i_highest) / fnom) ,'th armonico el el mayor', f_found)
        found   = True
    else:
        f_found = 0
        print('Fundamental f1x not found',b)
#    print (df_harmonics)
#        print('indices',indexes[f_in(i*fnom)-Hz:f_in(i*fnom)+Hz])
#        print('valores',sptrm[indexes[f_in(i*fnom)-Hz:f_in(i*fnom)+Hz]])
    print()                    
    return f_found,found

#------------------------------------------------------------------------------
def f(x):
    return x*fs/(l-0)
#------------------------------------------------------------------------------
def f_in(x):
    return int(np.round(x*(l-0)/fs))
#------------------------------------------------------------------------------
def Distan_Armonico(i_f_in,fnom):
    f_in = f(i_f_in)
    ratio    = f_in / fnom
    d = np.abs(np.round(ratio)-ratio)
    return d
#------------------------------------------------------------------------------
def df_Maxvalue(df_in):
    a       = df_in.max(axis=0)
    b       = df_in.idxmax(axis=0)
    value   = np.max(a)
    columna = a.index[np.argmax(a.values)]
    fila    = b.values[np.argmax(a.values)]
    df_out  = df_in.drop( [columna],axis= 1 )
    df_out  = df_out.drop( [fila])
    return value,fila,columna,df_out

#------------------------------------------------------------------------------
def Df_Dismount(df_in):
    orden = np.min(df_in.shape)
    max_list = []
    row_list = []
    col_list = []
    #print(orden)
    for k in range (orden):
        maximo,fila,columna,df_in = df_Maxvalue(df_in)
        #print(df_in)
        max_list.append(maximo)
        row_list.append(fila)
        col_list.append(columna)
    return max_list,row_list,col_list
#------------------------------------------------------------------------------
def decision_taking(fila,ratio_fila,columna,ratio_col):
    value_col  = columna/ratio_col
    value_fila = fila/ratio_fila
    if 0 < np.abs(value_fila-value_col) < 1:
        #print('hay singularidad','value_fila',value_fila,'value_col',value_col)
        value_col = np.round((value_fila + value_col)/2)
   
    return int(value_col) #----posible freq fundamental
#------------------------------------------------------------------------------
def Accurate_freq(n,sptrm):
    if sptrm[n-1] > sptrm[n+1]:
        f1 = f(n-1)
        f2 = f(n)
        x1 = sptrm[n-1]
        x2 = sptrm[n]
    else:
        f1 = f(n)
        f2 = f(n+1)
        x1 = sptrm[n]
        x2 = sptrm[n+1]
        
    f_accu = x1*f1/(x1+x2)  + x2*f2/(x1+x2)
    return f_accu
#------------------------------------------------------------------------------
def Remove_duplicate(duplicate): 
    final_list = np.array([]) 
    for num in duplicate:
        if (num not in final_list) and (np.isnan(num) == False): 
            final_list= np.append(final_list,num) 
    return final_list 
#------------------------------------------------------------------------------
def find_f1x_robust(sptrm,RMS,indexes, properties ,fnom):
#    print('--------------Buscando frecuencia fundamental---------------')
#    print('indexex',indexes)
    HarmonicList = [1,2,3,4,5,6,7,8,9,10] #---lo busco entre los 10 primeros armonicos
    Hz            = f_in(2)
    found         = False
    freq          = np.arange(l)/l*fs
    lista         = []                                           #------- akmeceno las combinaciones de indexex de peaks
    for i in  HarmonicList:   
#            print(i*fnom,'===>',f_in(i*fnom)-Hz,f_in(i*fnom),f_in(i*fnom)+Hz)
        a = indexes[indexes >= f_in(i*fnom) - f_in(i*1)]
        b = a [a <= f_in(i*fnom) + f_in(i*1)]
        lista.append(b)
#        print (b)
    #print(lista)
    listab = []                                                  #--- almaceno los numeros naturales de las combinaciones
    for counter,item in enumerate(combinations(HarmonicList,2)):
        listab.append(item)
     
    #print(listab)
#    potential         = np.array([])
    df_Num_Ocurrences = pd.DataFrame(index=['Hits'])             # contador de multiplicidades
    df_Ocurrences     = pd.DataFrame()                           # indice de multiplicidades
    df_RMS            = pd.DataFrame()
    for counter,item in enumerate(combinations(lista,2)):
#        print (counter,item)
#        print('i----',listab[counter],'----k')
        df = pd.DataFrame(np.zeros ((np.size(item[0]),np.size(item[1]))),index = item[0],columns = item[1] )
        
        for i in df.index:                #-------------valores pequeños = filas
            for k in df.columns:          #-------------valores grandes  = columnas
                parecido     = 1- (np.abs((k/i) - (listab[counter][1]/listab[counter][0]))/(listab[counter][1]/listab[counter][0]))
#                parecido     = 1- (np.abs(( Accurate_freq(k,sptrm) / Accurate_freq(i,sptrm) ) - (listab[counter][1]/listab[counter][0]))/(listab[counter][1]/listab[counter][0]))              
                df.loc[i][k] = parecido
#        print('--Hemos construido df')  
#        print(df  )  
        valores,filas,columnas = Df_Dismount(df)                                     # extraemos los máximos de casa cruce fila/columna
#        print('--------------------------',valores)
        for counter2,k in enumerate(columnas):            
            parecido = valores[counter2]
#            elemento = np.round(columnas[counter2]/listab[counter][1]) # columna/orden de combinatoria. elemento = f1x Potencial
            elemento = decision_taking(filas[counter2]    , listab[counter][0],
                                       columnas[counter2] , listab[counter][1])
#            print(filas[counter2],columnas[counter2],'parecido',parecido)
#            print('elemento selecionado',elemento)
            
            if parecido > 0.99 : #----------tenemos un multiplo              1. Incrementamos df_Num_Ocurrences
                                 #                                           2. Intorducimos k y i en la lista de df_Ocurrences
#                print ('se mete',int(filas[counter2]),int(columnas[counter2]),'en =>',elemento)
                indx_fila = np.where(indexes ==    filas[counter2])[0][0] 
                indx_col  = np.where(indexes == columnas[counter2])[0][0] 
                
                init_fila = int(np.round(properties["left_ips"][indx_fila]))                                                                     
                end_fila  = int(np.round(properties["right_ips"][indx_fila]))  
                piko_fila = np.sqrt(np.sum( sptrm[init_fila : end_fila+1]**2 )) 
                
                init_col  = int(np.round(properties["left_ips"][indx_col]))                                                                     
                end_col   = int(np.round(properties["right_ips"][indx_col]))  
                piko_col  = np.sqrt(np.sum( sptrm[init_col : end_col+1]**2 ))   

#                print(filas[counter2], 'en ',np.where(df_Num_Ocurrences.columns.values ==    filas[counter2])[0], 'de ',df_Num_Ocurrences.columns.values)                              
                condition = (elemento in df_Num_Ocurrences.columns.values)             #----elemento (potencial valor de f1x) esta ya intorducido??
#                print(elemento,df_Num_Ocurrences.columns.values,condition)
                if condition == False:# and  (  (int(filas[counter2]) in df_Ocurrences.columns.values)  == False) :                                     # NO => creo una nueva columna en df_potencial y df_potencial_bis 
#                    print('añadimos una columna')
                    df_Num_Ocurrences[elemento]          = 1                           #  nueva columna en df_Num_Ocurrences
                    Nfilas                               = 0
                    df_Ocurrences.loc[Nfilas,elemento]   = int(filas[counter2])        #--añado i en df_Ocurrences
                    df_Ocurrences.loc[Nfilas+1,elemento] = int(columnas[counter2])     #--añado k en df_Ocurrences
                
                    df_RMS.loc[Nfilas,elemento]          = piko_fila
                    df_RMS.loc[Nfilas+1,elemento]        = piko_col  

                if condition == True:                                                  #--Si ya esta introducido
                    df_Num_Ocurrences.loc['Hits',elemento] = df_Num_Ocurrences.loc['Hits',elemento] + 1 #incremento un valor 
                    
                    indice = np.argwhere (df_Ocurrences.columns.values == elemento )[0][0]              # indice para meter valores en df_potencial_bis
#                    print('contadores',df_Ocurrences.count(axis=0).values)
#                    print(indice,'=>',df_Ocurrences.count(axis=0).values[indice] )
                    indi = df_Ocurrences.count(axis=0).values[indice]
                    
                    cond1 = filas[counter2] in df_Ocurrences[elemento].values                           #--esta i en df_Ocurrences
                    if cond1 == False:
                        df_Ocurrences.loc[indi,elemento] = int(filas[counter2])
                        df_RMS.loc       [indi,elemento] = piko_fila
                        
                    cond2 = columnas[counter2] in df_Ocurrences[elemento].values                        #--esta k en df_Ocurrences
                    if cond2 == False:
                        df_Ocurrences.loc[indi,elemento] = int(columnas[counter2])
                        df_RMS.loc       [indi,elemento] = piko_col
    
#    print(df_RMS)
#    print(df_Num_Ocurrences)
    elemento_power = int(df_RMS.sum().index       [np.argmax(df_RMS.sum().values)                 ])    #nombre de la columna con mas POWER
    elemento_hits  = int(df_Num_Ocurrences.columns[np.argmax(df_Num_Ocurrences.loc['Hits'].values)])    #nombre de la columna con mas HITS
#    hits_hits      = df_Num_Ocurrences[elemento_hits].values                                            # HITS del elemento con mas HITS
#    hits_power     = df_Num_Ocurrences[elemento_power].values                                           # HITS del elemento con mas POWER

    if abs(elemento_power - elemento_hits) == 0 or abs(elemento_power - elemento_hits) == 1:
        probability                   = 0
        multiplos                     = np.sort(np.append(df_Ocurrences[elemento_hits].values , df_Ocurrences[elemento_power].values))
        multiplos                     = Remove_duplicate(multiplos)
        df_ListHarmonics              = pd.DataFrame(index= HarmonicList, columns = ['Sample','Distancia','RMS'])
        df_ListHarmonics['Distancia'] = np.ones(10)
        df_ListHarmonics['RMS'] = np.zeros(10)
        for counter in HarmonicList:
            found_harm = False
            for k in multiplos:
                multp     = k/ elemento_hits
                distancia = abs(multp-counter)
                if distancia < 0.05 and distancia < df_ListHarmonics.loc[counter]['Distancia']:
                    df_ListHarmonics.loc[counter,'Sample']    = k
                    df_ListHarmonics.loc[counter,'Distancia'] = distancia    
                    indx_k                                    = np.where(indexes == k)[0][0] 
                    init_k                                    = int(np.round(properties["left_ips"][indx_k]))                                                                     
                    end_k                                     = int(np.round(properties["right_ips"][indx_k]))  
                    piko_k                                    = np.sqrt(np.sum( sptrm[init_k : end_k+1]**2 ))
                    df_ListHarmonics.loc[counter,'RMS']       = piko_k
                    found_harm                                = True
                    
            if found_harm == False:
                df_ListHarmonics.loc[counter,'Sample']    = 'NoF'
                df_ListHarmonics.loc[counter,'Distancia'] = 'NoF'
            else:
                probability                               = probability +1
        
#        print('---------------------------------------df_ListHarmonics')
#        print(df_ListHarmonics['RMS'].values,np.max(df_ListHarmonics['RMS'].values))
        Highest_Harmonic   = int(np.argmax(df_ListHarmonics['RMS'].values)+1)
        i_Highest_harmonic = int(df_ListHarmonics.loc[Highest_Harmonic]['Sample'])
        found              = True
       
#        if sptrm[i_Highest_harmonic-1] > sptrm[i_Highest_harmonic+1]:
#            f1 = f(i_Highest_harmonic-1)
#            f2 = f(i_Highest_harmonic)
#            x1 = sptrm[i_Highest_harmonic-1]
#            x2 = sptrm[i_Highest_harmonic]
#        else:
#            f1 = f(i_Highest_harmonic)
#            f2 = f(i_Highest_harmonic+1)
#            x1 = sptrm[i_Highest_harmonic]
#            x2 = sptrm[i_Highest_harmonic+1]
#            
#        f_found = x1*f1/(x1+x2)  + x2*f2/(x1+x2)
        f_found = Accurate_freq(i_Highest_harmonic,sptrm)
        f_found = f_found / Highest_Harmonic
        print('F. Freq.=',format(f_found,'.03f' ),
              'at Harm N=',Highest_Harmonic,'=>',i_Highest_harmonic,'sample =>',
              format(df_ListHarmonics.loc[Highest_Harmonic]['RMS'],'.02f' ),'|',format(RMS,'.02f'),'RMS',
              '| [',probability,'] Harm. identified')
    else:        
        found   = False
        f_found = 0
        print()
        print('Fundamental frequency ----------NOT FOUND!!!!!!!!!!!!!!')

#    print()
#    print('-------------------------------------------df_Num_Ocurrences')        
#    print(df_Num_Ocurrences)
#    print('---------------------------------------df_Ocurrences')
#    print(df_Ocurrences)
#    print('---------------------------------------df_RMS')
#    print(df_RMS)
#    print(df_RMS.sum().values)
#    print('elemento con mayor POWER',elemento_power, 'with',df_RMS.pow(2).sum().pow(0.5)[elemento_power],'and',hits_power,'Hits')  
#    print('elemento con mas HITS   ',elemento_hits ,'with', df_RMS.pow(2).sum().pow(0.5)[elemento_hits],'and',hits_hits ,'Hits')
#    print('====================================================================') 

       
    return  f_found,found

#------------------ESTO ES BASURILLA SE PODRA BORRAR---------------------------
def df_shortcut(df_FFT,fs,machine_type):   
    print('------------------------Extracting fingerprint----------------------')
    l        = df_FFT.shape[1]
    n_traces = df_FFT.shape[0]    
    if machine_type == 'blower':
        print('----------------------------BLOWER----------------------')
        fingerprint_list = fprnt_list_blwrs
        fnom             = 24.9
    if machine_type == 'pump':
        print('--------------------------------------------------')
        fingerprint_list = fprnt_list_pumps
        fnom             = 48.75        
    fecha    = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))
    columnas = []
   
    columnas    = ['RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    delta       = 0.75 #-----------en Hz
    for medida in range(n_traces):
#        print ('Medida numero-------------------------------------------------',medida,df_harm.index[medida])
        f_1x                                 = 0
        abs_line                             = np.abs(df_FFT.iloc[medida].values)
        RMS_freq                             = np.sqrt(np.sum( abs_line**2 ) )
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        if  RMS_freq > 0.06: #-------------------------esta la máquina apagada?
            sptrm_C             = abs_line * np.sqrt(2)  # integramos en 1º z Nyquist por eso el voltaje x raiz(2) 
            indexes, properties = find_peaks(sptrm_C[0:l_mitad],height  = 1*0.01 ,prominence = 0.01 , width=1 , rel_height = 0.75)
            print(medida,fnom)
            xx,yy               = find_f1x_robust(sptrm_C,RMS_freq,indexes, properties ,fnom)
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',xx,yy)
    return xx
#------------------------------------------------------------------------------
def df_Harmonics(df_FFT,fs,machine_type):
    print('------------------------Extracting fingerprint----------------------')
    l        = df_FFT.shape[1]
    n_traces = df_FFT.shape[0]    
    if machine_type == 'blower':
        print('----------------------------BLOWER----------------------')
        fingerprint_list = fprnt_list_blwrs
        fnom             = 24.9
    if machine_type == 'pump':
        print('--------------------------------------------------')
        fingerprint_list = fprnt_list_pumps
        fnom             = 48.75        
    fecha    = []
    for k in df_FFT.index:
        #print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))
    columnas = []
    word     = ''
    for k in fingerprint_list:
        if k.order == 1:
            word = k.label
            columnas.append('i '    + word)
            columnas.append('Point '+ word)
            columnas.append('RMS '  + word)
            columnas.append('f '    + word)
            columnas.append('BW '   + word)
            columnas.append('n_s '  + word)
            columnas.append('n_e '  + word)
            columnas.append('S/N '  + word)
        if k.order > 1:
            word = k.label
            for i in np.arange(k.order)+1:
                word_bis = str(i)+'th '
                columnas.append('i '    + word_bis + word)
                columnas.append('Point '+ word_bis + word)
                columnas.append('RMS '  + word_bis + word)
                columnas.append('f '    + word_bis + word)
                columnas.append('BW '   + word_bis + word)
                columnas.append('n_s '  + word_bis + word)
                columnas.append('n_e '  + word_bis + word)
                columnas.append('S/N '  + word_bis + word)    
    columnas    = ['RMS (mm/s) f'] + columnas
    df_harm     = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))
    l_mitad     = int(l/2)
    f           = np.arange(l)/l*fs
    delta       = 0.75 #-----------en Hz
    for medida in range(n_traces):
#        print ('Medida numero-------------------------------------------------',medida,df_harm.index[medida])
        f_1x                                 = 0
        abs_line                             = np.abs(df_FFT.iloc[medida].values)
        RMS_freq                             = np.sqrt(np.sum( abs_line**2 ) )
        df_harm.iloc[medida]['RMS (mm/s) f'] = RMS_freq
        if  RMS_freq > 0.06: #-------------------------esta la máquina apagada?
            sptrm_C             = abs_line * np.sqrt(2)  # integramos en 1º z Nyquist por eso el voltaje x raiz(2) 
            indexes, properties = find_peaks(sptrm_C[0:l_mitad],height  = 0*0.01 ,prominence = 0.01 , width=1 , rel_height = 0.75)
            f_1x , f_1x_exist   = find_f1x(sptrm_C,indexes, properties ,fnom,machine_type)
#            f_1x , f_1x_exist   = find_f1x_robust(sptrm_C,RMS_freq,indexes, properties ,fnom)
            if f_1x_exist:   #----------hemos identificado f1x-----------------
                for h in fingerprint_list:        
                    if  h.order >1:
                        word = h.label
                        if h.f_norm == 'absoluto':   #-------------------------
                            fa = h.f1
                            fb = h.f2
                        i_fa            = int(np.round(fa*l/fs))
                        i_fb            = int(np.round(fb*l/fs))                   
                        upper           = indexes[indexes >= i_fa]
                        inner           = upper  [upper   <= i_fb] 
                        N_max_positions = Nmaxelements(sptrm_C[inner], h.order)                    
                        for counter,position in enumerate(N_max_positions):
                            word                                            = h.label
                            word_bis                                        = str(counter+1)+'th '            
                            init                                            = int(np.round(properties["left_ips"][position]))                                                                     
                            end                                             = int(np.round(properties["right_ips"][position]))  
                            piko                                            = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) 
                            df_harm.iloc[medida]['i '    + word_bis + word] = int(indexes[position])
                            df_harm.iloc[medida]['Point '+ word_bis + word] = sptrm_C[int(indexes[position])]
                            df_harm.iloc[medida]['RMS '  + word_bis + word] = piko #/ Max_value
                            df_harm.iloc[medida]['f '    + word_bis + word] = f[int(indexes[position])]
                            df_harm.iloc[medida]['BW '   + word_bis + word] = f[int(properties["widths"][position]) ]
                            df_harm.iloc[medida]['n_s '  + word_bis + word] = init
                            df_harm.iloc[medida]['n_e '  + word_bis + word] = end            
                for counter,k in enumerate(indexes) :
                    for h in fingerprint_list:
                        if h.order ==1:
                            if h.f_norm == 'absoluto':   #-------------------------
                                if h.tipo == 'Peak':
                                    fa = (h.f1 - delta) / f_1x
                                    fb = (h.f1 + delta) / f_1x
                                if h.tipo == 'Span':
                                    fa = h.f1 / f_1x
                                    fb = h.f2 / f_1x                            
                            if h.f_norm == 'relativo':   #-------------------------
                                if h.tipo == 'Peak':
                                    fa = h.f1 - delta / f_1x
                                    fb = h.f1 + delta / f_1x
                                if h.tipo == 'Span':
                                    fa = h.f1 
                                    fb = h.f2                               
                            if h.f_norm == 'mixto':      #-------------------------
                                if h.tipo == 'Peak':
                                    fa = (h.f1 + h.f2*f_1x - delta) / f_1x
                                    fb = (h.f1 + h.f2*f_1x + delta) / f_1x  
#                            print(h.label,fa ,f[k]/f_1x, fb,sptrm_C[k])
                            if fa  <= f[k]/f_1x <= fb:
                                word = h.label
                                init = int(np.round(properties["left_ips" ][counter]))
                                end  = int(np.round(properties["right_ips"][counter]))
                                piko = np.sqrt(np.sum( sptrm_C[init : end+1]**2 )) #--estos valores son RMS mm/s
                                if piko > df_harm.iloc[medida]['RMS '+word]:
                                    df_harm.iloc[medida]['i '     + word] = k
                                    df_harm.iloc[medida]['Point ' + word] = sptrm_C[k]
                                    df_harm.iloc[medida]['RMS '   + word] = piko #/ Max_value
                                    df_harm.iloc[medida]['f '     + word] = f[k]
                                    df_harm.iloc[medida]['BW '    + word] = f[int(properties["widths"][counter]) ]
                                    df_harm.iloc[medida]['n_s '   + word] = init
                                    df_harm.iloc[medida]['n_e '   + word] = end
                                    #df_harm.iloc[medida]['S/N '   + word] = 10*np.log10(piko**2/(df_harm.iloc[medida]['RMS (mm/s) f']**2-piko**2))           
                                
    print('-----------------------Fingerprints extracted-----------------------')
    return df_harm

#------------------------------------------------------------------------------
 
def check_test_results(harm):
    for columna in harm.columns:
        if columna[0]=='$':
            print(columna,'==>',harm.index[harm[str(columna)] == 'None'])
            print          
#------------------------------------------------------------------------------
#==============================================================================


def plot_waterfall_poly(df_in,harm,fs,fmin,fmax):
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

def plot_waterfall_lines(title,df_in,df_harm,fs,fmin,fmax):

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

        curva            = 2 * np.abs(df_in.iloc[counter].values)/1.63 *np.sqrt(2)
        curva_p          = (curva[n_fmin:n_fmax])
        inic_day         = datetime.datetime.fromtimestamp(df_in.index[counter]).day
        #print(t_traces[counter])
        #ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[np.mod(color_counter,7)], linewidth = 0.2)
        #ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[datetime.datetime.fromtimestamp(df_in.index[counter]).weekday()], linewidth = 0.2)
        ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color='b', linewidth = 0.2)

    ax.view_init(40, 90)
    ax.set_xlabel('Hertz')
    ax.set_xlim3d(fmin, fmax)
    ax.set_ylabel('days')
    ax.set_ylim3d(0, t_traces[np.size(t_traces)-1])
    ax.set_zlabel('RMS mm/s')
    #ax.set_zlim3d(np.min(color), np.max(color))
    #ax.set_title(Parameters['IdAsset']+' '+Parameters['Localizacion']+' mm/sg RMS')
    ax.set_title(title)
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
                if k == '2th i 2nd Max Value.':
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

    
    SPTRM_P             = 2 * np.abs(df_VELOCITY.iloc[captura].values * np.sqrt(2)) /1.63

    indexes, properties = find_peaks( SPTRM_P[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)

    minorLocator = AutoMinorLocator()
    
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    #plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    #ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)

    fig, ax1 = plt.subplots(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(f[0:l_mitad] , SPTRM_P[0:l_mitad],'b')
    #plt.plot(f[indexes]   , SPTRM_P[indexes]  ,'o')

    plotted =[]
    for counter,k in enumerate(df_harm.columns):
        
        if k[0] == 'i':
            #print (counter,k)
            index = int(df_harm.iloc[captura][counter])
            if f[index] !=0:
                offset = 0
                #plt.plot(f[index] , df_harm.iloc[captura][str(df_harm.columns[counter+2])]  ,'o')
                #plt.vlines(x=f[index], ymin=SPTRM_P[index] - properties["prominences"],ymax = df_harm.iloc[captura][str(df_harm.columns[counter+2])], color = "C1")
                plt.plot(f[index] , SPTRM_P[index]  ,'o')               
                offset = 0.02*plotted.count(index)
                plotted.append(index)
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
def Plot_Spectrum_log(captura,df_VELOCITY,df_harm):
    
    waveform = df_VELOCITY.iloc[captura].values
    
    date_exact = datetime.datetime.fromtimestamp(df_VELOCITY.index[captura])

    l          = np.size(waveform)
    l_mitad    = int(l/2)
    f          = np.arange(l)/(l-1)*fs
    #----------quitamos los 2 HZ del principio

    
    SPTRM_P             = 2 * np.abs(df_VELOCITY.iloc[captura].values * np.sqrt(2)) /1.63

    indexes, properties = find_peaks( SPTRM_P[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)

    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    #plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    #ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)

    fig, ax1 = plt.subplots(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(f[0:l_mitad] , np.log10(SPTRM_P[0:l_mitad]),'b')
    y         = signal.savgol_filter(SPTRM_P, 51, 1,mode='interp')
    plt.plot(f[0:l_mitad] , np.log10(y[0:l_mitad]),'r')
    
    #plt.plot(f[indexes]   , SPTRM_P[indexes]  ,'o')

    for counter,k in enumerate(df_harm.columns):

        if k[0] == 'i':
            #print (counter,k)
            index = int(df_harm.iloc[captura][counter])
            if f[index] !=0:
                offset = 0
                #plt.plot(f[index] , df_harm.iloc[captura][str(df_harm.columns[counter+2])]  ,'o')
                #plt.vlines(x=f[index], ymin=SPTRM_P[index] - properties["prominences"],ymax = df_harm.iloc[captura][str(df_harm.columns[counter+2])], color = "C1")
                plt.plot(f[index] , np.log10(SPTRM_P[index])  ,'o')
                if k == 'i 2th Max Value.':
                    offset = 0.1
                #plt.text(f[index] , SPTRM_P[index],str(k))
                #plt.text(f[index] ,df_harm.iloc[captura][str(df_harm.columns[counter+2])]+offset,str(df_harm.columns[counter+2])+' '+str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+2])],'.02f'  ) ) )
                plt.text(f[index] ,np.log10(SPTRM_P[index])+offset,str(df_harm.columns[counter+2])+' '+str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+2])],'.02f'  ) ) )
                #print( str( format ( df_harm.iloc[captura][str(df_harm.columns[counter+1])],'.02f'  ) )    )

    plt.ylabel('mm/s RMS')
    plt.title('Date: '+str(date_exact)+'           RMS: '+str(df_harm.iloc[captura]['RMS (mm/s) f'])+'mm/s' )
    plt.xlabel('Hz')
    plt.grid(True)

    plt.vlines(x=f[indexes], ymin=np.log10(SPTRM_P[indexes] - properties["prominences"]),ymax = np.log10(SPTRM_P[indexes]), color = "C1")
    plt.hlines(y=np.log10(properties["width_heights"]), xmin=f[np.round(properties["left_ips"]).astype(int)],xmax=f[np.round(properties["right_ips"]).astype(int)], color = "C1")

    ax1.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    
    return

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