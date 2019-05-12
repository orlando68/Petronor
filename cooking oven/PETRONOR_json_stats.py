# -*- coding: utf-8 -*-
"""
Editor de Spyder


"""
import datetime, time
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
from PETRONOR_lyb import load_json_file
from PETRONOR_lyb import PETROspectro
from PETRONOR_lyb import spectro
from PETRONOR_lyb import clearance
from matplotlib.colors import colorConverter

from detect_peaks import detect_peaks
from scipy.signal import find_peaks

import pandas as pd
import os
from pandas import DataFrame

import xlwt 
from   xlwt import Workbook 


from scipy import stats

#------------------------------------------------------------------------------
def load_vibrationData(rootdir, assetId,MeasurePointId):
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
            word = str(res.AssetId.values[0])+' '+str(res.MeasurePointId.values[0])
            if (word in lista_maquinas) == False:
                lista_maquinas.append( word )
                                                         #---------------------            
            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                #print(root)
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                data.append(np.asarray(res.Value.values[0])*cal_factor)
                #data.append(res.Value.values[0])
                #print(res.MeasurePointId.values[0],res.MeasurePointName.values[0] )
                fecha = res.ServerTimeStamp.values[0]
                #print(fecha,np.float(fecha[19:len(fecha)-1]))
                datetime_obj = datetime.datetime.strptime(fecha[0:19],format)
                segundos = time.mktime(datetime_obj.timetuple()) + np.float(fecha[19:len(fecha)-1]) 
                        #----------------------------tiempo exacto => segundos
                        #---almacenamos segundos----segundos => datetime EXACTO
                        #-------------------------datetime => segundos INEXACTO
                print(datetime.datetime.fromtimestamp(segundos),MeasurePointId,'N. puntos :', np.size(np.asarray(res.Value.values[0])) )
                date.append(segundos)
                if nfiles == 6: #----files per day per day
                    break 
                nfiles = nfiles +1

        counter = counter +1
    df_out     = DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out, lista_maquinas


#------------------------------------------------------------------------------
    
# Python program to find largest, smallest,  
# second largest and second smallest in a 
# list with complexity O(n) 
def Range(list1): 
    largest = list1[0] 
    largest2 = list1[0] 
    lowest = list1[0] 
    lowest2 = list1[0] 
    counter = 0
    i_largest = i_largest2 = 0
    for item in list1:        
        if item > largest:  
            largest = item
            i_largest = counter
        elif largest2!=largest and largest2 < item: 
                largest2 = item
                i_largest2 = counter
        elif item < lowest: 
            lowest = item 
            i_lowest = counter
        elif lowest2 != lowest and lowest2 > item: 
            lowest2 = item 
            i_lowest2 = counter
        
        counter = counter +1
    print("Largest element is        :", largest) 
    #print("Smallest element is       :", lowest) 
    print("Second Largest element is :", largest2) 
    #print("Second Smallest element is:", lowest2) 
#    return i_largest,largest,i_largest2,largest2,i_lowest,lowest,i_lowest2,lowest2
    return i_largest,i_largest2
#------------------------------------------------------------------------------
def DataFrame_filt(df_in,b_filt,a_filt):
    l         = df_accel.shape[1]#-500
    n_columns = df_accel.shape[0]
    df_out    = pd.DataFrame(index   = df_in.index,
                             columns = df_in.columns.values,
                             data    = np.ones((n_columns,l)))
    df_out    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):
        trace               = signal.filtfilt(b_filt, a_filt, df_in.loc[indice].values)
        df_out.loc[indice]  = trace[0:l]
    return df_out

#------------------------------------------------------------------------------
def DataFrame_remove_DC(df_in):

    df_out    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):
        #trace               = signal.filtfilt(b_filt, a_filt, df_in.loc[indice].values)
        media               = np.mean(df_in.loc[indice].values)
        df_out.loc[indice]  = df_in.loc[indice].values - media
    return df_out
#------------------------------------------------------------------------------
def DataFrame_integrate(df_in):#,b_filt,a_filt):

    df_out    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):
        #trace                  = signal.filtfilt(b_filt, a_filt, df_in.loc[indice].values)
        trace                  = df_in.iloc[counter].values
        #DC                     = np.mean(df_in.iloc[counter].values)
        #df_out.iloc[counter]   = np.cumsum(df_in.iloc[counter].values-DC)
        df_out.iloc[counter]   = np.cumsum(trace)
        #print (np.cumsum(df_in.iloc[counter]-mean),df_out.iloc[counter]  )
            
    return df_out
#------------------------------------------------------------------------------
def DataFrame_fft_abs(df_in):
    l         = df_in.shape[1]
    hann      = np.hanning(l) #
    df_out    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):        
        trace                 = df_in.iloc[counter].values * hann
        df_out.iloc[counter]  = np.abs( np.fft.fft(trace/l) )
    return df_out

#------------------------------------------------------------------------------
def DataFrame_fft_angle(df_in):
    l         = df_in.shape[1]
    hann      = np.hanning(l)
    df_out    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):        
        trace                 = df_in.iloc[counter].values * hann
        df_out.iloc[counter]  = np.angle( np.fft.fft(trace/l) )
    return df_out
#------------------------------------------------------------------------------
def df_clearance(df_in):
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')        
    df_in['Clearance Failure'] = none_list
    
    for i in range (n_traces):
        v_1x   = df_in.iloc[i]['1.0']
        v_2x   = df_in.iloc[i]['2.0']
        v_3x   = df_in.iloc[i]['3.0']
        
        v_0_5x = df_in.iloc[i]['0.5']
        v_1_5x = df_in.iloc[i]['1.5']
        v_2_5x = df_in.iloc[i]['2.5']
                                            #-------1.0x 2.0x 3.0x decreciente
        bool1 =  v_1x >v_2x > v_3x
                                            # --2.0x >2% 1.0x and 3.0x >2% 1.0x
        bool2 = (v_2x > 0.02 * v_1x) and (v_3x > 0.02 * v_1x) 
                                            #-------0.5x 1.5x 2.5x decreciente        
        bool3 =  v_0_5x > v_1_5x > v_2_5x     
                                            # ------0.5x >2% 1.0x and 1.5x > 2% 1.0x and 2.5x > 2% 1.0x
        bool4 = (v_0_5x > 0.02 * v_1x) and (v_1_5x > 0.02 * v_1x) and (v_2_5x > 0.02 * v_1x)
        
        #print (bool1,bool2,bool3,bool4)
        bool_A = bool1 and bool2
        bool_B = bool3 and bool4
        if (bool_A == False) and (bool_B == False):
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Green'
        
        if (bool_A == True)  or  (bool_B == True):
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Yellow'
            
        if (bool_A == True)  and (bool_B == True):
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Red'
            
    return df_in
#------------------------------------------------------------------------------    
def df_unbalance(df_in):
    
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')        
    df_in['Unbalance Failure'] = none_list
    
    for i in range (n_traces):
        #print(df_in.iloc[i].values[1:8])
                                            # max armonicos = 1.0x
        f_max1 = df_in.iloc[i]['1.0'] 
                                            # max del resto de pikos                                    
        s_max1 = df_in.iloc[i]['2nd Highest']
        
                                            #---1X meno que el umbral
        bool_A  = f_max1        < 4 
                                            #---El 15% 1x < resto armonicos. 
                                            #   es decir 1X no es dominante
        bool_B  = f_max1 * 0.15 < s_max1

                                            #--------------------------Green
        if (bool_A == True ) and (bool_B == True):  
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'
                                            #--------------------------yellow 
                                            #   Xor = cualquiera de ellas
                                            #        pero no ambas
        if (bool_A == False) ^   (bool_B == False): 
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Yellow'
                                            #--------------------------Red   
                                            # las dos falsas
        if (bool_A == False) and (bool_B == False):
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'
            
    return df_in
#------------------------------------------------------------------------------
def oil_whirl(df_in):
    
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')        
    df_in['Oil Whirl Failure'] = none_list
    
    for i in range (n_traces):
                                            #-----------green-----------------
                                            # no detected Peak in '0.38-0.48'        
        if df_in.iloc[i]['Oil Whirl'] < E1:  
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Green'
                                            #-----------yellow-----------------
                                            # Detected Peak in '0.38-0.48'
                                            #         but
                                            # Peak in '0.38-0.48' < 2% 1.0x
        if df_in.iloc[i]['Oil Whirl'] > E1 and df_in.iloc[i]['Oil Whirl'] < 0.02 * df_in.iloc[i]['1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Yellow'
                                            #-----------red--------------------
                                            # Peak in '0.38-0.48' > 2% 1.0x                                            
        if df_in.iloc[i]['Oil Whirl'] > E1 and df_in.iloc[i]['Oil Whirl'] > 0.02 * df_in.iloc[i]['1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Red'
            
    return df_in
    
#------------------------------------------------------------------------------
def oil_whip(df_in):
    
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    
    for i in range (n_traces):
        none_list.append('None')        
    df_in['Oil Whip Failure'] = none_list
    
    for i in range (n_traces):
        A = (df_in.iloc[i]['0.5'] >= E1 and df_in.iloc[i]['BW 0.5'] >= 4)  and  ((df_in.iloc[i]['2.5'] >= E1) and df_in.iloc[i]['BW 2.5'] >= 4)
        B = df_in.iloc[i]['0.5'   ] > 0.02 *  df_in.iloc[i]['1.0']
        C = df_in.iloc[i]['2.5'   ] > 0.02 *  df_in.iloc[i]['1.0']        
                                             #  Tabla de verdad progresiva
                                             #  puede empezar siendo verde,
                                             #  acabar siendo rojo
                                             
                                             #-----------green-----------------
                                             # 2H BW at 0.5 = 0 and 2H BW at 2.5 = 0                                        

        if A == False and ( (B and C) == False ):  
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Green'            
                                             #---------yellow------------------
                                             # 2H BW at 0.5 > 0
                                             # 2H BW at 2.5 > 0
                                             # 2H BW at 0.5 >2% 1.0x
                                             # 2H BW at 2.5 >2% 1.0x
        if A == True or ((B or C) == True) :  
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Yellow'           
                                             #-----------red-------------------    
                                             #     2H BW at 0.5 >2% 1.0x
                                             #           AND
                                             #     2H BW at 2.5 >2% 1.0x
        if A and B and C:  
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Red'    
    return df_in
#------------------------------------------------------------------------------

def CENTRIFUGAL_FAN_AERODYNAMIC_FAILURE(df_in):
    
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')        
    df_in['C Fan Aero. Failure'] = none_list
    
    A = df_in.iloc[i]['1BPF'] > E1
    B = df_in.iloc[i]['1BPF'] > E1 and df_in.iloc[i]['2BPF'] > E1
    C = df_in.iloc[i]['1BPF'] > E1 and (df_in.iloc[i]['1BPF-1.0'] > E1 or df_in.iloc[i]['1BPF+1.0'] > E1)
    D = C and df_in.iloc[i]['2BPF'] > E1
    E = C and df_in.iloc[i]['2BPF'] > E1 and (df_in.iloc[i]['2BPF-1.0'] > E1 or df_in.iloc[i]['2BPF+1.0'] > E1)
    
    for i in range (n_traces):
                                             #  Tabla de verdad progresiva
                                             #  puede empezar siendo verde,
                                             #  acabar siendo rojo
        if A == True or B == True:  
            df_in.loc[df_in.index[i],'C Fan Aero. Failure'] = 'Green'            
        if C == True or D == True:  
            df_in.loc[df_in.index[i],'C Fan Aero. Failure'] = 'Yellow'        
        if E == True:  
            df_in.loc[df_in.index[i],'C Fan Aero. Failure'] = 'Red'
    
    return df_in
#------------------------------------------------------------------------------    
def flow_turbulence(df_in):
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')        
    df_in['Flow Turbulence'] = none_list
    
    A = df_in.iloc[i]['Flow T.'] >= 0.2
    B = 0.2 <= df_in.iloc[i]['Flow T.'] <= df_in.iloc[i]['1.0']
    C =       df_in.iloc[i]['Flow T.'] >  df_in.iloc[i]['1.0']
    
    if A:
        df_in.loc[df_in.index[i],'Flow T.'] = 'Green'
    if B:
        df_in.loc[df_in.index[i],'Flow T.'] = 'Yellow'
    if C:
        df_in.loc[df_in.index[i],'Flow T.'] = 'Red'
    return df_in

#------------------------------------------------------------------------------
def Plain_bearing_block_looseness(df_in):
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')        
    df_in['PBB looseness'] = none_list
    
    A = df_in.iloc[i]['1.0'] > E1 and df_in.iloc[i]['2.0'] > E1 and df_in.iloc[i]['3.0'] > E1 and df_in.iloc[i]['1.0'] <df_in.iloc[i]['2.0'] > df_in.iloc[i]['3.0']
    B = df_in.iloc[i]['0.5'] > E1 and df_in.iloc[i]['1/3'] > E1 and df_in.iloc[i]['1/4'] > E1

    if not A and not B:
        df_in.loc[df_in.index[i],'PBB looseness'] = 'Green'
    if A or B:
        df_in.loc[df_in.index[i],'PBB looseness'] = 'Yellow'
    if A and B:
        df_in.loc[df_in.index[i],'PBB looseness'] = 'Red'
    return df_in
#------------------------------------------------------------------------------
def SHAFT_MISALIGNMENTS(df_in):
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')        
    df_in['Shaft Mis.'] = none_list
    
    A = df_in.iloc[i]['1.0'] > E1 and df_in.iloc[i]['2.0'] > E1 and df_in.iloc[i]['2.0']      < 0.5 *df_in.iloc[i]['1.0']
    B = df_in.iloc[i]['1.0'] > E1 and df_in.iloc[i]['2.0'] > E1 and 1.5 *df_in.iloc[i]['1.0'] > df_in.iloc[i]['2.0'] > 0.5 *df_in.iloc[i]['1.0']
    C = df_in.iloc[i]['1.0'] > E1 and df_in.iloc[i]['2.0'] > E1 and 1.5 *df_in.iloc[i]['1.0'] < df_in.iloc[i]['2.0'] 
    D = df_in.iloc[i]['3.0'] > E1 and df_in.iloc[i]['4.0'] > E1 and df_in.iloc[i]['4.0'] > E1
    
    if not A and not B:
        df_in.loc[df_in.index[i],'Shaft Mis.'] = 'Green'
    if B and D:
        df_in.loc[df_in.index[i],'Shaft Mis.'] = 'Yellow'
    if C and D:
        df_in.loc[df_in.index[i],'Shaft Mis.'] = 'Red'
#------------------------------------------------------------------------------
def df_Harmonics(df_FFT,df_time,fs,fichero):

    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]
    n_blades   = 12
    fecha = []
    for k in df_FFT.index:
        print (k,datetime.datetime.fromtimestamp(k))
        fecha.append(datetime.datetime.fromtimestamp(k))
    
    columnas   = ['i Oil Whirl'  ,'Oil Whirl'  ,'f Oil Whirl'  ,'BW Oil Whirl'  ,
                  'i Flow T.'    ,'Flow T.'    ,'f Flow T.'    ,'BW Flow T.'    ,
                  'i 1/4'        ,'1/4'        ,'f 1/4'        ,'BW 1/4'        ,
                  'i 1/3'        ,'1/3'        ,'f 1/3'        ,'BW 1/3'        ,
                  'i 0.5'        ,'0.5'        ,'f 0.5'        ,'BW 0.5'        ,
                  'i 2nd Highest','2nd Highest','f 2nd Highest','BW 2nd Highest',
                  'i 1.0'        ,'1.0'        ,'f 1.0'        ,'BW 1.0'        ,
                  'i 1.5'        ,'1.5'        ,'f 1.5'        ,'BW 1.5'        ,
                  'i 2.0'        ,'2.0'        ,'f 2.0'        ,'BW 2.0'        ,
                  'i 2.5'        ,'2.5'        ,'f 2.5'        ,'BW 2.5'        ,
                  'i 3.0'        ,'3.0'        ,'f 3.0'        ,'BW 3.0'        ,
                  'i 3.5'        ,'3.5'        ,'f 3.5'        ,'BW 3.5'        ,
                  'i 4.0'        ,'4.0'        ,'f 4.0'        ,'BW 4.0'        ,
                  'i 5.0'        ,'5.0'        ,'f 5.0'        ,'BW 5.0'        ,
                  'i 1BPF'       ,'1BPF'       ,'f 1BPF'       ,'BW 1BPF'       ,
                  'i 1BPF-1.0'   ,'1BPF-1.0'   ,'f 1BPF-1.0'   ,'BW 1BPF-1.0'   ,
                  'i 1BPF+1.0'   ,'1BPF+1.0'   ,'f 1BPF+1.0'   ,'BW 1BPF+1.0'   ,
                  'i 2BPF'       ,'2BPF'       ,'f 2BPF'       ,'BW 2BPF'       ,
                  'i 2BPF-1.0'   ,'2BPF-1.0'   ,'f 2BPF-1.0'   ,'BW 2BPF-1.0'   ,
                  'i 2BPF+1.0'   ,'2BPF+1.0'   ,'f 2BPF+1.0'   ,'BW 2BPF+1.0'   ,
                  'T. RMS(t)'  ,'T. RMS(f)'                       ]
    df_harm    = pd.DataFrame(#index = df_FFT.index,
                              index = fecha,
                              columns = columnas,
                              data = np.zeros((n_traces,len(columnas))))
    f_1x       = 1480/60
    f_1x       = 24.7
    l_mitad    = int(l/2)
    f          = np.arange(l)/l*fs
    
    escala ='Peak' # ---------------escalado para CALCULAR
    if escala == 'RMS':
        cte = 1.63 * 1          # 2              por la ventana de hanning
                                # 1 / np.sqrt(2) por RMS
    else:
        cte = 1.63 * np.sqrt(2) # 2              por la ventana de hanning
                                # 1              por Peak
    
    for medida in range(n_traces):
        print ('medida numero',medida)
        

        sptrm_C                             = cte * df_FFT.iloc[medida].values * 2 # Solo trabajamos con 1º z Nyquist       
        n_maxi_C                            = np.argmax(sptrm_C[0:l_mitad])
       
        f_1x                                = f[n_maxi_C]
        df_harm.iloc[medida]['f 1.0']       = f_1x
        df_harm.iloc[medida]['T. RMS(t)']   = np.sqrt(np.sum( (df_time.iloc[medida].values    ) ** 2)/l)
        df_harm.iloc[medida]['T. RMS(f)']   = np.sqrt(np.sum( (1.63*df_FFT.iloc[medida].values) ** 2)  )
        
        TRH                                 = 0*np.std(sptrm_C[0:l_mitad])/3
        
        #----------------------------------------------------------------------------------------- El segundopico mas grande
        #indexes                             = detect_peaks(sptrm_C[0:l_mitad], mph = TRH , mpd = 1*l/fs)
        indexes, properties                 = find_peaks(sptrm_C[0:l_mitad],height  = TRH ,prominence = 0.01 , width=1 , rel_height = 0.75)

        array_peaks                           = sptrm_C[indexes]
        i_L                                   = np.argmax(array_peaks)
        #print(array_peaks[i_L])
        array_peaks[i_L]                      = 0
        i_L2                                  = np.argmax(array_peaks)
        #print(array_peaks[i_L2])
        df_harm.iloc[medida]['i 2nd Highest']  = indexes[i_L2]
        df_harm.iloc[medida]['f 2nd Highest']  = f[indexes[i_L2]]
        df_harm.iloc[medida]['2nd Highest']    = np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 ))
        df_harm.iloc[medida]['BW 2nd Highest'] = properties["widths"][i_L2]
        #print(np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 )),sptrm_C[indexes[i_L2]])

                                         #----------------------------------
        
        for counter,k in enumerate(indexes) :       
            #--------------------------------------------------------------------------------OIL Whirl--pico 0.38X to 0.48X (9.38125-11.85)
            if  f_1x*0.38   <= f[k] <= f_1x*0.48:
                print('Oil')
                piko                          = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                if piko > df_harm.iloc[medida]['Oil Whirl']:
                    df_harm.iloc[medida]['i Oil Whirl' ] = k
                    df_harm.iloc[medida]['Oil Whirl'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f Oil Whirl' ] = f[k]
                    df_harm.iloc[medida]['BW Oil Whirl'] = f[int(properties["widths"][counter]) ]
            #-----------------------------------------------------------------------------------------el pico de 12 a 24 HZ        
            if  12   <= f[k] <= 24:
                print('Flow')
                piko                          = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                if piko > df_harm.iloc[medida]['Flow T.']:
                    df_harm.iloc[medida]['i Flow T.' ] = k
                    df_harm.iloc[medida]['Flow T.'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f Flow T.' ] = f[k]
                    df_harm.iloc[medida]['BW Flow T.'] = f[int(properties["widths"][counter]) ]                    
                             #-----------------------------------------------------------1/3x-----------                                                                               
            if  1/3-0.02   <= f[k]/f_1x <= 1/3+0.02  :
                piko                          = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                print('>>>>>>    1/3    >>>>>>>>>>')               
                if piko > df_harm.iloc[medida]['1/3']:
                    df_harm.iloc[medida]['i 1/3' ] = k
                    df_harm.iloc[medida]['1/3'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 1/3' ] = f[k]
                    df_harm.iloc[medida]['BW 1/3'] = f[int(properties["widths"][counter]) ]
                             #-----------------------------------------------------------1/4x----------- 
            if  1/4-0.02   <= f[k]/f_1x <= 1/4+0.02  :
                piko                          = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                print('>>>>>>    1/4    >>>>>>>>>>')
                if piko > df_harm.iloc[medida]['1/4']:
                    df_harm.iloc[medida]['i 1/4' ] = k
                    df_harm.iloc[medida]['1/4'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 1/4' ] = f[k]
                    df_harm.iloc[medida]['BW 1/4'] = f[int(properties["widths"][counter]) ]                
                             #-----------------------------------------------------------0.5x-----------                                                                               
            if  0.5-0.02   <= f[k]/f_1x <= 0.5+0.02  :
                piko                          = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                print('>>>>>>    0.5    >>>>>>>>>>')
                if piko > df_harm.iloc[medida]['0.5']:
                    df_harm.iloc[medida]['i 0.5' ] = k
                    df_harm.iloc[medida]['0.5'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 0.5' ] = f[k]
                    df_harm.iloc[medida]['BW 0.5'] = f[int(properties["widths"][counter]) ]
                #print(f[k],'bandwidth >>>>>>>>>>>>>>>>>>>>>',f[int(properties["widths"][counter]) ] )                                                                             
                                            #-----------------------------------------------------------1.0x---------                  
            if  1-0.02   <= f[k]/f_1x <= 1+0.02  :
                print('>>>>>>    1.0    >>>>>>>>>>')
                piko_1x                        = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 1.0' ] = k
                df_harm.iloc[medida]['1.0'   ] = piko_1x #/ Max_value
                df_harm.iloc[medida]['f 1.0' ] = f[k]
                df_harm.iloc[medida]['BW 1.0'] = f[int(properties["widths"][counter]) ]
                                            #-----------------------------------------------------------1.5x---------                 
            if  1.5-0.02   <= f[k]/f_1x <= 1.5+0.02  :
                
                print('>>>>>>    1.5    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                if piko > df_harm.iloc[medida]['1.5']:
                    df_harm.iloc[medida]['i 1.5' ] = k
                    df_harm.iloc[medida]['1.5'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 1.5' ] = f[k]
                    df_harm.iloc[medida]['BW 1.5'] = f[int(properties["widths"][counter]) ]
                                            #-----------------------------------------------------------2.0x---------           
            if  2-0.02   <= f[k]/f_1x <= 2+0.02  :
                print('>>>>>>    2.0    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 2.0' ] = k
                df_harm.iloc[medida]['2.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 2.0' ] = f[k]
                df_harm.iloc[medida]['BW 2.0'] = f[int(properties["widths"][counter]) ]
                                            #-----------------------------------------------------------2.5x---------                
            if  2.5-0.02   <= f[k]/f_1x <= 2.5+0.02  :
                print('>>>>>>    2.5    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                if piko > df_harm.iloc[medida]['2.5']:
                    df_harm.iloc[medida]['i 2.5' ] = k
                    df_harm.iloc[medida]['2.5'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 2.5' ] = f[k]
                    df_harm.iloc[medida]['BW 2.5'] = f[int(properties["widths"][counter]) ]
                                            #------------------------------3.0x                                      
            if  3-0.02   <= f[k]/f_1x <= 3+0.02     :
                print('>>>>>>    3.0    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 3.0' ] = k
                df_harm.iloc[medida]['3.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 3.0' ] = f[k]
                df_harm.iloc[medida]['BW 3.0'] = f[int(properties["widths"][counter]) ]
                                            #------------------------------3.5x                  
            if  3.5-0.02   <= f[k]/f_1x <= 3.5+0.02  :
                print('>>>>>>    3.5    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                if piko > df_harm.iloc[medida]['3.5']:
                    df_harm.iloc[medida]['i 3.5' ] = k
                    df_harm.iloc[medida]['3.5'   ] = piko #/ Max_value
                    df_harm.iloc[medida]['f 3.5' ] = f[k]
                    df_harm.iloc[medida]['BW 3.5'] = f[int(properties["widths"][counter]) ]
                                            #------------------------------4.0x   
            if  4-0.02   <= f[k]/f_1x <= 4+0.02 :
                print('>>>>>>    4.0    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 4.0' ] = k
                df_harm.iloc[medida]['4.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 4.0' ] = f[k]
                df_harm.iloc[medida]['BW 4.0'] = f[int(properties["widths"][counter]) ]
                #print()
            if  5-0.02   <= f[k]/f_1x <= 5+0.02 :
                print('>>>>>>    5.0    >>>>>>>>>>')
                piko                           = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 5.0' ] = k
                df_harm.iloc[medida]['5.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 5.0' ] = f[k]
                df_harm.iloc[medida]['BW 5.0'] = f[int(properties["widths"][counter]) ]
                                             #------AERODYNAMIC FAILURE--------
            if n_blades-0.02 <= f[k]/f_1x <= n_blades+0.02:
                #print('holaaaaaaaa',f[k],df_harm.iloc[medida]['f 1BPF'])
                piko                            = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 1BPF '] = k #/ Max_value
                df_harm.iloc[medida]['1BPF   '] = piko #/ Max_value
                df_harm.iloc[medida]['f 1BPF '] = f[k]
                df_harm.iloc[medida]['BW 1BPF'] = f[int(properties["widths"][counter]) ]
            if n_blades-1-0.02 <= f[k]/f_1x <= n_blades-1+0.02 :
                piko                                = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 1BPF-1.0 '] = k
                df_harm.iloc[medida]['1BPF-1.0   '] = piko #/ Max_value
                df_harm.iloc[medida]['f 1BPF-1.0 '] = f[k]
                df_harm.iloc[medida]['BW 1BPF-1.0'] = f[int(properties["widths"][counter]) ]
            
            if n_blades+1-0.02 <= f[k]/f_1x <= n_blades+1+0.02 :
                piko                                = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 1BPF+1.0 '] = k
                df_harm.iloc[medida]['1BPF+1.0   '] = piko #/ Max_value
                df_harm.iloc[medida]['f 1BPF+1.0 '] = f[k]
                df_harm.iloc[medida]['BW 1BPF+1.0'] = f[int(properties["widths"][counter]) ]

            if 2*n_blades-0.02 <= f[k]/f_1x <= 2*n_blades+0.02:
                piko                                = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 2BPF' ] = k
                df_harm.iloc[medida]['2BPF'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 2BPF' ] = f[k]
                df_harm.iloc[medida]['BW 2BPF'] = f[int(properties["widths"][counter]) ]
                
            if 2*n_blades-1-0.02 <= f[k]/f_1x <= 2*n_blades-1+0.02 :
                piko                                = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 2BPF-1.0' ] = k
                df_harm.iloc[medida]['2BPF-1.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 2BPF-1.0' ] = f[k]
                df_harm.iloc[medida]['BW 2BPF-1.0'] = f[int(properties["widths"][counter]) ]
            
            if 2*n_blades+1-0.02 <= f[k]/f_1x <= 2*n_blades+1+0.02 :
                piko                                = np.sqrt(np.sum( sptrm_C[ int(properties["left_ips"][counter]) : int(properties["right_ips"][counter]) ]**2 ))
                df_harm.iloc[medida]['i 2BPF+1.0' ] = k
                df_harm.iloc[medida]['2BPF+1.0'   ] = piko #/ Max_value
                df_harm.iloc[medida]['f 2BPF+1.0' ] = f[k]
                df_harm.iloc[medida]['BW 2BPF+1.0'] = f[int(properties["widths"][counter]) ]
                
 
    return df_harm
#------------------------------------------------------------------------------
    


def plot_waterfall(df_in,fs,fmin,fmax):
    alfa      = 0.7
    cc        = lambda arg: colorConverter.to_rgba(arg, alpha=alfa)
    color1    = cc('b')
    color2    = cc('y')
   
    col_list  = [color1,color2]
    n_traces  = df_in.shape[0]
    l         = df_in.shape[1]
    #f         = np.arange(l)/l*fs
            
    escala ='Peak' #escalado para PLOTEAR
    if escala == 'RMS':
        cte   = 2/np.sqrt(2) # 2          por la ventana de hanning
                             # 1 / np.sqrt(2) por RMS
        label = 'RMS'
    else:
        cte   = 2*1          # 2          por la ventana de hanning
                             # 1          por Peak
        label = 'Peak'
        
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
        
        curva            = cte *  df_in.iloc[counter].values* 2 # Solo trabajamos con 1º z Nyquist
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
    ax.set_ylabel('days')
    ax.set_ylim3d(0, traces[np.size(traces)-1])
    ax.set_zlabel(label+'mm/s')
    ax.set_zlim3d(np.min(color), np.max(color))
    ax.set_title('mm/sg '+ label)
    
    second_plot = plt.axes([.05, .75, .2, .2], facecolor='w')
    #fig, ax1 = plt.subplots()
    plt.plot(traces,harm.loc[:,'T. RMS(t)'].values)
    plt.plot(traces,harm.loc[:,'T. RMS(f)'].values)
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('RMS')
    plt.tight_layout()
    #plt.show()
    
    return color , verts
#------------------------------------------------------------------------------   

def plot_waterfall2(df_in,fs,fmin,fmax):
   
    col_list  = ['b','r']
    l         = df_in.shape[1]
    #df_in.sort_index(ascending=False,inplace=True)
    escala ='RMS' #escalado para PLOTEAR
    if escala == 'Peak':
        cte   = 2/np.sqrt(2) # 2          por la ventana de hanning
                             # 1 / np.sqrt(2) por RMS
        label = 'RMS'
    else:
        cte   = 2*1          # 2          por la ventana de hanning
                             # 1          por Peak
        label = 'Peak'
        
    #fmax      = 55
    n_fmin    = np.int(l*fmin/(fs))
    n_fmax    = np.int(l*fmax/(fs))
    f_portion = np.arange(n_fmin,n_fmax)/n_fmax*fmax
    
    y_portion = np.ones(np.size(f_portion)) #---array para el eje Y
    
    t_traces    = df_in.index.values
    t_traces    = (t_traces-t_traces[0])/24/3600
    
    #fig       = plt.figure()
    fig=plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax        = fig.gca(projection='3d')    
    inic_day  = datetime.datetime.fromtimestamp(df_in.index[0]).day
    color_counter = 0
    for counter,indice in enumerate(df_in.index):
        
        if datetime.datetime.fromtimestamp(df_in.index[counter]).day != inic_day:
            color_counter = color_counter+1
        
        curva            = cte *  df_in.iloc[counter].values * 2 # Solo trabajamos con 1º z Nyquist
        curva_p          = (curva[n_fmin:n_fmax])
        inic_day         = datetime.datetime.fromtimestamp(df_in.index[counter]).day
        #print(t_traces[counter])
        ax.plot(f_portion ,y_portion*t_traces[counter],curva_p,color=col_list[np.mod(color_counter,2)], linewidth = 0.2)

    ax.view_init(40, 90)
    ax.set_xlabel('Hertz')
    ax.set_xlim3d(fmin, fmax)
    ax.set_ylabel('days')
    ax.set_ylim3d(0, t_traces[np.size(t_traces)-1])
    ax.set_zlabel(label+'mm/s')
    #ax.set_zlim3d(np.min(color), np.max(color))
    ax.set_title('mm/sg '+ label)
    
    second_plot = plt.axes([.03, .75, .2, .2], facecolor='w')
    plt.plot(t_traces,harm.loc[:,'T. RMS(t)'].values)
    plt.plot(t_traces,harm.loc[:,'1.0'].values)
    plt.legend(('Total RMS', 'Peak @1x'),loc='upper right')
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('')
    
    
    
    thrid_plot = plt.axes([.03, .03, .2, .2], facecolor='w')    
    plt.plot(t_traces,harm.loc[:,'f 1.0'].values)    
    plt.plot(t_traces,harm.loc[:,'f 2.0'].values)    
    plt.plot(t_traces,harm.loc[:,'f 3.0'].values)    
    plt.plot(t_traces,harm.loc[:,'f 4.0'].values)
    plt.legend(( 'f 1.0','f 2.0','f 3.0','f 4.0'),loc='upper right')
    plt.grid(True)
    plt.xlabel('says')
    plt.ylabel('mm/s')
    plt.title('')
    
    fourth_plot = plt.axes([.8, .03, .2, .2], facecolor='w')
    plt.plot(t_traces,harm.loc[:,'f 0.5'].values)
    plt.plot(t_traces,harm.loc[:,'f 1.5'].values)
    plt.plot(t_traces,harm.loc[:,'f 2.5'].values)    
    plt.plot(t_traces,harm.loc[:,'f 3.5'].values)
    plt.legend(('f 0.5','f 1.5','f 2.5','f 3.5'),loc='upper right')
    plt.grid(True)
    plt.xlabel('days')
    plt.ylabel('mm/s')
    plt.title('')
    
    plt.tight_layout()
    plt.show()
    
    """
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t_traces,harm.loc[:,'T. RMS(t)'].values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t_traces,harm.loc[:,'Hz'].values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    
    trid_plot = plt.axes([.05, .05, .2, .2], facecolor='w')
    plt.plot(t_traces,harm.loc[:,'Hz'].values)
    plt.legend(('Total RMS', 'Peak @1x'),loc='upper right')
    plt.grid(True)
    plt.xlabel('sg')
    plt.ylabel('Hz')
    plt.title('')
    """
    return 
#------------------------------------------------------------------------------  
def write_to_excel(sheet,instante,df_in,n_trace):
    
    for row_counter,indice in enumerate(df_in.index):
        sheet.write( n_trace*4+1+row_counter,0,str(datetime.datetime.fromtimestamp(instante)) )
        
        for column_counter,columna in enumerate(df_in.columns):
            sheet.write( n_trace*4+1+row_counter,column_counter+1, df_in.iloc[row_counter][column_counter]) 
    
    return
#------------------------------------------------------------------------------
def find_closest(date):
    segundos = time.mktime(date.timetuple()) 
    indice = np.argmin( np.abs(df_speed.index.values-segundos) )
    print('Hora de la captura', datetime.datetime.fromtimestamp(df_speed.index[indice]))
    print('Diferencia en minutos',(df_speed.index[indice]-segundos)/60) 
    
    waveform = df_speed.iloc[indice]

    b, a       = signal.butter(3,2*10/fs,'highpass',analog=False)
    l          = np.size(waveform)
    l_mitad    = int(l/2)
    f          = np.arange(l)/l*fs
    #----------quitamos los 2 HZ del principio
    
    label = 'Peak'
    cte_p = 1
    cte_c = np.sqrt(2)

    hann       = np.hanning(l)
    wave_f     = signal.filtfilt(b, a, waveform)
    sptrm_P    = np.abs(np.fft.fft(wave_f * 2    * hann)/l)
    sptrm_C    = np.abs(np.fft.fft(wave_f * 1.63 * hann)/l)

    TRH        = 0*np.std(sptrm_P[0:l_mitad])/3
    
    indexes, properties = find_peaks( cte_p*2*sptrm_C[0:l_mitad],height  = TRH ,prominence = 0.01 , width=1 , rel_height = 0.75)
    
    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')
    plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)
    plt.plot(f[0:l_mitad] , cte_p*2*sptrm_P[0:l_mitad],'b')
    #plt.plot(f[indexes]   , cte_p*2*sptrm_P[indexes]  ,'o')

    for counter,k in enumerate(harm.columns):
        
        if k[0] == 'i':
            print (counter,k)
            index = int(harm.iloc[indice][counter])
            if f[index] !=0:
                plt.plot(f[index] , cte_p*2*sptrm_P[index]  ,'o')
                plt.text(f[index] , cte_p*2*sptrm_P[index],str(k))
            
    plt.ylabel('mm/s '+label)
    plt.title(str(date))
    plt.xlabel('Hz')
    plt.grid(True)
    
    plt.vlines(x=f[indexes], ymin=cte_p*2*sptrm_P[indexes] - properties["prominences"],ymax = cte_p*2*sptrm_P[indexes], color = "C1")
    plt.hlines(y=properties["width_heights"], xmin=f[properties["left_ips"].astype(int)],xmax=f[properties["right_ips"].astype(int)], color = "C1")
#    plt.hlines(y=properties["width_heights"], xmin=f[properties["left_bases"].astype(int)],xmax=f[properties["right_bases"].astype(int)], color = "C2")
    
    
    ax1.xaxis.set_minor_locator(minorLocator)

    plt.tick_params(which='both', width=2)
    plt.tick_params(which='major', length=7)
    plt.tick_params(which='minor', length=4, color='r')
    
    
    
    
    
    
    
    
    return
    
    
pi        = np.pi
G         = 9.81

path      = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018'
month     = '10'
day       = '01'
hour      = '01'

                    #  False => from database
                    # True   => from file
load_disk = True 
when      = '\\' +month + '\\' +day + '\\' + hour
when_l    = month+'_'+day+'_'+hour
path      = path + when
fs        = 5120.0
b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
#------------------------------------------------------------------------------

#maquina          = 'H4-FA-0001';localizacion     = 'MH2' #SH4/MH2
maquina          = 'H4-FA-0002';localizacion     = 'SH3' #SH3/4
#maquina          = 'C2-FA-0001';localizacion     = 'MH2' #SH4/MH2
#maquina          = 'C2-FA-0002';localizacion     = 'MH2' #SH4/MH2

if load_disk == True:
    df_accel,l_devices = load_vibrationData(path,maquina,localizacion)
    df_accel.to_pickle('accel_'+maquina+'_'+localizacion+'_'+when_l+'.pkl')  #------------en G´s-
else:
    df_accel           = pd.read_pickle('accel_'+maquina+'_'+localizacion+'_'+ when_l +'.pkl') 

n_traces           = df_accel.shape[0]
l                  = df_accel.shape[1]
#f                  = np.arange(l)/l*fs
#tiempo             = np.arange(l)/fs

                         #---tiene una DC horrible
df_accel_rem_DC    = DataFrame_remove_DC(df_accel)
df_speed           = G*1000*DataFrame_integrate(df_accel_rem_DC)/fs
                          #---creo esta DF para generar los DF de los espectros
#df_accel_f         = DataFrame_filt(df_accel,b,a)
df_speed_f         = DataFrame_filt(df_speed,b,a)    

df_SPEED           = DataFrame_fft_abs(df_speed_f)

n_file_SPEED       = 'SPEED_'+maquina+'_'+localizacion+'_'+when_l

"""
if load_disk == False:    
    df_SPEED.to_pickle( n_file_SPEED + '.pkl')
else:
    df_SPEED           = pd.read_pickle(n_file_SPEED + '.pkl') 
"""



harm     = df_Harmonics(df_SPEED, df_speed_f, fs,n_file_SPEED)

harm     = df_unbalance(harm)
harm     = df_clearance(harm)
harm     = oil_whirl(harm)
harm     = oil_whip(harm)
harm     = CENTRIFUGAL_FAN_AERODYNAMIC_FAILURE(harm)
harm     = flow_turbulence(harm)
harm     = Plain_bearing_block_looseness(harm)
df       = harm
df.index = pd.to_datetime(df.index)
find_closest(datetime.datetime(2018, int('0'+month), int('0'+day), 0, 0))

#df_SPEED = pd.read_pickle('SPEED_'+maquina+'_'+localizacion+'_'+month+'_'+day+'.xls') 

color,vertices = plot_waterfall(df_SPEED,fs,0,400)
plot_waterfall2(df_SPEED,fs,0,400)

#if load_disk == False:
writer = pd.ExcelWriter('result_'+maquina+'_'+localizacion+'_'+month+'_'+day+'.xlsx')
df.to_excel(writer, 'DataFrame')
writer.save()




