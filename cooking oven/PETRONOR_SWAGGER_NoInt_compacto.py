import requests

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
from matplotlib.colors import colorConverter

#from detect_peaks import detect_peaks
from scipy.signal import find_peaks

import pandas as pd
import os
from pandas import DataFrame

#import xlwt
#from   xlwt import Workbook


#from scipy import stats

class fingerprint:
    def __init__(self,        label,tipo,f1,f2,relative):
        self.label = label
        self.tipo  = tipo
        self.f1    = f1
        self.f2    = f2
        self.relative = relative

    def __str__(self):
        s = ''.join(['Label    : ', str(self.label),   '\n',
                     'Tipo     : ', str(self.tipo),    '\n',
                     'f1       : ', str(self.f1),      '\n',
                     'f2       : ', str(self.f2),      '\n',
                     'Relative : ', str(self.relative),'\n',
                     '\n'])
        return s

#------------------------------------------------------------------------------

def load_vibrationData(input_data, MeasurePointId, num_tramas, assetId):

    data           = []
    date           = []
    lista_maquinas = []
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
                cal_factor = np.float(res.Props.iloc[0][4]['Value'])
                data.append(np.asarray(res.Value.values[0])*cal_factor)

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
                date.append(fecha)

    df_out     = DataFrame(data=data, index=date)
    df_out.sort_index(inplace=True)
    return df_out, lista_maquinas



#------------------------------------------------------------------------------

def velocity(df_in):
    l         = df_in.shape[1]
    hann      = np.hanning(l) #
    b, a      = signal.butter(3,2*5/fs,'highpass',analog=False)
    G         = 9.81
    
    
    df_abs    = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    df_angle  = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    df_time   = pd.DataFrame(np.nan,index = df_in.index,columns = df_in.columns.values)
    for counter,indice in enumerate(df_in.index):
        trace                   = df_in.iloc[counter].values- np.mean(df_in.iloc[counter].values)
        trace                   = np.cumsum (G*1000*trace/fs)
        trace                   = signal.filtfilt(b, a, trace) * hann
        TRACE                   = np.fft.fft(trace/l)
        df_abs.iloc[counter]    = np.abs( TRACE )
        df_angle.iloc[counter]  = np.angle( TRACE )
    return df_abs,df_angle,df_time
    

#------------------------------------------------------------------------------
def df_clearance(df_in):
    print('Clearance Failure')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['Clearance Failure'] = none_list

    for i in range (n_traces):
        v_1x   = df_in.iloc[i]['E 1.0']
        v_2x   = df_in.iloc[i]['E 2.0']
        v_3x   = df_in.iloc[i]['E 3.0']

        v_0_5x = df_in.iloc[i]['E 1/2']
        v_1_5x = df_in.iloc[i]['E 3/2']
        v_2_5x = df_in.iloc[i]['E 5/2']
                                            #-------1.0x 2.0x 3.0x decreciente
        bool1 =  v_1x >v_2x > v_3x
                                            # --2.0x >2% 1.0x and 3.0x >2% 1.0x
        bool2 = (v_2x > 0.02 * v_1x) and (v_3x > 0.02 * v_1x)
                                            #-------0.5x 1.5x 2.5x decreciente
        bool3 =  v_0_5x > v_1_5x > v_2_5x
                                            # ------0.5x >2% 1.0x and 1.5x > 2% 1.0x and 2.5x > 2% 1.0x
        bool4 = (v_0_5x > 0.02 * v_1x) and (v_1_5x > 0.02 * v_1x) and (v_2_5x > 0.02 * v_1x)

        #print (bool1,bool2,bool3,bool4)
        A = bool1 and bool2
        B = bool3 and bool4

        if (A == False) and (B == False):
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Green'

        if A or B:
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Yellow'

        if A and B:
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Red'

    return df_in
#------------------------------------------------------------------------------
def df_unbalance(df_in):
    print('Unbalance Failure')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['Unbalance Failure'] = none_list

    for i in range (n_traces):
        #print(df_in.iloc[i].values[1:8])
                                            # max armonicos = 1.0x
        f_max1 = df_in.iloc[i]['E 1.0']
                                            # max del resto de pikos
        s_max1 = df_in.iloc[i]['E 2nd Highest']

                                            #---1X meno que el umbral
        A  = f_max1        < 4
                                            #---El 15% 1x < resto armonicos.
                                            #   es decir 1X no es dominante
        B  = f_max1 * 0.15 < s_max1

                                            #--------------------------Green
        if A and B:
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'
                                            #--------------------------yellow
                                            #   Xor = cualquiera de ellas
                                            #        pero no ambas
        if (A == False) ^   (B == False):
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Yellow'
                                            #--------------------------Red
                                            # las dos falsas
        if (A == False) and (B == False):
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'

    return df_in
#------------------------------------------------------------------------------
def oil_whirl(df_in):
    print('Oil Whirl Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Oil Whirl Failure'] = none_list

    for i in range (n_traces):
                                            #-----------green-----------------
                                            # no detected Peak in '0.38-0.48'
        if df_in.iloc[i]['E Oil Whirl'] < E1:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Green'
                                            #-----------yellow-----------------
                                            # Detected Peak in '0.38-0.48'
                                            #         but
                                            # Peak in '0.38-0.48' < 2% 1.0x
        if df_in.iloc[i]['E Oil Whirl'] > E1 and df_in.iloc[i]['E Oil Whirl'] < 0.02 * df_in.iloc[i]['E 1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Yellow'
                                            #-----------red--------------------
                                            # Peak in '0.38-0.48' > 2% 1.0x
        if df_in.iloc[i]['E Oil Whirl'] > E1 and df_in.iloc[i]['E Oil Whirl'] > 0.02 * df_in.iloc[i]['E 1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Red'

    return df_in

#------------------------------------------------------------------------------
def oil_whip(df_in):
    print('Oil Whip Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)

    for i in range (n_traces):
        none_list.append('None')
    df_in['Oil Whip Failure'] = none_list

    for i in range (n_traces):
        A = (df_in.iloc[i]['E 1/2'] >= E1 and df_in.iloc[i]['BW 1/2'] >= 4)  and  ((df_in.iloc[i]['E 5/2'] >= E1) and df_in.iloc[i]['BW 2.5'] >= 4)
        B = df_in.iloc[i]['E 1/2' ] > 0.02 *  df_in.iloc[i]['E 1.0']
        C = df_in.iloc[i]['E 5/2' ] > 0.02 *  df_in.iloc[i]['E 1.0']
        #print(A,B,C)
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
        if A ^ ((B ^ C)) :
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Yellow'
                                             #-----------red-------------------
                                             #     2H BW at 0.5 >2% 1.0x
                                             #           AND
                                             #     2H BW at 2.5 >2% 1.0x
        if A and B and C:
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Red'

        #print('oild whip failure',df_in.loc[df_in.index[i],'Oil Whip Failure'])
    return df_in
#------------------------------------------------------------------------------

def Blade_faults(df_in):
    print('Blade Faults Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Blade Faults Failure'] = none_list

    for i in range (n_traces):
        A = df_in.iloc[i]['E 12.0'] > E1
        B = df_in.iloc[i]['E 12.0'] > E1 and df_in.iloc[i]['E 24.0'] > E1
        C = df_in.iloc[i]['E 12.0'] > E1 and (df_in.iloc[i]['E 11.0'] > E1 or df_in.iloc[i]['E 13.0'] > E1)
        D = C and df_in.iloc[i]['E 24.0'] > E1
        E = C and df_in.iloc[i]['E 24.0'] > E1 and (df_in.iloc[i]['E 23.0'] > E1 or df_in.iloc[i]['E 25.0'] > E1)
        F = df_in.iloc[i]['E 12.0'] < E1 and df_in.iloc[i]['E 24.0'] < E1
        #print('Blade Faults         ',A,B,C,D,E,F)
                                             #  Tabla de verdad progresiva
                                             #  puede empezar siendo verde,
                                             #  acabar siendo rojo
        if A or B or F:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Green'
        if C or D:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Yellow'
        if E:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Red'

        #print(df_in.loc[df_in.index[i],'Blade Faults'] )
    return df_in
#------------------------------------------------------------------------------
def flow_turbulence(df_in):
    print('Flow Turbulence Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Flow Turbulence Failure'] = none_list

    for i in range (n_traces):
        A =        df_in.iloc[i]['E Flow T.'] <= 0.2
        B = 0.2 <= df_in.iloc[i]['E Flow T.'] <= df_in.iloc[i]['E 1.0']
        C =        df_in.iloc[i]['E Flow T.'] >  df_in.iloc[i]['E 1.0']
        #print('Flow Tur.           ',A,B,C)
        if A:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Green'
        if B:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Yellow'
        if C:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Flow Turbulence'] )
    return df_in

#------------------------------------------------------------------------------
def Plain_bearing_block_looseness(df_in):
    print('PBB looseness Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['PBB looseness Failure'] = none_list
    for i in range (n_traces):
        A = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and df_in.iloc[i]['E 3.0'] > E1 and df_in.iloc[i]['E 1.0'] <df_in.iloc[i]['E 2.0'] > df_in.iloc[i]['E 3.0']
        B = df_in.iloc[i]['E 1/2'] > E1 and df_in.iloc[i]['E 1/3'] > E1 and df_in.iloc[i]['E 1/4'] > E1
        #print('Plain Bearin block   ',A,B)
        if not A and not B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Green'
        if A or B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'PBB looseness'])
    return df_in
#------------------------------------------------------------------------------
def SHAFT_MISALIGNMENTS(df_in):
    print('Shaft Mis. Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Shaft Mis. Failure'] = none_list

    for i in range (n_traces):
        A = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and      df_in.iloc[i]['E 2.0'] < 0.5 *  df_in.iloc[i]['E 1.0']
        B = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and 1.5 *df_in.iloc[i]['E 1.0'] >        df_in.iloc[i]['E 2.0'] > 0.5 *df_in.iloc[i]['E 1.0']
        C = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and 1.5 *df_in.iloc[i]['E 1.0'] <        df_in.iloc[i]['E 2.0']
        D = df_in.iloc[i]['E 2.0'] > E1 and df_in.iloc[i]['E 3.0'] > E1 and      df_in.iloc[i]['E 4.0'] > E1 and df_in.iloc[i]['E 5.0'] > E1
        #print('Shaft miss          ',A,B,C,D)
        if A or not D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Green'
        if B and D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Yellow'
        if C and D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])
    return df_in

#------------------------------------------------------------------------------
def Pressure_Pulsations(df_in):
    print('Pressure P. Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Pressure P. Failure'] = none_list

    for i in range (n_traces):
        A = df_in.iloc[i]['E 1/3'] > E1 and df_in.iloc[i]['E 2/3'] > E1 and df_in.iloc[i]['E 4/3'] > E1 and df_in.iloc[i]['E 5/3'] > E1
        B = (df_in.iloc[i]['E 4/3'] > df_in.iloc[i]['E 1/3']) and (df_in.iloc[i]['E 8/3'] > df_in.iloc[i]['E 1/3']) and (df_in.iloc[i]['E 4.0'] > df_in.iloc[i]['E 1/3'])
        #print('Shaft miss          ',A,B,C,D)
        if not A:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Green'
        if A:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])
    return df_in

#------------------------------------------------------------------------------
def Surge_Effect(df_in):
    print('Surge E. Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Surge E. Failure'] = none_list


    for i in range (n_traces):
        A = df_in.iloc[i]['E Surge E. 0.33x 0.5x'] > E1
        B = df_in.iloc[i]['E Surge E. 12/20k'] > E1

        if not A:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Green'
        if A:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])
    return df_in

#------------------------------------------------------------------------------
def Severe_Misaligment(df_in):
    print('Severe Mis. Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')
    df_in['Severe Mis. Failure'] = none_list

    for i in range (n_traces):

        counter_A = 0
        for m in [2,3,4,5,6,7,8,9,10]:
            if df_in.iloc[i]['E '+str(m)+'.0'] > 0.02*df_in.iloc[i]['E 1.0']:
                counter_A = counter_A+1
        A = counter_A >= 3

        counter_B = 0
        for m in ['5/2','7/2','9/2','11/2','13/2','15/2','17/2','19/2']:
            if df_in.iloc[i]['E '+m] > 0.02*df_in.iloc[i]['E 1.0']:
                counter_B = counter_B+1
        B = counter_B >= 3

        #print ('Counter A in Severe Mis. Failure',counter_A,A)
        #print ('Counter B in Severe Mis. Failure',counter_B,B)

        C = df_in.iloc[i]['E 2.0'] > df_in.iloc[i]['E 1.0']

        if not A:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Green'
        if A ^ B:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Yellow'
        if A and B and C:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])
    return df_in

#------------------------------------------------------------------------------
def Loose_Bedplate(df_in):
    print('Loose Bedplate Failure')
    n_traces   = df_in.shape[0]
    none_list  = []

    for i in range (n_traces):
        none_list.append('None')
    df_in['Loose Bedplate Failure'] = none_list

    for i in range (n_traces):

        A = 2.5 * np.sqrt(2) > df_in.iloc[i]['E 1.0'] > 0
        B = 2.5 * np.sqrt(2) < df_in.iloc[i]['E 1.0'] < 6.00 * np.sqrt(2)
        C = 6.0 * np.sqrt(2) < df_in.iloc[i]['E 1.0']
        D = df_in.iloc[i]['E 3.0'] > df_in.iloc[i]['E 2.0']
        #print ('Loose Bedplate',A,B,C,D)
        if A:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Green'
        if B ^ C:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Yellow'
        if C and D:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Red'
#        print(df_in.iloc[i]['E 1.0']/np.sqrt(2),A,B,C)
#        print(D)
#        print('________________',df_in.loc[df_in.index[i],'Loose Bedplate Failure'])
    return df_in
#------------------------------------------------------------------------------

def df_Harmonics(df_FFT,fs):
    print('Extracting fingerprint')
    l          = df_FFT.shape[1]
    n_traces   = df_FFT.shape[0]

    fignerprint_list = [
                        fingerprint('1/2' ,'Peak',1/2 ,0,True),
                        fingerprint('1.0' ,'Peak',1   ,0,True),
                        fingerprint('3/2' ,'Peak',1.5 ,0,True),
                        fingerprint('2.0' ,'Peak',2   ,0,True),
                        fingerprint('5/2' ,'Peak',2.5 ,0,True),
                        fingerprint('3.0' ,'Peak',3   ,0,True),
                        fingerprint('7/2' ,'Peak',3.5 ,0,True),
                        fingerprint('4.0' ,'Peak',4   ,0,True),
                        fingerprint('9/2' ,'Peak',4.5 ,0,True),
                        fingerprint('5.0' ,'Peak',5   ,0,True),
                        fingerprint('11/2','Peak',5.5 ,0,True),
                        fingerprint('6.0' ,'Peak',6   ,0,True),
                        fingerprint('13/2','Peak',6.5 ,0,True),
                        fingerprint('7.0' ,'Peak',7   ,0,True),
                        fingerprint('15/2','Peak',7.5 ,0,True),
                        fingerprint('8.0' ,'Peak',8   ,0,True),
                        fingerprint('17/2','Peak',8.5 ,0,True),
                        fingerprint('9.0' ,'Peak',9   ,0,True),
                        fingerprint('19/2','Peak',9.5 ,0,True),
                        fingerprint('10.0','Peak',10  ,0,True),
                        fingerprint('21/2','Peak',10.5,0,True),

                        fingerprint('2/3' ,'Peak',2/3 ,0,True),
                        fingerprint('4/3' ,'Peak',4/3 ,0,True),
                        fingerprint('5/3' ,'Peak',5/3 ,0,True),
                        fingerprint('8/3' ,'Peak',8/3 ,0,True),

                        fingerprint('1/3' ,'Peak',1/3 ,0,True),
                        fingerprint('1/4' ,'Peak',1/4 ,0,True),

                        fingerprint('11.0','Peak',11  ,0,True),
                        fingerprint('12.0','Peak',12  ,0,True),
                        fingerprint('13.0','Peak',13  ,0,True),
                        fingerprint('23.0','Peak',23  ,0,True),
                        fingerprint('24.0','Peak',24  ,0,True),
                        fingerprint('25.0','Peak',25  ,0,True),

                        fingerprint('2nd Highest'        ,'Peak',0    ,fs/2 ,False),

                        fingerprint('Oil Whirl'          ,'Span',0.38 ,0.48 ,True),
                        fingerprint('500rpm'             ,'Span',0.38 ,0.48 ,True),
                        fingerprint('Flow T.'            ,'Span',12   ,24   ,False),
                        fingerprint('Surge E. 0.33x 0.5x','Span',0.33 ,0.5  ,True),
                        fingerprint('Surge E. 12/20k'    ,'Span',12000,20000,False),
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
        columnas.append('P '+word)
        columnas.append('E '+word)
        columnas.append('f '+word)
        columnas.append('BW '+word)

    df_harm    = pd.DataFrame(index = fecha,columns = columnas,data = np.zeros((n_traces,len(columnas))))

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
        print ('---------------------------------------------medida numero',df_harm.index[medida])
        sptrm_C                             = cte * df_FFT.iloc[medida].values * 2 # Solo trabajamos con 1º z Nyquist
        n_maxi_C                            = np.argmax(sptrm_C[0:l_mitad])
        indexes, properties                 = find_peaks(sptrm_C[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)
        array_peaks                         = sptrm_C[indexes]

        if 24<f[n_maxi_C]<25.5:
            f_1x       = f[n_maxi_C]
            #----------------------------------------------------------------------------------------- El segundopico mas grande
            i_L                                   = np.argmax(array_peaks)
            array_peaks[i_L]                      = 0
            i_L2                                  = np.argmax(array_peaks)

            df_harm.iloc[medida]['i 2nd Highest']  = indexes[i_L2]
            df_harm.iloc[medida]['P 2nd Highest']  = sptrm_C[indexes[i_L2]]
            df_harm.iloc[medida]['E 2nd Highest']  = np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 ))
            df_harm.iloc[medida]['f 2nd Highest']  = f[indexes[i_L2]]
            df_harm.iloc[medida]['BW 2nd Highest'] = properties["widths"][i_L2]
            #print(np.sqrt(np.sum( sptrm_C[int(properties["left_ips"][i_L2]) : int(properties["right_ips"][i_L2]) ]**2 )),sptrm_C[indexes[i_L2]])

                                             #----------------------------------
            delta = 0.03
            for counter,k in enumerate(indexes) :
                for h in fignerprint_list:
                    if h.tipo !='2nd Highest' :
                        fa = 0
                        fb = 0
                        if h.tipo == 'Peak':
                            fa = h.f1 - delta
                            fb = h.f1 + delta
                        if h.tipo == 'Span':
                            fa = h.f1
                            fb = h.f2
                        if h.relative == False:
                            fa = fa / f_1x
                            fb = fb / f_1x

                        if fa  <= f[k]/f_1x <= fb:
                            word = h.label
                            piko = np.sqrt(np.sum( sptrm_C[    int(np.round(properties["left_ips"][counter]) )  : int(np.round(properties["right_ips"][counter] )) +1     ]**2 ))
                            #print (word,'   ','piko=',piko,'f=',f[k],'fa=',fa,'f[k]/f_1x =',f[k]/f_1x,'fb=',fb)
                            if piko > df_harm.iloc[medida]['E '+word]:
                                df_harm.iloc[medida]['i '  + word] = k
                                df_harm.iloc[medida]['P '  + word] = sptrm_C[k]
                                df_harm.iloc[medida]['E '  + word] = piko #/ Max_value
                                df_harm.iloc[medida]['f '  + word] = f[k]
                                df_harm.iloc[medida]['BW ' + word] = f[int(properties["widths"][counter]) ]

    print('Fingerprints extracted')
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
    ax.set_ylabel('3edays')
    ax.set_ylim3d(0,    traces[np.size(traces)-1])
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



    return

#------------------------------------------------------------------------------
def find_closest(date,df_VELOCITY,df_velocity_f,df_harm):
    segundos = time.mktime(date.timetuple())
    captura = np.argmin( np.abs(df_velocity_f.index.values-segundos) )
    print('Hora de la captura', datetime.datetime.fromtimestamp(df_velocity_f.index[captura]))
    print('Diferencia en minutos',(df_velocity_f.index[captura]-segundos)/60)
    print(captura)
    waveform = df_velocity_f.iloc[captura].values

    date_exact = datetime.datetime.fromtimestamp(df_velocity_f.index[captura])

    l          = np.size(waveform)
    l_mitad    = int(l/2)
    f          = np.arange(l)/l*fs
    #----------quitamos los 2 HZ del principio

    label = 'Peak'
    SPTRM_P             = 2 * df_VELOCITY.iloc[captura].values * 2
    SPTRM_P_II          = 2 * np.abs(np.fft.fft(waveform*np.hanning(l)/l)) * 2

    indexes, properties = find_peaks( SPTRM_P[0:l_mitad],height  = 0 ,prominence = 0.01 , width=1 , rel_height = 0.75)

    minorLocator = AutoMinorLocator()
    #plt.figure(num=None, figsize=(24, 11), dpi=80, facecolor='w', edgecolor='k')

    #plt.figure(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    #ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=4)

    fig, ax1 = plt.subplots(num=None, figsize=(18, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(f[0:l_mitad] , SPTRM_P[0:l_mitad],'b')
    #plt.plot(f[indexes]   , SPTRM_P[indexes]  ,'o')

    for counter,k in enumerate(harm.columns):

        if k[0] == 'i':
            #print (counter,k)
            index = int(harm.iloc[captura][counter])
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
def Failures(df_in):
    print('Unbalance Failure')
    n_traces   = df_in.shape[0]
    none_list  = []
    E1         = 0.15*np.sqrt(2)
    for i in range (n_traces):
        none_list.append('None')

    df_in['Clearance Failure'] = none_list
    df_in['Unbalance Failure'] = none_list
    df_in['Oil Whirl Failure'] = none_list
    df_in['Oil Whip Failure']  = none_list
    df_in['Blade Faults Failure'] = none_list
    df_in['Flow Turbulence Failure'] = none_list
    df_in['PBB looseness Failure'] = none_list
    df_in['Shaft Mis. Failure'] = none_list
    df_in['Pressure P. Failure'] = none_list
    df_in['Surge E. Failure'] = none_list
    df_in['Severe Mis. Failure'] = none_list
    df_in['Loose Bedplate Failure'] = none_list

    for i in range (n_traces):
        #=====================================================Clearance Failure
        
        v_1x   = df_in.iloc[i]['E 1.0']
        v_2x   = df_in.iloc[i]['E 2.0']
        v_3x   = df_in.iloc[i]['E 3.0']

        v_0_5x = df_in.iloc[i]['E 1/2']
        v_1_5x = df_in.iloc[i]['E 3/2']
        v_2_5x = df_in.iloc[i]['E 5/2']
                                            #-------1.0x 2.0x 3.0x decreciente
        bool1 =  v_1x >v_2x > v_3x
                                            # --2.0x >2% 1.0x and 3.0x >2% 1.0x
        bool2 = (v_2x > 0.02 * v_1x) and (v_3x > 0.02 * v_1x)
                                            #-------0.5x 1.5x 2.5x decreciente
        bool3 =  v_0_5x > v_1_5x > v_2_5x
                                            # ------0.5x >2% 1.0x and 1.5x > 2% 1.0x and 2.5x > 2% 1.0x
        bool4 = (v_0_5x > 0.02 * v_1x) and (v_1_5x > 0.02 * v_1x) and (v_2_5x > 0.02 * v_1x)

        #print (bool1,bool2,bool3,bool4)
        A = bool1 and bool2
        B = bool3 and bool4

        if (A == False) and (B == False):
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Green'
        if A or B:
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'Clearance Failure'] = 'Red'
        #=====================================================Unbalance Failure
        
        #print(df_in.iloc[i].values[1:8])
                                            # max armonicos = 1.0x
        f_max1 = df_in.iloc[i]['E 1.0']
                                            # max del resto de pikos
        s_max1 = df_in.iloc[i]['E 2nd Highest']
                                            #---1X meno que el umbral
        A  = f_max1        < 4
                                            #---El 15% 1x < resto armonicos.
                                            #   es decir 1X no es dominante
        B  = f_max1 * 0.15 < s_max1
                                            #--------------------------Green
        if A and B:
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'
                                            #--------------------------yellow
                                            #   Xor = cualquiera de ellas
                                            #        pero no ambas
        if (A == False) ^   (B == False):
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Yellow'
                                            #--------------------------Red
                                            # las dos falsas
        if (A == False) and (B == False):
            df_in.loc[df_in.index[i],'Unbalance Failure'] = 'Green'


        #=====================================================Oil Whirl Failure
        
                                            #-----------green-----------------
                                            # no detected Peak in '0.38-0.48'
        if df_in.iloc[i]['E Oil Whirl'] < E1:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Green'
                                            #-----------yellow-----------------
                                            # Detected Peak in '0.38-0.48'
                                            #         but
                                            # Peak in '0.38-0.48' < 2% 1.0x
        if df_in.iloc[i]['E Oil Whirl'] > E1 and df_in.iloc[i]['E Oil Whirl'] < 0.02 * df_in.iloc[i]['E 1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Yellow'
                                            #-----------red--------------------
                                            # Peak in '0.38-0.48' > 2% 1.0x
        if df_in.iloc[i]['E Oil Whirl'] > E1 and df_in.iloc[i]['E Oil Whirl'] > 0.02 * df_in.iloc[i]['E 1.0']:
            df_in.loc[df_in.index[i],'Oil Whirl Failure'] = 'Red'

        #======================================================Oil Whip Failure-------------------------
        
        A = (df_in.iloc[i]['E 1/2'] >= E1 and df_in.iloc[i]['BW 1/2'] >= 4)  and  ((df_in.iloc[i]['E 5/2'] >= E1) and df_in.iloc[i]['BW 2.5'] >= 4)
        B = df_in.iloc[i]['E 1/2' ] > 0.02 *  df_in.iloc[i]['E 1.0']
        C = df_in.iloc[i]['E 5/2' ] > 0.02 *  df_in.iloc[i]['E 1.0']
        #print(A,B,C)
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
        if A ^ ((B ^ C)) :
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Yellow'
                                             #-----------red-------------------
                                             #     2H BW at 0.5 >2% 1.0x
                                             #           AND
                                             #     2H BW at 2.5 >2% 1.0x
        if A and B and C:
            df_in.loc[df_in.index[i],'Oil Whip Failure'] = 'Red'

        #print('oild whip failure',df_in.loc[df_in.index[i],'Oil Whip Failure'])
        
        #==================================================Blade Faults Failure
        
        A = df_in.iloc[i]['E 12.0'] > E1
        B = df_in.iloc[i]['E 12.0'] > E1 and df_in.iloc[i]['E 24.0'] > E1
        C = df_in.iloc[i]['E 12.0'] > E1 and (df_in.iloc[i]['E 11.0'] > E1 or df_in.iloc[i]['E 13.0'] > E1)
        D = C and df_in.iloc[i]['E 24.0'] > E1
        E = C and df_in.iloc[i]['E 24.0'] > E1 and (df_in.iloc[i]['E 23.0'] > E1 or df_in.iloc[i]['E 25.0'] > E1)
        F = df_in.iloc[i]['E 12.0'] < E1 and df_in.iloc[i]['E 24.0'] < E1
        #print('Blade Faults         ',A,B,C,D,E,F)
                                             #  Tabla de verdad progresiva
                                             #  puede empezar siendo verde,
                                             #  acabar siendo rojo
        if A or B or F:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Green'
        if C or D:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Yellow'
        if E:
            df_in.loc[df_in.index[i],'Blade Faults Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Blade Faults'] )

        #===============================================Flow Turbulence Failure
        
        A =        df_in.iloc[i]['E Flow T.'] <= 0.2
        B = 0.2 <= df_in.iloc[i]['E Flow T.'] <= df_in.iloc[i]['E 1.0']
        C =        df_in.iloc[i]['E Flow T.'] >  df_in.iloc[i]['E 1.0']
        #print('Flow Tur.           ',A,B,C)
        if A:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Green'
        if B:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Yellow'
        if C:
            df_in.loc[df_in.index[i],'Flow Turbulence Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Flow Turbulence'] )

        #=================================================PBB looseness Failure
        
        A = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and df_in.iloc[i]['E 3.0'] > E1 and df_in.iloc[i]['E 1.0'] <df_in.iloc[i]['E 2.0'] > df_in.iloc[i]['E 3.0']
        B = df_in.iloc[i]['E 1/2'] > E1 and df_in.iloc[i]['E 1/3'] > E1 and df_in.iloc[i]['E 1/4'] > E1
        #print('Plain Bearin block   ',A,B)
        if not A and not B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Green'
        if A or B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'PBB looseness Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'PBB looseness'])
        #====================================================Shaft Mis. Failure
        
        A = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and      df_in.iloc[i]['E 2.0'] < 0.5 *  df_in.iloc[i]['E 1.0']
        B = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and 1.5 *df_in.iloc[i]['E 1.0'] >        df_in.iloc[i]['E 2.0'] > 0.5 *df_in.iloc[i]['E 1.0']
        C = df_in.iloc[i]['E 1.0'] > E1 and df_in.iloc[i]['E 2.0'] > E1 and 1.5 *df_in.iloc[i]['E 1.0'] <        df_in.iloc[i]['E 2.0']
        D = df_in.iloc[i]['E 2.0'] > E1 and df_in.iloc[i]['E 3.0'] > E1 and      df_in.iloc[i]['E 4.0'] > E1 and df_in.iloc[i]['E 5.0'] > E1
        #print('Shaft miss          ',A,B,C,D)
        if A or not D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Green'
        if B and D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Yellow'
        if C and D:
            df_in.loc[df_in.index[i],'Shaft Mis. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])
        #===================================================Pressure P. Failure
        
        A = df_in.iloc[i]['E 1/3'] > E1 and df_in.iloc[i]['E 2/3'] > E1 and df_in.iloc[i]['E 4/3'] > E1 and df_in.iloc[i]['E 5/3'] > E1
        B = (df_in.iloc[i]['E 4/3'] > df_in.iloc[i]['E 1/3']) and (df_in.iloc[i]['E 8/3'] > df_in.iloc[i]['E 1/3']) and (df_in.iloc[i]['E 4.0'] > df_in.iloc[i]['E 1/3'])
        #print('Shaft miss          ',A,B,C,D)
        if not A:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Green'
        if A:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'Pressure P. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])

        #======================================================Surge E. Failure
        
        A = df_in.iloc[i]['E Surge E. 0.33x 0.5x'] > E1
        B = df_in.iloc[i]['E Surge E. 12/20k'] > E1

        if not A:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Green'
        if A:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Yellow'
        if A and B:
            df_in.loc[df_in.index[i],'Surge E. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])

        #===================================================Severe Mis. Failure
        
        counter_A = 0
        for m in [2,3,4,5,6,7,8,9,10]:
            if df_in.iloc[i]['E '+str(m)+'.0'] > 0.02*df_in.iloc[i]['E 1.0']:
                counter_A = counter_A+1
        A = counter_A >= 3
        counter_B = 0
        for m in ['5/2','7/2','9/2','11/2','13/2','15/2','17/2','19/2']:
            if df_in.iloc[i]['E '+m] > 0.02*df_in.iloc[i]['E 1.0']:
                counter_B = counter_B+1
        B = counter_B >= 3
        #print ('Counter A in Severe Mis. Failure',counter_A,A)
        #print ('Counter B in Severe Mis. Failure',counter_B,B)
        C = df_in.iloc[i]['E 2.0'] > df_in.iloc[i]['E 1.0']

        if not A:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Green'
        if A ^ B:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Yellow'
        if A and B and C:
            df_in.loc[df_in.index[i],'Severe Mis. Failure'] = 'Red'
        #print(df_in.loc[df_in.index[i],'Shaft Mis.'])

        #================================================Loose Bedplate Failure
       
        A = 2.5 * np.sqrt(2) > df_in.iloc[i]['E 1.0'] > 0
        B = 2.5 * np.sqrt(2) < df_in.iloc[i]['E 1.0'] < 6.00 * np.sqrt(2)
        C = 6.0 * np.sqrt(2) < df_in.iloc[i]['E 1.0']
        D = df_in.iloc[i]['E 3.0'] > df_in.iloc[i]['E 2.0']
        #print ('Loose Bedplate',A,B,C,D)
        if A:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Green'
        if B ^ C:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Yellow'
        if C and D:
            df_in.loc[df_in.index[i],'Loose Bedplate Failure'] = 'Red'
#        print(df_in.iloc[i]['E 1.0']/np.sqrt(2),A,B,C)
#        print(D)
#        print('________________',df_in.loc[df_in.index[i],'Loose Bedplate Failure'])
    return df_in
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    parameters = {
        'IdPlanta': 'BPT',
        'IdAsset': 'H4-FA-0002',
        'Fecha':       '2018-10-12T00:52:46.9988564Z',
        'FechaInicio': '2018-10-12T00:52:46.9988564Z',
        'NumeroTramas': '14',
        'Parametros': 'waveform'
    }

    pi        = np.pi

    path      = '/Users/inigo/Desktop/petronor'
    path      = 'C:/OPG106300/TRABAJO/Proyectos/Petronor-075879.1 T 20000/Trabajo/python'
    # extract month from key 'Fecha' in parameters object
    month     = parameters['Fecha'].split('-')[1]
    # extract day from key 'Fecha' in parameters object
    day       = parameters['Fecha'].split('-')[2].split('T')[0]
    hour      = parameters['Fecha'].split('-')[2].split('T')[1]

    #when      = '/' +month + '/' +day + '/' + hour
    #when_l    = month+'_'+day+'_'+hour
    #path      = path + when
    fs        = 5120.0
    maquina   = parameters['IdAsset']
    localizacion     = 'SH4' #SH3/4

    # construct API endpoint
    api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
    '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
    '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])

    # make GET request to API endpoint
    response = requests.get(api_endpoint)
    # convert response from server into json object
    response_json = response.json()

    # check if server is up or not. They usually restart the server every day...
    # check if there are Nones in 'ValoresTrama'
    flag = response_json[0]['waveform'][0]['ValoresTrama']
    # if there are Nones
    if None in flag:
        print("Server is down...")
    # if there are not, run script functions
    else:

        assetId = parameters['IdAsset']
        numTramas = parameters['NumeroTramas']
        df_accel,l_devices = load_vibrationData(response_json, localizacion, numTramas, assetId)

        df_SPEED_abs,df_SPEED_angle,df_speed_f = velocity(df_accel)
        
 #       df_speed_f.to_pickle('speed_f_' + maquina + '_' + localizacion + '_' + when_l + '.pkl')
 #       df_SPEED.to_pickle  ('SPEED_'   + maquina + '_' + localizacion + '_' + when_l + '.pkl')

        harm    = df_Harmonics(df_SPEED_abs, fs)
        start = time. time()
       
        harm     = df_unbalance(harm)
        harm     = df_clearance(harm)
        harm     = oil_whirl(harm)
        harm     = oil_whip(harm)
        harm     = Blade_faults(harm)
        harm     = flow_turbulence(harm)
        harm     = Plain_bearing_block_looseness(harm)
        harm     = SHAFT_MISALIGNMENTS(harm)
        harm     = Pressure_Pulsations(harm)
        harm     = Surge_Effect(harm)
        harm     = Severe_Misaligment(harm)
        harm     = Loose_Bedplate(harm)
       
       # harm     = Failures(harm)

        end = time. time()
        print(end - start)

        writer = pd.ExcelWriter('result_'+maquina+'_'+localizacion+'_'+month+'_'+day+'.xlsx')
        harm.to_excel(writer, 'DataFrame')
        writer.save()

        find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,df_speed_f,harm)

        #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak')
        #color,vertices = plot_waterfall(df_SPEED,fs,0,400)
        plot_waterfall2(df_SPEED_abs,fs,0,400)

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FINNNNNNNNNNNNNNNN')

        ####POST

        # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
        #requests.post('/api/Models/SetResultModel', output=OUTPUT)
