
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

#------------------------------------------------------------------------------
def Load_Vibration_Data_Global(parameters):
    f_fin   = 300
    i_f_fin = int(l*f_fin/fs)
    f       = np.arange(l)/(l-1)*fs
    f       = f[0:i_f_fin]
    df_freq = pd.DataFrame([f],columns=np.arange(i_f_fin))
    
    df_speed = 0
    df_SPEED = 0
    if parameters['Source'] == 'Petronor Server': 
        print('------------------------------------Accediendo servidor Petronor')
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
        
        df = Load_Vibration_Data_From_Get(response_json, parameters)   
            
            

    return df


#------------------------------------------------------------------------------

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
    if input_data[0]['waveform'][0]['IdPosicion'] == parameters['Localizacion']:
            flag_pos = 0
    if input_data[0]['waveform'][1]['IdPosicion'] == parameters['Localizacion']:
            flag_pos = 1
    print('--------------------', input_data[0]['waveform'][flag_pos]['IdPosicion'])
    #frames = input_data[0]['waveform'][flag_pos]['ValoresTrama']
    
   
    frames =[]
   
    for counter, trama in enumerate(input_data[0]['waveform'][flag_pos]['ValoresTrama']):

        if trama != None:
            print (counter,'bieeeeeeeeeeeeen')
            frames.append(trama)

    
    """
    for values_trama in frames:
        

            res = pd.DataFrame.from_dict(values_trama, orient='index')
            res = res.transpose()

            word = str(res.AssetId.values[0])+' '+str(res.MeasurePointId.values[0])
            if (word in lista_maquinas) == False:
                lista_maquinas.append( word )

            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == MeasurePointId:
                #print(root)
                print(res.Props.iloc[0][0]['Value'])
                print(res.Props.iloc[0][4]['Value'])
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
    """
    return  frames

    

#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0001',
        'Localizacion' : 'SH4', #SH4/MH2
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-21T18:00:00.9988564Z',
        'FechaInicio'  : '2018-10-12T00:52:46.9988564Z',
        'NumeroTramas' : '100',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '',#'12'
        'Hour'         : ''    
    }

   
    dd = Load_Vibration_Data_Global(parameters)
    

    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FINNNNNNNNNNNNNNNN')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
