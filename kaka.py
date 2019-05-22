import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.stats import kurtosis
from scipy import stats
import matplotlib
from numba import jit
from scipy import signal



class FailureMode:
    def __init__(self,FailureName, df_FingerPrint, Harmonics, TI_signal, SP_SIGNAL,random_specs,template_specs):
        
        self.FailureName         = FailureName
        self.df_FingerPrint      = df_FingerPrint
        self.Harmonics           = Harmonics
        self.TI_signal              = TI_signal
        self.SP_SIGNAL              = SP_SIGNAL
        self.random_specs        = random_specs
        self.template_specs      = template_specs
        
    def __func__(self):
        print('hola')
        if self.FailureName == '$Loose Bedplate Failure':
            fs          = 5120
            df_FingerPrint = Loose_BedPlate(df_SIGNAL)
            
            
def flow(x):
    return np.cos(x)
class data_process:
    def __init__(self,x,y,FailureName, df_FingerPrint, Harmonics, TI_signal, SP_SIGNAL,random_specs,template_specs):
        a  =100
        self.x          = x  *a
        self.y                    = y 
        self.FailureName         = FailureName
        self.df_FingerPrint      = df_FingerPrint
        self.Harmonics           = Harmonics
        self.TI_signal              = TI_signal
        self.SP_SIGNAL              = SP_SIGNAL
        self.random_specs        = random_specs
        self.template_specs      = template_specs         
        print('valor de x',x,self.x)
    def __str__(self):
        s = ''.join(['valor de equis     : ', str(self.x), '\n',
                      str(self.y), '\n',
                     '\n'])       
        print('hola')
        return s
    def __ExtractFP__(self,cte,**kwargs):
        if self.FailureName == '$Loose Bedplate Failure':
            self.df_FingerPrint = Loose_Bedplate(self.df_FingerPrint)
        print ('numero de valoes',self.x*cte+self.y,self.x,cte)
        print (flow(cte),flow(self.x))
        plt.plot(np.random.rand(self.x*cte+self.y))
        plt.show()
        for k in kwargs:
            print(k)
        return 
    def __poco__(self):
        print('hola random')

            

#------------------------------------------------------------------------------
if __name__ == '__main__':        
  
    
    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Localizacion' : 'SH4', #SH3/4
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-20T00:20:00.9988564Z',
        'FechaInicio'  : '2019-02-12T00:52:46.9988564Z',
        'NumeroTramas' : '1',
        'Parametros'   : 'waveform' ,
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '12',#'12'
        'Hour'         : ''
        }
    
    n_random = 100 #---Numeroseñales sintéticas de cada tipo (Red, Green, Yellow)
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
    
    harm         = df_Harmonics(df_SPEED, fs,'blower')
    mia = FailureMode('$Loose Bedplate Failure',harm,['1.0'],df_speed.iloc[0].values,df_SPEED.iloc[0].values,1,1)    


    kk = data_process(1,1,'$Loose Bedplate Failure',harm,['1.0'],df_speed.iloc[0].values,df_SPEED.iloc[0].values,1,1)
    #print(kk.__init__(1,0))
    bb= kk.__str__()
    #print(bb)
    #kk.__init__(100,0)
    kk.__ExtractFP__(10,pepe=300,carlos= 2)
    kk.__poco__

#class FailureMode:
#    def __init__(self, FailureName, df_FingerPrint, Harmonics, signal, SIGNAL, random_specs, template_specs):
#        self.FailureName         = FailureName
#        self.df_FingerPrint      = df_FingerPrint
#        self.Harmonics           = Harmonics
#        self.signal              = signal
#        self.SIGNAL               = SIGNAL
#        self.random_specs        = random_specs
#        self.template_specs      = template_specs
#        
#    def __func__(self):
#        if self.FailureName == '$Loose Bedplate Failure':
#            fs          = 5120
#            df_FingerPrint = Loose_BedPlate(df_SIGNAL)
