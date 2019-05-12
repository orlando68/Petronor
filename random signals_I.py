import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
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
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '11',
        'Day'          : '',#'12'
        'Hour'         : '' 
    }

    fs                    = 5120
    l                     = 16384
    l_2                   = np.int(l/2)
    t                     = np.arange(l)/fs
    f                     = np.arange(l)/(l-1)*fs
    df_speed,df_SPEED_abs = Load_Vibration_Data_Global(parameters)
    
    spectrum              = np.fft.fft(df_speed.iloc[0])/l
    df_SPEED_abs.iloc[0]  = np.abs(spectrum)        # sin ventana de hanning
    harm                  = df_Harmonics(df_speed,df_SPEED_abs, fs)
    
    
    spec_rand = np.copy(spectrum)
    picos     = np.array([4.5,6.5,4])
    i = 0
    for counter,k in enumerate(harm.columns):

        if k == 'i 1.0':
            
            index  = int(harm.iloc[0][counter  ])
            inic   = int(harm.iloc[0][counter+5])
            fin    = int(harm.iloc[0][counter+6])
            power  = harm.iloc[0][counter+2]
            
            factor = 5/power 
#            print(np.sqrt(2*np.sum(np.abs(spectrum[inic:fin])**2)))
#            print(counter,power,'mm/s')
#            print(np.abs(spectrum[inic:fin]))
#            print(np.abs(spectrum[l-fin+1:l-inic+1]))
            spec_rand[inic:fin]         = factor          * spectrum[inic:fin]
            spec_rand[l-fin+1:l-inic+1] = np.conj(factor) * spectrum[l-fin+1:l-inic+1]
          
    
    signal = l*np.fft.ifft(spec_rand)
    
    print (np.max(np.imag(signal)))
    plt.figure()
    plt.subplot(211)
    plt.plot(f,np.abs(spectrum),f,np.abs(spec_rand))
    plt.plot(f[inic:fin],np.abs(spectrum[inic:fin]),'o')
    plt.subplot(212)
    plt.plot(t,df_speed.iloc[0],t,np.real(signal))
    plt.show()
    
    
    



