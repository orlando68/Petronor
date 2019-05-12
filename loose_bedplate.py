import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#-----------esto es una prueba


#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
def PK(a):
    if a > E1:
        out = True
    else:
        out = False
    return out


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
    
    
    pi        = np.pi
    E1        = 0.15
    
    fs                    = 5120
    l                     = 16384
    l_2                   = np.int(l/2)
    t                     = np.arange(l)/fs
    f                     = np.arange(l)/(l-1)*fs
    
    
    df_speed,df_SPEED_abs = Load_Vibration_Data_Global(parameters)
    spectrum              = np.fft.fft(df_speed.iloc[0])/l
    df_SPEED_abs.iloc[0]  = np.abs(spectrum)        # sin ventana de hanning
    harm                  = df_Harmonics(df_speed,df_SPEED_abs, fs,'blower')
    spec_rand = np.copy(spectrum)
    
    
    inic_1x   = int(harm.iloc[0]['n_s 1.0'])
    fin_1x    = int(harm.iloc[0]['n_e 1.0'])
    power_1x  = harm.iloc[0]['RMS 1.0']
    
    inic_2x   = int(harm.iloc[0]['n_s 2.0'])
    fin_2x    = int(harm.iloc[0]['n_e 2.0'])
    power_2x  = harm.iloc[0]['RMS 2.0']
    
    inic_3x   = int(harm.iloc[0]['n_s 3.0'])
    fin_3x    = int(harm.iloc[0]['n_e 3.0'])
    power_3x  = harm.iloc[0]['RMS 3.0']
    
    n_random  = 10
    columnas   = ['1x Good','2x Good','3x Good',
                  '1x Satisfactory','2x Satisfactory','3x Satisfactory',
                  '1x Unacceptable','2x Unacceptable','3x Unacceptable']
    df_random  = pd.DataFrame(index = range(n_random), columns = columnas, data = np.zeros((n_random,9)) )
    #df_random  = pd.DataFrame(index = range(n_random), columns = columnas, data = np.ones((n_random,9)) )
                  #--DataFrame con todos los valores de amplitud aleatorios válidos
    l1 = 0
    l2 = 0
    l3 = 0
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
        
        dice = 10*np.abs(np.random.randn(3))  #----------------lanzamos el dado
        x1   = dice[0]
        x2   = dice[1]
        x3   = dice[2]
        #print (dice)
        A = 0 < x1 < 0.71 
        B = 0.71 < x1 < 1.8
        C = 1.8 < x1
        D = (PK(x2) and PK(x3)) and x3 > x2
        
        if A and l1 < n_random:#-----------------------------------Good signals
            df_random.iloc[l1]['1x Good'] = x1
            df_random.iloc[l1]['2x Good'] = x2
            df_random.iloc[l1]['3x Good'] = x3
            l1= l1+1
            #-------from now on  acceptable
        if (B ^ C) and l2 < n_random:#-----------------------Satisfactory signals
            df_random.iloc[l2]['1x Satisfactory'] = x1
            df_random.iloc[l2]['2x Satisfactory'] = x2
            df_random.iloc[l2]['3x Satisfactory'] = x3
            l2= l2+1
        if (C and D) and l3 < n_random: #--------------------Unacceptable signals
            df_random.iloc[l3]['1x Unacceptable'] = x1
            df_random.iloc[l3]['2x Unacceptable'] = x2
            df_random.iloc[l3]['3x Unacceptable'] = x3
            l3= l3+1
         
        if l1 == n_random and l2 == n_random and l3 == n_random:
            print(l1,l2,l3)
            break
          
    
    for k in range(n_random): #----- IFFT de cada una de las señales sinteticas
        
        #----------------------------------------------------------Good signals
        fact_1x = df_random.iloc[k]['1x Good']/power_1x
        spec_rand[inic_1x:fin_1x]         = fact_1x          * spectrum[inic_1x:fin_1x]
        spec_rand[l-fin_1x+1:l-inic_1x+1] = np.conj(fact_1x) * spectrum[l-fin_1x+1:l-inic_1x+1]
        
        fact_2x = df_random.iloc[k]['2x Good']/power_2x
        spec_rand[inic_2x:fin_2x]         = fact_2x          * spectrum[inic_2x:fin_2x]
        spec_rand[l-fin_2x+1:l-inic_2x+1] = np.conj(fact_2x) * spectrum[l-fin_2x+1:l-inic_2x+1]
        
        fact_3x = df_random.iloc[k]['2x Good']/power_3x
        spec_rand[inic_3x:fin_3x]         = fact_3x          * spectrum[inic_3x:fin_3x]
        spec_rand[l-fin_3x+1:l-inic_3x+1] = np.conj(fact_3x) * spectrum[l-fin_3x+1:l-inic_3x+1]
        
        signal_math = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal
        signal = np.real(signal_math[2600:13500])
        #--------------------------------------------------Satisfactory signals
        fact_1x = df_random.iloc[k]['1x Satisfactory']/power_1x
        spec_rand[inic_1x:fin_1x]         = fact_1x          * spectrum[inic_1x:fin_1x]
        spec_rand[l-fin_1x+1:l-inic_1x+1] = np.conj(fact_1x) * spectrum[l-fin_1x+1:l-inic_1x+1]
        
        fact_2x = df_random.iloc[k]['2x Satisfactory']/power_2x
        spec_rand[inic_2x:fin_2x]         = fact_2x          * spectrum[inic_2x:fin_2x]
        spec_rand[l-fin_2x+1:l-inic_2x+1] = np.conj(fact_2x) * spectrum[l-fin_2x+1:l-inic_2x+1]
        
        fact_3x = df_random.iloc[k]['2x Satisfactory']/power_3x
        spec_rand[inic_3x:fin_3x]         = fact_3x          * spectrum[inic_3x:fin_3x]
        spec_rand[l-fin_3x+1:l-inic_3x+1] = np.conj(fact_3x) * spectrum[l-fin_3x+1:l-inic_3x+1]
    
        signal_math = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal
        signal = np.real(signal_math[2600:13500])
        
        #--------------------------------------------------Unacceptable signals
        fact_1x = df_random.iloc[k]['1x Unacceptable']/power_1x
        spec_rand[inic_1x:fin_1x]         = fact_1x          * spectrum[inic_1x:fin_1x]
        spec_rand[l-fin_1x+1:l-inic_1x+1] = np.conj(fact_1x) * spectrum[l-fin_1x+1:l-inic_1x+1]
        
        fact_2x = df_random.iloc[k]['2x Unacceptable']/power_2x
        spec_rand[inic_2x:fin_2x]         = fact_2x          * spectrum[inic_2x:fin_2x]
        spec_rand[l-fin_2x+1:l-inic_2x+1] = np.conj(fact_2x) * spectrum[l-fin_2x+1:l-inic_2x+1]
        
        fact_3x = df_random.iloc[k]['2x Unacceptable']/power_3x
        spec_rand[inic_3x:fin_3x]         = fact_3x          * spectrum[inic_3x:fin_3x]
        spec_rand[l-fin_3x+1:l-inic_3x+1] = np.conj(fact_3x) * spectrum[l-fin_3x+1:l-inic_3x+1]
    
        signal_math = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal
        #signal = np.real(signal_math[2600:13500])
        signal = np.real(signal_math)

    print (np.max(np.imag(signal)))
    plt.figure()
    plt.subplot(211)
    plt.plot(f,np.abs(spectrum),f,np.abs(spec_rand))
    plt.plot(f[inic_1x:fin_1x],np.abs(spectrum[inic_1x:fin_1x]),'o')
    plt.subplot(212)
    #plt.plot(t,df_speed.iloc[0])
    plt.plot(np.real(signal))
    plt.show()
    
    
    
    



