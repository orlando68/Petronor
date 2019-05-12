import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal


#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
def PK(a):
    if a > E1:
        out = True
    else:
        out = False
    return out

def Synth_Peak(fc,BW,data_in):
    
    step = 0.01
    
    n_points = int(np.ceil((l*BW/fs)))
    if np.mod(n_points,2) == 0:
        n_points = n_points+1
    
    n = int(n_points/2)
    lado = np.arange(1,n)
    peak_array = np.array([0])
    peak_array = np.append(peak_array,lado)
    peak_array = np.append(peak_array,n)
    peak_array = np.append(peak_array,lado[::-1])
    peak_array = np.append(peak_array,0)
    peak_array = 0*500*step*np.random.rand(np.size(peak_array)) + peak_array
    
    Veff_peak  = np.sqrt(np.sum(peak_array**2))
    peak_array = peak_array / Veff_peak    
    
    i_cen      = int(l*fc/fs)
    ini        = i_cen - n+1
    fin        = i_cen + n+1 +1
    
    spectrum   = np.fft.fft(data_in)/l
    spec_rand  = np.copy(spectrum)

   
    spec_rand[ini:fin]         = 0.8 * peak_array
    spec_rand[l-fin+1:l-ini+1] = 0.8 * peak_array[::-1]
    
    signal_math = l*np.fft.ifft(spec_rand)

    if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
        print('Cuidado señal no valida!!!!!!!!!!!!!!!')
        #----espectro de la señal sintetica => spec_rand
        #----señal sintetica en el tiempo   => signal
    data_out = np.real(signal_math)
    
    
    return data_out,spec_rand

def Synth_feature(fc,BW,level,window,data_in):
    
    step = 0.01
    
    n_points = int(np.ceil((l*BW/fs)))
    
    
    if np.mod(n_points,2) == 0:
        n_points = n_points+1
    
    n = int(n_points/2)
    if window == 'triangle':
        peak_array = signal.windows.triang(n_points)
    if window == 'boxcar':
        peak_array = signal.windows.boxcar(n_points)
    if window == 'gaussian':
        peak_array = signal.windows.gaussian(n_points,7)
    if window == 'tukey':
        peak_array = signal.windows.tukey(n_points)
    if window == 'dirac':
        peak_array = np.zeros(n_points)
        peak_array[n]=1
        
    peak_array = 1*50*step*np.random.rand(np.size(peak_array)) + peak_array
    
    Veff_peak  = np.sqrt(np.sum(peak_array**2))
    peak_array = peak_array / Veff_peak    
    
    i_cen      = int(l*fc/fs)
    ini        = i_cen - n+1
    fin        = i_cen + n+1 +1
    
    spectrum   = np.fft.fft(data_in)/l
    spec_rand  = np.copy(spectrum)

   
    spec_rand[ini:fin]         = level * peak_array
    spec_rand[l-fin+1:l-ini+1] = level * peak_array[::-1]
    
    signal_math = l*np.fft.ifft(spec_rand)

    if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
        print('Cuidado señal no valida!!!!!!!!!!!!!!!')
        #----espectro de la señal sintetica => spec_rand
        #----señal sintetica en el tiempo   => signal
    data_out = np.real(signal_math)
    
    
    return data_out,spec_rand



def Shift_Noise(fc,BW,data_in,level):

    ini        = int(l*(fc-BW/2)/fs)
    fin        = int(l*(fc+BW/2)/fs)
    
    spectrum   = np.fft.fft(data_in)/l
    
    Veff_peak  = np.sqrt(np.sum(spectrum[ini:fin+1]**2))
    
    factor     = level/Veff_peak
    spec_rand  = np.copy(spectrum)

   
    spec_rand[ini:fin]         = factor * spectrum[ini:fin]
    spec_rand[l-fin+1:l-ini+1] = factor * spectrum[l-fin+1:l-ini+1]
    
    signal_math = l*np.fft.ifft(spec_rand)

    if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
        print('Cuidado señal no valida!!!!!!!!!!!!!!!')
        #----espectro de la señal sintetica => spec_rand
        #----señal sintetica en el tiempo   => signal
    data_out = np.real(signal_math)
    
    
    return data_out,spec_rand

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
    harm                  = df_Harmonics(df_speed,df_SPEED_abs, fs)
    #signal,spec_rand = Synth_Peak(9,5,df_speed.iloc[0])
    signal,spec_rand = Synth_feature(1000,5,0.1,'tukey',df_speed.iloc[0])
    #signal,spec_rand = Shift_Noise(9,5,df_speed.iloc[0],0.3)
    

    print (np.max(np.imag(signal)))
    plt.figure()
    plt.subplot(211)
    plt.plot(f,np.abs(spectrum),f,np.abs(spec_rand))
    #plt.plot(f[ini:fin],np.abs(spectrum[ini:fin]),'o')
    plt.subplot(212)
    #plt.plot(t,df_speed.iloc[0])
    plt.plot(np.real(signal))
    plt.show()
    
    
    
    



