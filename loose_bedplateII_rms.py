import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.stats import kurtosis
from scipy import stats
import matplotlib
#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
def PK(a):
    if a > E1:
        out = True
    else:
        out = False
    return out

def Wnl(X):
        
        params = stats.exponweib.fit(X, floc=0, f0=1)
        #print (params)
        shape = params[1]
        scale = params[3] 
        
        weibull_pdf = (shape / scale) * (X / scale)**(shape-1) * np.exp(-(X/scale)**shape)
        return -np.nansum(np.log(weibull_pdf))

def Wnl_new(X):
        #X = X[np.logical_and(X>=0,X>=0)]
        X = np.abs(X)
        params = stats.exponweib.fit(X, floc=0, f0=1)
        #print (params)
        shape = params[1]
        scale = params[3] 
        
        weibull_pdf = (shape / scale) * (X / scale)**(shape-1) * np.exp(-(X/scale)**shape)
        return -np.nansum(np.log(weibull_pdf))
def Nnl(X):
        mean = np.mean(X)
        std  = np.std(X)
        normal_pdf = np.exp(-(X-mean)**2/(2*std**2)) / (std*np.sqrt(2*np.pi))
        return -np.nansum(np.log(normal_pdf))
    

def entropy(X):
        # remove nans
        #X.dropna(inplace=True)
        X = np.abs(X)
        pX = X / X.sum()
        #print ('pX=',pX)
        return -np.nansum(pX*np.log2(pX))
         
    


if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Localizacion' : 'SH4', #SH3/4
        'Source'       : 'Local Database', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-20T00:20:00.9988564Z',
        'FechaInicio'  : '2019-02-12T00:52:46.9988564Z',
        'NumeroTramas' : '1',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '12',#'12'
        'Hour'         : '' 
    }
    
    
    pi                    = np.pi
    E1                    = 0.15
    
    fs                    = 5120
    l                     = 16384
    l_2                   = np.int(l/2)
    t                     = np.arange(l)/fs
    f                     = np.arange(l)/(l-1)*fs
    A_noise = 0*0.8
    n_random  = 100 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    start = 0;    end   = l
    start = 2600;    end   = 13500
    length = end-start
    
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
    harm              = df_Harmonics(df_speed,df_SPEED, fs,'blower')
    harm              = Loose_Bedplate(harm)
    
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    
    for k in range(np.size(df_speed.index)):
        #print (k)
        if harm.iloc[k]['$Loose Bedplate Failure'] == 'Green':
            color = 'g'
        if harm.iloc[k]['$Loose Bedplate Failure'] == 'Yellow':
            color = 'y'
        if harm.iloc[k]['$Loose Bedplate Failure'] == 'Red':
            color = 'r'
        rms = np.std(df_speed.iloc[k].values[start:end])
        #rms = 1
        signal_real = df_speed.iloc[k].values[start:end] / rms
#        ax.scatter(stats.kurtosis(np.abs(signal_real),fisher = False),
#                   Wnl_new       (       signal_real),
#                   entropy       (       signal_real) , facecolors='none',edgecolor = color,marker='o'
#                   )
        ax.scatter(Nnl(signal_real),
                   Wnl_new(signal_real),
                   entropy(signal_real) , facecolors='none',edgecolor = color,marker='o'
                   )        
        
    
    
    spectrum              = np.fft.fft(df_speed.iloc[0].values)/l #-----me quedo con la primera 
    plt.figure(2)
    plt.title('real')
    n, bins, patches = plt.hist(x=df_speed.iloc[0].values, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    #print('>>>>>>>>>>>>>>>',stats.kurtosis(df_speed.iloc[0]),stats.kurtosis(df_speed.iloc[0],fisher = False))
    #df_SPEED_abs.iloc[0]  = np.abs(spectrum)        # sin ventana de hanning
    #harm                  = df_Harmonics(df_speed,df_SPEED_abs, fs,'blower')
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
    
   
  
    columnas  = ['1x Good','2x Good','3x Good','kurtosis_G','skewness_G','Wnl_G','entropy_G',
                 '1x Acceptable','2x Acceptable','3x Acceptable','kurtosis_A','skewness_A','Wnl_A','entropy_A',
                 '1x Unacceptable','2x Unacceptable','3x Unacceptable','kurtosis_U','skewness_U','Wnl_U','entropy_U'
                ]
    df_random = pd.DataFrame(index = range(n_random), columns = columnas, data = np.zeros((n_random,np.size(columnas))) )
    
    df_sign_G = pd.DataFrame(index = range(n_random), columns = range(length), data = np.zeros((n_random,length) ))
    df_sign_A = pd.DataFrame(index = range(n_random), columns = range(length), data = np.zeros((n_random,length) ))
    df_sign_U = pd.DataFrame(index = range(n_random), columns = range(length), data = np.zeros((n_random,length) ))
    #df_random  = pd.DataFrame(index = range(n_random), columns = columnas, data = np.ones((n_random,9)) )
                  #--DataFrame con todos los valores de amplitud aleatorios válidos
    l1       = 0
    l2       = 0
    l3       = 0
    #-------------------------------------------------NORMAL-------------------
    mean_1x  = 4.8
    std_1x   = 2
    
    mean_2x  = mean_3x = 0.9
    std_2x   = std_3x  = 0.50
    
    #-------------------------------------------------LOGNORMAL----------------
    lmean_1x =  2 * np.log(mean_1x) - np.log(mean_1x**2+std_1x**2)/2
    lstd_1x  = np.sqrt(-2 * np.log(mean_1x) + np.log(mean_1x**2+std_1x**2))

    lmean_2x = lmean_3x = 2 * np.log(mean_2x) - np.log(mean_2x**2+std_2x**2)/2
    lstd_2x  = lstd_3x  = np.sqrt(-2 * np.log(mean_2x) + np.log(mean_2x**2+std_2x**2))

       
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
        
           #----------------lanzamos el dado
#--------------------------------------------------NORMAL----------------------        
        x1   = np.abs(mean_1x  + std_1x * np.random.randn(1))
        x2   = np.abs(mean_2x  + std_2x * np.random.randn(1))
        x3   = np.abs(mean_3x  + std_3x * np.random.randn(1))
           
#-------------------------------------------------LOGNORMAL--------------------          
#        x1   = np.random.lognormal(lmean_1x,std_1x,1)
#        x2   = np.random.lognormal(lmean_2x,std_2x,1)
#        x3   = np.random.lognormal(lmean_3x,std_2x,1)
        
        #print(l1,l2,l3)
        #print(x1,x2,x3)
        print(l1,l2,l3)
        A = 0    < x1 < 0.71 
        B = 0.71 < x1 < 1.8
        C = 3  < x1
        D = (PK(x2) and PK(x3)) and x3 > x2*2
        
        if A and l1 < n_random:#-----------------------------------Good signals
            df_random.iloc[l1]['1x Good']         = x1
            df_random.iloc[l1]['2x Good']         = x2
            df_random.iloc[l1]['3x Good']         = x3
            
            l1= l1+1
            #-------from now on  acceptable
        if (B ^ C) and l2 < n_random:#-----------------------Acceptable signals
            df_random.iloc[l2]['1x Acceptable']   = x1
            df_random.iloc[l2]['2x Acceptable']   = x2
            df_random.iloc[l2]['3x Acceptable']   = x3
            l2= l2+1
        if (C and D) and l3 < n_random: #------------------Unacceptable signals
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
       
        signal = np.real(signal_math[start:end]) + A_noise * np.random.randn(np.size(signal))
        rms = np.std(signal)
        #rms = 1
        signal = signal / rms
        
        df_sign_G.iloc[k]              = signal
        #df_random.iloc[k]['kurtosis_G'] = stats.kurtosis(np.abs(signal),fisher = False)
        df_random.iloc[k]['kurtosis_G'] = Nnl(np.abs(signal))
        df_random.iloc[k]['skewness_G'] = stats.skew(signal)
        df_random.iloc[k]['Wnl_G']      = Wnl_new(signal)
        df_random.iloc[k]['entropy_G']  = entropy(signal)
        #--------------------------------------------------Acceptable signals
        fact_1x = df_random.iloc[k]['1x Acceptable']/power_1x
        spec_rand[inic_1x:fin_1x]         = fact_1x          * spectrum[inic_1x:fin_1x]
        spec_rand[l-fin_1x+1:l-inic_1x+1] = np.conj(fact_1x) * spectrum[l-fin_1x+1:l-inic_1x+1]
        
        fact_2x = df_random.iloc[k]['2x Acceptable']/power_2x
        spec_rand[inic_2x:fin_2x]         = fact_2x          * spectrum[inic_2x:fin_2x]
        spec_rand[l-fin_2x+1:l-inic_2x+1] = np.conj(fact_2x) * spectrum[l-fin_2x+1:l-inic_2x+1]
        
        fact_3x = df_random.iloc[k]['2x Acceptable']/power_3x
        spec_rand[inic_3x:fin_3x]         = fact_3x          * spectrum[inic_3x:fin_3x]
        spec_rand[l-fin_3x+1:l-inic_3x+1] = np.conj(fact_3x) * spectrum[l-fin_3x+1:l-inic_3x+1]
    
        signal_math = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal
            
        signal = np.real(signal_math[start:end]) + A_noise * np.random.randn(np.size(signal))
        rms = np.std(signal)
        #rms = 1
        signal = signal / rms
                                    
        df_sign_A.iloc[k]              = signal
        #df_random.iloc[k]['kurtosis_A'] = stats.kurtosis(np.abs(signal),fisher = False)
        df_random.iloc[k]['kurtosis_A'] = Nnl(signal)
        df_random.iloc[k]['skewness_A'] = stats.skew(signal)
        df_random.iloc[k]['Wnl_A']      = Wnl_new(signal)
        df_random.iloc[k]['entropy_A']  = entropy(signal)
        #print( Wnl(signal), Wnl_new(signal) )
        
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
        signal = np.real(signal_math[start:end]) + A_noise * np.random.randn(np.size(signal))
        rms = np.std(signal)
        #rms = 1
        signal = signal / rms
        
        df_sign_U.iloc[k]               = signal
        #df_random.iloc[k]['kurtosis_U'] = stats.kurtosis(np.abs(signal),fisher = False)
        df_random.iloc[k]['kurtosis_U'] = Nnl(signal)
        df_random.iloc[k]['skewness_U'] = stats.skew(signal)
        df_random.iloc[k]['Wnl_U']      = Wnl_new(signal)
        df_random.iloc[k]['entropy_U']  = entropy(signal)

    """
    plt.figure(10)
    plt.title('Good')
    n, bins, patches = plt.hist(x=df_sign_G.iloc[k].values, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.figure(11)
    plt.title('Acceptable')
    n, bins, patches = plt.hist(x=df_sign_A.iloc[k].values, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    plt.figure(12)
    plt.title('Unacceptable')
    n, bins, patches = plt.hist(x=df_sign_U.iloc[k].values, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    """
    
    
    plt.figure()
    plt.plot(df_speed.iloc[0].values[start:end])
    plt.plot(df_sign_A.iloc[0].values)
    
    plt.show()
        
    for k in range(n_random):
        x = df_random.iloc[k]['kurtosis_G']
        y = df_random.iloc[k]['Wnl_G']
        z = df_random.iloc[k]['entropy_A']
        label = str(format ( df_random.iloc[k]['1x Good'],'.01f'))+' '+str(format ( df_random.iloc[k]['2x Good'],'.01f'))+' '+str(format ( df_random.iloc[k]['3x Good'],'.01f'))
        ax.scatter(x,y,z,c = 'g' )
        #ax.text   (x,y,z,label,fontsize=7)
        #ax.scatter(df_random.iloc[k]['kurtosis_G'],df_random.iloc[k]['Wnl_G'],df_random.iloc[k]['entropy_G'],c = 'g' )
        x = df_random.iloc[k]['kurtosis_A']
        y = df_random.iloc[k]['Wnl_A']
        z = df_random.iloc[k]['entropy_A']
        label = str(format ( df_random.iloc[k]['1x Acceptable'],'.01f'))+' '+str(format ( df_random.iloc[k]['2x Acceptable'],'.01f'))+' '+str(format ( df_random.iloc[k]['3x Acceptable'],'.01f'))
        ax.scatter(x,y,z,c = 'y' )
        #ax.text   (x,y,z,label,fontsize=7)
        #ax.scatter(df_random.iloc[k]['kurtosis_U'],df_random.iloc[k]['Wnl_U'],df_random.iloc[k]['entropy_U'],c = 'r' )
        x = df_random.iloc[k]['kurtosis_U']
        y = df_random.iloc[k]['Wnl_U']
        z = df_random.iloc[k]['entropy_U']
        label = str(format ( df_random.iloc[k]['1x Unacceptable'],'.01f'))+' '+str(format ( df_random.iloc[k]['2x Unacceptable'],'.01f'))+' '+str(format ( df_random.iloc[k]['3x Unacceptable'],'.01f'))
        ax.scatter(x,y,z,c = 'r' )
        #ax.text   (x,y,z,label,fontsize=7)
        
    ax.set_xlabel('kurtosis/Nml')
    ax.set_ylabel('Wnl')
    ax.set_zlabel('entropy')
    plt.show()
       
