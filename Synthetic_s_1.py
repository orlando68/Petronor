import requests
from PETRONOR_lyb import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from scipy.stats import kurtosis
from scipy import stats
import matplotlib
from numba import jit
#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
def PK(a):
    if a > E1:
        out = True
    else:
        out = False
    return out
@jit
def Wnl_o(X):
        
        params = stats.exponweib.fit(X, floc=0, f0=1)
        #print (params)
        shape = params[1]
        scale = params[3] 
        
        weibull_pdf = (shape / scale) * (X / scale)**(shape-1) * np.exp(-(X/scale)**shape)
        return -np.nansum(np.log(weibull_pdf))
@jit
def Wnl(X):
        #X = X[np.logical_and(X>=0,X>=0)]
        X = np.abs(X)
        params = stats.exponweib.fit(X, floc=0, f0=1)
        
        shape = params[1]
        scale = params[3] 
        #print('shape',shape,'scale',scale)
        weibull_pdf = (shape / scale) * (X / scale)**(shape-1) * np.exp(-(X/scale)**shape)
        return -np.nansum(np.log(weibull_pdf))
@jit
def Nnl(X):
        mean = np.mean(X)
        std  = np.std(X)
        normal_pdf = np.exp(-(X-mean)**2/(2*std**2)) / (std*np.sqrt(2*np.pi))
        return -np.nansum(np.log(normal_pdf))
    
@jit
def Entropy_rob(X):
        # remove nans
        #X.dropna(inplace=True)
        X = np.abs(X)
        pX = X / X.sum()
        #print ('pX=',pX)
        return -np.nansum(pX*np.log2(pX))
    
@jit
def Entropy(X):
        # remove nans
        #X.dropna(inplace=True)
        hist, bin_edges = np.histogram(X,bins=100, density=True)
        #6print (np.sum(hist*np.diff(bin_edges)))
        pX = hist*np.diff(bin_edges)
        #print ('pX=',pX)
        return -np.nansum(pX*np.log2(pX))
@jit         
def Rms(x):
    out = x/np.std(x)
    return out

def  decision_table(Bool_A,la,Bool_B,lb,Bool_C,lc,df_in,dice,harmonics,n_reales):
    
#    print (dice)        
    if Bool_A and la < n_random:#-----------------------------------Good signals
        df_in.loc[df_in.index[la+n_reales]           ,'Failure Type'] = 'g'
        for k in harmonics:
            df_in.loc[df_in.index[la+n_reales],k]   = dice[k]
        la                                                  = la + 1

    if Bool_B and lb < n_random:#-----------------------Acceptable signals
        df_in.loc[df_in.index[lb+1*n_random+n_reales],'Failure Type'] = 'y'
        for k in harmonics:
            df_in.loc[df_in.index[lb+1*n_random+n_reales],k] = dice[k]
        lb                                                  = lb + 1

    if Bool_C and lc < n_random: #------------------Unacceptable signals
        df_in.loc[df_in.index[lc+2*n_random+n_reales],'Failure Type'] = 'r'
        for k in harmonics:
            df_in.loc[df_in.index[lc+2*n_random+n_reales],k] = dice[k]
        lc                                    = lc + 1 

    return df_in,la,lb,lc 
#------------------------------------------------------------------------------
def Synth_Loose_Bedplate(df_speed_in,df_SPEED_in):
    
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Loose_Bedplate(harm)
    
    columnas_out = ['Type','Failure Type','Kurtosis','Skewness','Wnl_o','Wnl','Entropy','Nnl']
    df_Values    = pd.DataFrame(index   = range( df_speed.shape[0] + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( df_speed.shape[0] + 3*n_random , np.size(columnas_out))) )
    
    for counter,k in enumerate (harm.index):
        signal                               = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']        = 'Real'
        df_Values.loc[counter,'Failure Type']    = harm.iloc[counter]['$Loose Bedplate Failure']
        df_Values.loc[counter,'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[counter,'Skewness']    = stats.skew(signal)
        df_Values.loc[counter,'Wnl']         = Wnl(signal)
        df_Values.loc[counter,'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[counter,'Entropy']     = Entropy(signal)
        #print(stats.entropy(signal,base = 2),Entropy(signal))
        df_Values.loc[counter,'Nnl']         = Nnl(signal)
            
    spectrum      = np.fft.fft(df_speed_in.iloc[0].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    harmonics     = ['1.0','2.0','3.0']
    columnas_rand = ['Failure Type'] + harmonics
                        #--Dataframe con valores de partida señal GOLDEN
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[0]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[0]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[0]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )
    df_randn_spcs.loc['mean','1.0'] = 4.8
    df_randn_spcs.loc['std','1.0']  = 1.2
    df_randn_spcs.loc['mean','2.0'] = 0.9
    df_randn_spcs.loc['std','2.0']  = 0.5
    df_randn_spcs.loc['mean','3.0'] = 0.9
    df_randn_spcs.loc['std','3.0']  = 0.5
    
                            #--DataFrame con todos los valores de amplitud aleatorios válidos
    df_random     = pd.DataFrame(index = range(3*n_random), 
                                 columns = columnas_rand, 
                                 data = np.zeros((3*n_random,np.size(columnas_rand))) )
#          Bedplate       1.0       2.0       3.0
#        0        g  1.997650  0.982614  0.340704
#        1        y  5.789201  1.149204  0.459223
#        2        r  5.584069  0.466865  1.256888
   
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    df_envelope                     = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    df_envelope.loc['0','1.0']      = 10
    df_envelope.loc['0','2.0']      = 1.2
    df_envelope.loc['0','3.0']      = 2.4
#                1.0  2.0  3.0
#            0  10.0  1.2  2.4    
    
    l1 = l2 = l3  = 0
    
    print(df_envelope)
    print(df_dado)
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            #print(df_dado.loc['0'][k],df_envelope.loc['0'][k])
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            A = 0   < df_dado.loc['0']['1.0'] < 2.0 
            B = 2.0 < df_dado.loc['0']['1.0'] < 5.0
            C = 5.0 < df_dado.loc['0']['1.0']
            D = (PK(df_dado.loc['0']['2.0']) and PK(df_dado.loc['0']['2.0'])) and df_dado.loc['0']['3.0'] > 1*df_dado.loc['0']['2.0']
            
            if A and l1 < n_random:#-----------------------------------Good signals
                df_random.loc[df_random.index[l1]           ,'Failure Type'] = 'g'
                for k in harmonics:
                    df_random.loc[df_random.index[l1],k]            = df_dado.loc['0'][k]
                l1                                                  = l1 + 1

            if (B ^ C) and l2 < n_random:#-----------------------Acceptable signals
                df_random.loc[df_random.index[l2+1*n_random],'Failure Type'] = 'y'
                for k in harmonics:
                    df_random.loc[df_random.index[l2+1*n_random],k] = df_dado.loc['0'][k]
                l2                                                  = l2 + 1

            if (C and D) and l3 < n_random: #------------------Unacceptable signals
                df_random.loc[df_random.index[l3+2*n_random],'Failure Type'] = 'r'
                for k in harmonics:
                    df_random.loc[df_random.index[l3+2*n_random],k] = df_dado.loc['0'][k]
                l3                                                  = l3 + 1

            if l1 == n_random and l2 == n_random and l3 == n_random:
                print(l1,l2,l3)
                break                
    #df_random = pd.read_pickle('dados.pkl')
    print(df_random)      
    for k in range(df_random.shape[0]): #----- IFFT de cada una de las señales sinteticas
        for harm_nb,hrm_name in enumerate(harmonics):
            inic = int(df_golden.loc['n_s',hrm_name])
            fin  = int(df_golden.loc['n_e',hrm_name])
#            print('power ',hrm_name ,df_golden.loc['RMS',hrm_name] )
#            print(inic,fin)
#            print(hrm_name,'---------',df_random.iloc[k][hrm_name])
#            print(df_golden.loc['RMS',hrm_name],inic,fin)
            #print('--------------',df_random.iloc[k][hrm_name],df_golden.loc['RMS',hrm_name] )
            fact                        = df_random.iloc[k][hrm_name]/df_golden.loc['RMS',hrm_name] 
            spec_rand[inic:fin]         = fact          * spectrum[inic:fin]
            spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * spectrum[l-fin+1:l-inic+1]
        
        #print(np.std(spec_rand))
        signal_math                       = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal         
        signal = np.real(signal_math[start:end])# + A_noise * np.random.randn(np.size(signal))
        signal = Rms(signal) #--ALBERTO, esta es tu señal ya normalizada!!!!!!!
        
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Type']        = 'Synth'
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Failure Type']    =  df_random.iloc[k]['Failure Type']
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Skewness']    = stats.skew(signal)
        #df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl_o']    = Wnl_o(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl']         = Wnl(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy']     = Entropy(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Nnl']         = Nnl(signal) 
    
    return df_Values


#==============================================================================
#==============================================================================
#------------------------------------------------------------------------------
def Synth_Loose_Bedplate_n(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Loose_Bedplate(harm)
    harmonics     = ['1.0','2.0','3.0']
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter]['$Loose Bedplate Failure']
        df_Values.loc[counter,'Kurtosis']     = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[counter,'Skewness']     = stats.skew(signal)
        df_Values.loc[counter,'Wnl']          = Wnl(signal)
        df_Values.loc[counter,'Entropy']      = Entropy(signal)
        df_Values.loc[counter,'Nnl']          = Nnl(signal)
            
    spectrum      = np.fft.fft(df_speed_in.iloc[0].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
                        #--Dataframe con valores de partida señal GOLDEN
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[0]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[0]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[0]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )
    df_randn_spcs.loc['mean','1.0'] = 4.8
    df_randn_spcs.loc['std','1.0']  = 1.2
    df_randn_spcs.loc['mean','2.0'] = 0.9
    df_randn_spcs.loc['std','2.0']  = 0.5
    df_randn_spcs.loc['mean','3.0'] = 0.9
    df_randn_spcs.loc['std','3.0']  = 0.5
       
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    df_envelope                     = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    df_envelope.loc['0','1.0']      = 10
    df_envelope.loc['0','2.0']      = 1.2
    df_envelope.loc['0','3.0']      = 2.4   
    
    l1 = l2 = l3  = 0
    
    print(df_envelope)
    print(df_dado)
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            A = 0   < df_dado.loc['0']['1.0'] < 2.0 
            B = 2.0 < df_dado.loc['0']['1.0'] < 5.0
            C = 5.0 < df_dado.loc['0']['1.0']
            D = (PK(df_dado.loc['0']['2.0']) and PK(df_dado.loc['0']['2.0'])) and df_dado.loc['0']['3.0'] > 1*df_dado.loc['0']['2.0']
#             decision_table(Bool_A,la,Bool_B,lb,Bool_C,lc,df_in,dice,counter,n_reales)
            df_Values,l1,l2,l3 = decision_table(A,l1,
                                                (B ^ C),l2,
                                                (C and D),l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)
            
#            if A and l1 < n_random:#-----------------------------------Good signals
#                df_Values.loc[df_Values.index[l1+n_reales]           ,'Failure Type'] = 'g'
#                for k in harmonics:
#                    df_Values.loc[df_Values.index[l1+n_reales],k]   = df_dado.loc['0'][k]
#                l1                                                  = l1 + 1
#
#            if (B ^ C) and l2 < n_random:#-----------------------Acceptable signals
#                df_Values.loc[df_Values.index[l2+1*n_random+n_reales],'Failure Type'] = 'y'
#                for k in harmonics:
#                    df_Values.loc[df_Values.index[l2+1*n_random+n_reales],k] = df_dado.loc['0'][k]
#                l2                                                  = l2 + 1
#
#            if (C and D) and l3 < n_random: #------------------Unacceptable signals
#                df_Values.loc[df_Values.index[l3+2*n_random+n_reales],'Failure Type'] = 'r'
#                for k in harmonics:
#                    df_Values.loc[df_Values.index[l3+2*n_random+n_reales],k] = df_dado.loc['0'][k]
#                l3                                                  = l3 + 1

            if l1 == n_random and l2 == n_random and l3 == n_random:
                print(l1,l2,l3)
                break                
      
    for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
        for harm_nb,hrm_name in enumerate(harmonics):
            inic                        = int(df_golden.loc['n_s',hrm_name])
            fin                         = int(df_golden.loc['n_e',hrm_name])
            fact                        = df_Values.iloc[k+n_reales][hrm_name]/df_golden.loc['RMS',hrm_name] 
            spec_rand[inic:fin]         = fact          * spectrum[inic:fin]
            spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * spectrum[l-fin+1:l-inic+1]
        
        
        signal_math                       = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal         
        signal = np.real(signal_math[start:end])
        signal = Rms(signal) #--ALBERTO, esta es tu señal ya normalizada!!!!!!!
        
        df_Values.loc[ df_Values.index[k + n_reales] , 'Type']        = 'Synth'
        df_Values.loc[ df_Values.index[k + n_reales] , 'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[ df_Values.index[k + n_reales] , 'Skewness']    = stats.skew(signal)
        df_Values.loc[ df_Values.index[k + n_reales] , 'Wnl']         = Wnl(signal)
        df_Values.loc[ df_Values.index[k + n_reales] , 'Entropy']     = Entropy(signal)
        df_Values.loc[ df_Values.index[k + n_reales] , 'Nnl']         = Nnl(signal) 
    
    return df_Values

#==============================================================================
#==============================================================================

def Synth_Surge_Effect(df_speed_in,df_SPEED_in):
    
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Surge_Effect(harm)
    
    columnas_out = ['Type','Failure Type','Kurtosis','Skewness','Wnl_o','Wnl','Entropy','Nnl']
    df_Values    = pd.DataFrame(index   = range( df_speed.shape[0] + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( df_speed.shape[0] + 3*n_random , np.size(columnas_out))) )
    
    for counter,k in enumerate (harm.index):
        signal                               = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']        = 'Real'
        df_Values.loc[counter,'Failure Type']    = harm.iloc[counter]['$Surge E. Failure']
        df_Values.loc[counter,'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[counter,'Skewness']    = stats.skew(signal)
        df_Values.loc[counter,'Wnl']         = Wnl(signal)
        df_Values.loc[counter,'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[counter,'Entropy']     = Entropy(signal)
        #print(stats.entropy(signal,base = 2),Entropy(signal))
        df_Values.loc[counter,'Nnl']         = Nnl(signal)
            
    spectrum      = np.fft.fft(df_speed_in.iloc[0].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    harmonics     = ['Surge E. 0.33x 0.5x','Surge E. 12/20k']
    columnas_rand = ['Surge E. Failure'] + harmonics
                        #--Dataframe con valores de partida señal GOLDEN
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[0]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[0]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[0]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )
    df_randn_spcs.loc['mean','Surge E. 0.33x 0.5x'] = 0.05
    df_randn_spcs.loc['std' ,'Surge E. 0.33x 0.5x'] = 0.1
    df_randn_spcs.loc['mean','Surge E. 12/20k']     = 0.05
    df_randn_spcs.loc['std' ,'Surge E. 12/20k']     = 0.1

    
                            #--DataFrame con todos los valores de amplitud aleatorios válidos
    df_random     = pd.DataFrame(index = range(3*n_random), 
                                 columns = columnas_rand, 
                                 data = np.zeros((3*n_random,np.size(columnas_rand))) )
#          Bedplate       1.0       2.0       3.0
#        0        g  1.997650  0.982614  0.340704
#        1        y  5.789201  1.149204  0.459223
#        2        r  5.584069  0.466865  1.256888
   
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    df_envelope                     = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    df_envelope.loc['0','Surge E. 0.33x 0.5x']      = 0.7
    df_envelope.loc['0','Surge E. 12/20k']          = 0.7
#                1.0  2.0  3.0
#            0  10.0  1.2  2.4    
    
    l1 = l2 = l3  = 0
    
    print(df_envelope)
    print(df_dado)
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            #print(df_dado.loc['0'][k],df_envelope.loc['0'][k])
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template,df_dado.loc['0']['Surge E. 0.33x 0.5x'],df_dado.loc['0']['Surge E. 12/20k'])
        if bool_template:#-----------------TEMPLATE
            A = PK(df_dado.loc['0']['Surge E. 0.33x 0.5x'])
            B = PK(df_dado.loc['0']['Surge E. 12/20k'])
            print(A,B,df_dado.loc['0']['Surge E. 0.33x 0.5x'],df_dado.loc['0']['Surge E. 12/20k'])
            print(l1,l2,l3)
            if not A and l1 < n_random:#-----------------------------------Good signals
                df_random.loc[df_random.index[l1]           ,'Failure Type'] = 'g'
                for k in harmonics:
                    df_random.loc[df_random.index[l1],k]            = df_dado.loc['0'][k]
                l1                                                  = l1 + 1

            if (A or B) and l2 < n_random:#-----------------------Acceptable signals
                df_random.loc[df_random.index[l2+1*n_random],'Failure Type'] = 'y'
                for k in harmonics:
                    df_random.loc[df_random.index[l2+1*n_random],k] = df_dado.loc['0'][k]
                l2                                                  = l2 + 1

            if (A and B) and l3 < n_random: #------------------Unacceptable signals
                df_random.loc[df_random.index[l3+2*n_random],'Failure Type'] = 'r'
                for k in harmonics:
                    df_random.loc[df_random.index[l3+2*n_random],k] = df_dado.loc['0'][k]
                l3                                                  = l3 + 1

            if l1 == n_random and l2 == n_random and l3 == n_random:
                print(l1,l2,l3)
                break                
    #df_random = pd.read_pickle('dados.pkl')
    print(df_random)      
    for k in range(df_random.shape[0]): #----- IFFT de cada una de las señales sinteticas
        for harm_nb,hrm_name in enumerate(harmonics):
            inic = int(df_golden.loc['n_s',hrm_name])
            fin  = int(df_golden.loc['n_e',hrm_name])
#            print('power ',hrm_name ,df_golden.loc['RMS',hrm_name] )
#            print(inic,fin)
#            print(hrm_name,'---------',df_random.iloc[k][hrm_name])
#            print(df_golden.loc['RMS',hrm_name],inic,fin)
            #print('--------------',df_random.iloc[k][hrm_name],df_golden.loc['RMS',hrm_name] )
            fact                        = df_random.iloc[k][hrm_name]/df_golden.loc['RMS',hrm_name] 
            spec_rand[inic:fin]         = fact          * spectrum[inic:fin]
            spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * spectrum[l-fin+1:l-inic+1]
        
        #print(np.std(spec_rand))
        signal_math                       = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal         
        signal = np.real(signal_math[start:end])# + A_noise * np.random.randn(np.size(signal))
        signal = Rms(signal) #--ALBERTO, esta es tu señal ya normalizada!!!!!!!
        
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Type']        = 'Synth'
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Failure Type']    =  df_random.iloc[k]['Failure Type']
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Skewness']    = stats.skew(signal)
        #df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl_o']    = Wnl_o(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl']         = Wnl(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy']     = Entropy(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Nnl']         = Nnl(signal) 
    
    return df_Values

#==============================================================================
#==============================================================================
def Synth_Severe_Misaligment(df_speed_in,df_SPEED_in):
    
    n_JSON_sel   = 0
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Severe_Misaligment(harm)
    
    columnas_out = ['Type','Failure Type','Kurtosis','Skewness','Wnl_o','Wnl','Entropy','Nnl']
    df_Values    = pd.DataFrame(index   = range( df_speed.shape[0] + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( df_speed.shape[0] + 3*n_random , np.size(columnas_out))) )
    
    for counter,k in enumerate (harm.index):
        signal                               = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']        = 'Real'
        df_Values.loc[counter,'Failure Type']    = harm.iloc[counter]['$Severe Mis. Failure']
        df_Values.loc[counter,'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[counter,'Skewness']    = stats.skew(signal)
        df_Values.loc[counter,'Wnl']         = Wnl(signal)
        df_Values.loc[counter,'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[counter,'Entropy']     = Entropy(signal)
        #print(stats.entropy(signal,base = 2),Entropy(signal))
        df_Values.loc[counter,'Nnl']         = Nnl(signal)
            
    spectrum      = np.fft.fft(df_speed_in.iloc[n_JSON_sel].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    harmonics     = ['1.0','2.0','3.0','4.0','5/2','7/2','9/2']
    columnas_rand = ['Failure Type'] + harmonics
                        #--Dataframe con valores de partida señal GOLDEN
    df_golden     = pd.DataFrame(index   = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data    = np.zeros([3,np.size(harmonics)]) )
    
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_JSON_sel]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_JSON_sel]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_JSON_sel]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )
    df_randn_spcs.loc['mean','1.0'] = 4.6
    df_randn_spcs.loc['std','1.0']  = 0.85
    df_randn_spcs.loc['mean','2.0'] = 0.6
    df_randn_spcs.loc['std','2.0']  = 0.2
    df_randn_spcs.loc['mean','3.0'] = 1
    df_randn_spcs.loc['std','3.0']  = 0.15
    df_randn_spcs.loc['mean','3.0'] = 0.5
    df_randn_spcs.loc['std','3.0']  = 0.01
    df_randn_spcs.loc['mean','4.0'] = 0.5
    df_randn_spcs.loc['std','4.0']  = 0.01
    
    df_randn_spcs.loc['mean','5/2'] = df_randn_spcs.loc['mean','7/2'] = df_randn_spcs.loc['mean','9/2'] = 0.08
    df_randn_spcs.loc['std','5/2']  = df_randn_spcs.loc['std','7/2']  = df_randn_spcs.loc['std','9/2']  = 0.15
                            #--DataFrame con todos los valores de amplitud aleatorios válidos
    df_random     = pd.DataFrame(index = range(3*n_random), 
                                 columns = columnas_rand, 
                                 data = np.zeros((3*n_random,np.size(columnas_rand))) )
#          Bedplate       1.0       2.0       3.0
#        0        g  1.997650  0.982614  0.340704
#        1        y  5.789201  1.149204  0.459223
#        2        r  5.584069  0.466865  1.256888
   
    df_dado                    = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    df_envelope                = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    df_envelope.loc['0','1.0'] = 10
    df_envelope.loc['0','2.0'] = 0.9
    df_envelope.loc['0','3.0'] = 1.4
    df_envelope.loc['0','4.0'] = 0.9
    df_envelope.loc['0','5/2'] = df_envelope.loc['0','5/2']      = df_envelope.loc['0','5/2']      = 0.2
   
    
    l1 = l2 = l3  = 0
    
    print(df_envelope)
    print(df_dado)

    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True 
        for k in harmonics: 
                       #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1))    
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]
            #print(bool_template) 
            #---------------------
        
        bool_template = True                           
        if bool_template:#-----------------TEMPLATE
            N_picos_A = Number_PEAKS(0.02*df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0'],df_dado.iloc[0]['4.0'], 
                                  harm.iloc[n_JSON_sel]['RMS 5.0'],harm.iloc[n_JSON_sel]['RMS 6.0'],
                                  harm.iloc[n_JSON_sel]['RMS 7.0'],harm.iloc[n_JSON_sel]['RMS 8.0'],
                                  harm.iloc[n_JSON_sel]['RMS 9.0'],harm.iloc[n_JSON_sel]['RMS 10.0'])
            
            A         = N_picos_A >= 3 and  PK(df_dado.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(0.02*df_dado.iloc[0]['1.0'],df_dado.iloc[0]['5/2'],df_dado.iloc[0]['7/2'],df_dado.iloc[0]['9/2'],
                                  harm.iloc[n_JSON_sel]['RMS 11/2'],harm.iloc[n_JSON_sel]['RMS 13/2'],
                                  harm.iloc[n_JSON_sel]['RMS 15/2'],harm.iloc[n_JSON_sel]['RMS 17/2'],
                                  harm.iloc[n_JSON_sel]['RMS 19/2'])
            
            B         = N_picos_B >= 3 and  PK(df_dado.iloc[0]['1.0'] )
            C         = df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['1.0']
            print(A,N_picos_A,B,N_picos_B,C)
            if not A and l1 < n_random:#-----------------------------------Good signals
                df_random.loc[df_random.index[l1]           ,'Failure Type'] = 'g'
                for k in harmonics:
                    df_random.loc[df_random.index[l1],k]            = df_dado.loc['0'][k]
                l1                                                  = l1 + 1

            if (A or B) and l2 < n_random:#-----------------------Acceptable signals
                df_random.loc[df_random.index[l2+1*n_random],'Failure Type'] = 'y'
                for k in harmonics:
                    df_random.loc[df_random.index[l2+1*n_random],k] = df_dado.loc['0'][k]
                l2                                                  = l2 + 1

            if (A and B and C) and l3 < n_random: #------------------Unacceptable signals
                df_random.loc[df_random.index[l3+2*n_random],'Failure Type'] = 'r'
                for k in harmonics:
                    df_random.loc[df_random.index[l3+2*n_random],k] = df_dado.loc['0'][k]
                l3                                                  = l3 + 1

            if l1 == n_random and l2 == n_random and l3 == n_random:
                print(l1,l2,l3)
                break
        print(l1,l2,l3)                
    #df_random = pd.read_pickle('dados.pkl')
          
    for k in range(df_random.shape[0]): #----- IFFT de cada una de las señales sinteticas
        for harm_nb,hrm_name in enumerate(harmonics):
            inic = int(df_golden.loc['n_s',hrm_name])
            fin  = int(df_golden.loc['n_e',hrm_name])
#            print('power ',hrm_name ,df_golden.loc['RMS',hrm_name] )
#            print(inic,fin)
#            print(hrm_name,'---------',df_random.iloc[k][hrm_name])
#            print(df_golden.loc['RMS',hrm_name],inic,fin)
            #print('--------------',df_random.iloc[k][hrm_name],df_golden.loc['RMS',hrm_name] )
            fact                        = df_random.iloc[k][hrm_name]/df_golden.loc['RMS',hrm_name] 
            spec_rand[inic:fin]         = fact          * spectrum[inic:fin]
            spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * spectrum[l-fin+1:l-inic+1]
        
        #print(np.std(spec_rand))
        signal_math                       = l*np.fft.ifft(spec_rand)
        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
            print('Cuidado señal no valida!!!!!!!!!!!!!!!')
            #----espectro de la señal sintetica => spec_rand
            #----señal sintetica en el tiempo   => signal         
        signal = np.real(signal_math[start:end])# + A_noise * np.random.randn(np.size(signal))
        signal = Rms(signal) #--ALBERTO, esta es tu señal ya normalizada!!!!!!!
        
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Type']        = 'Synth'
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Failure Type']    =  df_random.iloc[k]['Failure Type']
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Kurtosis']    = stats.kurtosis(np.abs(signal),fisher = False)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Skewness']    = stats.skew(signal)
        #df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl_o']    = Wnl_o(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Wnl']         = Wnl(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy']     = Entropy(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Entropy_rob'] = Entropy_rob(signal)
        df_Values.loc[ df_Values.index[k + df_speed.shape[0]] , 'Nnl']         = Nnl(signal) 
    
    return df_Values
#------------------------------------------------------------------------------
if __name__ == '__main__':
   
        
    pi       = np.pi
    E1       = 0.15
    
    fs       = 5120
    l        = 16384
    l_2      = np.int(l/2)
    t        = np.arange(l)/fs
    f        = np.arange(l)/(l-1)*fs
    A_noise  = 0*0.8
    n_random = 100 #---Numeroseñales sintéticas de cada tipo (Red, Green, Yellow)
    #start    = 0;       end   = l
    start    = 2600;    end   = 13500
    length   = end-start
    
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
        'Hour'         : '10' 
                }
            
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
  
    #df_Values1        = Synth_Severe_Misaligment(df_speed,df_SPEED)
    df_Values        = Synth_Loose_Bedplate_n(df_speed,df_SPEED)
    #df_Values        =       Synth_Surge_Effect(df_speed,df_SPEED)
    
    fig               = plt.figure(1)
    ax                = fig.add_subplot(111, projection='3d')    
    for k in range(df_Values.shape[0]):
        x = df_Values.iloc[k]['Kurtosis']
        y = df_Values.iloc[k]['Wnl']
        z = df_Values.iloc[k]['Entropy']
        
#        if (k-df_speed.shape[0])>0:
#            label = str(k-df_speed.shape[0])
#            ax.text   (x,y,z,label,fontsize=7)
        color = df_Values.iloc[k]['Failure Type']
        if df_Values.iloc[k]['Type'] == 'Real':
            ax.scatter(x,y,z, facecolors='gray',edgecolor = color,marker='o')
        else:
            ax.scatter(x,y,z,c = color)
    
    ax.set_xlabel('Kurtosis')
    ax.set_ylabel('Wnl')
    ax.set_zlabel('Entropy')
    plt.show()

#    df_Values2        = Synth_Loose_Bedplate(df_speed,df_SPEED)
#       
#    fig = plt.figure(1)
#    ax  = fig.add_subplot(111, projection='3d')    
#    for k in range(df_Values2.shape[0]):
#        x = df_Values2.iloc[k]['Kurtosis']
#        y = df_Values2.iloc[k]['Wnl']
#        z = df_Values2.iloc[k]['Entropy']
#        #label = str(format ( df_random.iloc[k]['1x Good'],'.01f'))+' '+str(format ( df_random.iloc[k]['2x Good'],'.01f'))+' '+str(format ( df_random.iloc[k]['3x Good'],'.01f'))
#        color = df_Values2.iloc[k]['Bedplate']
#        if df_Values2.iloc[k]['Type'] == 'Real':
#            ax.scatter(x,y,z, facecolors='gray',edgecolor = color,marker='o')
#        else:
#            ax.scatter(x,y,z,c = color)
# 
#    ax.set_xlabel('Kurtosis')
#    ax.set_ylabel('Wnl')
#    ax.set_zlabel('Entropy')
#    plt.show()
#    

