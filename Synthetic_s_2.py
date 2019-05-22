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
#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'
#--------------------------------------------------------------------------------
#def PK(a):
#    if a > E2:
#        out = True
#    else:
#        out = False
#    return out
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

def FP_Extraction(sig,df_in,kounter):
    df_in.loc[kounter,'Kurtosis']     = stats.kurtosis(np.abs(sig),fisher = False)
    df_in.loc[kounter,'Skewness']     = stats.skew(sig)
    df_in.loc[kounter,'Wnl']          = Wnl(sig)
    df_in.loc[kounter,'Entropy']      = Entropy(sig)
    df_in.loc[kounter,'Nnl']          = Nnl(sig)
    return df_in

def Plot_3D(df_in, eje_x, eje_y, eje_z,titulo):
    fig               = plt.figure()
    ax                = fig.add_subplot(111, projection='3d')    
    for k in range(df_in.shape[0]):
        x = df_in.iloc[k][eje_x]
        y = df_in.iloc[k][eje_y]
        z = df_in.iloc[k][eje_z]
        
#        if (k-df_speed.shape[0])>0:
#            label = str(k-df_speed.shape[0])
#            ax.text   (x,y,z,label,fontsize=7)
        color = df_in.iloc[k]['Failure Type']
        if df_in.iloc[k]['Type'] == 'Real':
            ax.scatter(x,y,z, facecolors='gray',edgecolor = color,marker='o')
        else:
            ax.scatter(x,y,z,c = color)
    ax.set_title (titulo)
    ax.set_xlabel(eje_x)
    ax.set_ylabel(eje_y)
    ax.set_zlabel(eje_z)
    plt.show()

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
    signal_math                = l*np.fft.ifft(spec_rand)

    if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
        print('Cuidado señal no valida!!!!!!!!!!!!!!!')
        #----espectro de la señal sintetica => spec_rand
        #----señal sintetica en el tiempo   => signal
    data_out = np.real(signal_math)
    return data_out





class FailureMode:
    def __init__(self,FailureName, 
                 df_TI_signal, df_SP_SIGNAL,n_golden,
                 Harmonics,rand_mean,rand_std,template_specs):
        
        self.FailureName         = FailureName
    
        self.df_TI_signal        = df_TI_signal             
        self.df_SP_SIGNAL        = df_SP_SIGNAL             
        self.n_golden            = n_golden
        self.Harmonics           = Harmonics
        self.rand_mean           = rand_mean
        self.rand_std            = rand_std
        self.template_specs      = template_specs
        
        self.df_FingerPrint      = 0
        self.spectrum            = 0
        self.spec_rand           = 0
        self.df_gold_OUT     = 0
        
    def __func__(self):
        
        df_RD_specs_OUT             = pd.DataFrame(index = ['lon','mean','std'],columns = self.Harmonics, 
                                        data = np.zeros((3,np.size(self.Harmonics))) )
        df_RD_specs_OUT.loc['mean'] = self.rand_mean
        df_RD_specs_OUT.loc['std']  = self.rand_std
        df_env_specs_OUT            = pd.DataFrame(index = ['0'], columns = self.Harmonics, 
                                                   data = np.zeros((1,np.size(self.Harmonics))) )
        df_env_specs_OUT.loc['0']   = self.template_specs
        self.df_FingerPrint         = df_Harmonics(self.df_SP_SIGNAL, fs,'blower')

        exec('self.df_FingerPrint = '+self.FailureName+'(self.df_FingerPrint)')
       
        n_real_OUT                    = self.df_SP_SIGNAL.shape[0]
        print('n_reales',n_real_OUT)
        columnas_OUT                 = ['Type','Failure Type']+self.Harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
        df_Values_OUT                = pd.DataFrame(index   = range( n_real_OUT + 3*n_random), #---output
                                                                    columns = columnas_OUT, 
                                                                    data    = np.zeros(( n_real_OUT + 3*n_random , np.size(columnas_OUT))) )
                    #--Dataframe con valores de partida señal GOLDEN
                         
        self.df_gold_OUT                 = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                                   columns = self.Harmonics,
                                                   data    = np.zeros([3,np.size(self.Harmonics)]) )
        for k in self.Harmonics:
            self.df_gold_OUT.loc['RMS',k] = self.df_FingerPrint.iloc[self.n_golden]['RMS '+ k]
            self.df_gold_OUT.loc['n_s',k] = self.df_FingerPrint.iloc[self.n_golden]['n_s '+ k]
            self.df_gold_OUT.loc['n_e',k] = self.df_FingerPrint.iloc[self.n_golden]['n_e '+ k]
            
        df_dice_OUT                  = pd.DataFrame(index = ['0'], 
                                                   columns = self.Harmonics, 
                                                   data = np.zeros((1,np.size(self.Harmonics))) )
        
        for counter,k in enumerate (self.df_FingerPrint.index):
            signal                                    = Rms(df_speed.iloc[counter].values[start:end])
            df_Values_OUT.loc[counter,'Type']         = 'Real'
            df_Values_OUT.loc[counter,'Failure Type'] = self.df_FingerPrint.iloc[counter]['$'+self.FailureName+'_Failure']
            df_Values_OUT                             = FP_Extraction(signal,df_Values_OUT,counter )       
            
            self.spectrum      = np.fft.fft(self.df_TI_signal.iloc[self.n_golden].values)/l #-----me quedo con la primera 
            self.spec_rand     = np.copy(self.spectrum) 
        
#        print(df_Values_OUT)    
#        print(df_gold_OUT)
#        print(df_RD_specs_OUT)
#        print(df_env_specs_OUT)
#        print(df_dice_OUT)
        print('----terminado-----')
        return df_Values_OUT,self.df_gold_OUT,df_RD_specs_OUT,df_env_specs_OUT,df_dice_OUT,n_real_OUT
        
    def __func1__(self,df_Values1,n_reales):
        a = 10

        
        for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
            for harm_nb,hrm_name in enumerate(self.Harmonics):
                inic                        = int(self.df_gold_OUT.loc['n_s',hrm_name])
                fin                         = int(self.df_gold_OUT.loc['n_e',hrm_name])
                fact                        = df_Values1.iloc[k+n_reales][hrm_name]/self.df_gold_OUT.loc['RMS',hrm_name] 
                self.spec_rand[inic:fin]         = fact          * self.spectrum[inic:fin]
                self.spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * self.spectrum[l-fin+1:l-inic+1]  #----espectro de la señal sintetica => spec_rand   
            signal_math                                            = l*np.fft.ifft(self.spec_rand)
            if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
                print('Cuidado señal no valida!!!!!!!!!!!!!!!')               
                
            signal                                                 = np.real(signal_math[start:end])
            signal                                                 = Rms(signal)          #--signal => señal sintetica en tiempo   ESTA ALBERTO!!!
            df_Values1.loc[ df_Values1.index[k + n_reales] , 'Type'] = 'Synth'
            df_Values1                                              = FP_Extraction(signal,df_Values1,k + n_reales)

        
        return df_Values1
    
    def __func_2__(self):
        b= 1
        return b
#==============================================================================
#                           NOT WORKING                                                  1
#==============================================================================
def Synth_Severe_Misaligment(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Severe Mis. Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Severe_Misaligment(harm)
    harmonics    = ['1.0','2.0','3.0','4.0','5/2','7/2','9/2']
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )

                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','1.0'] = 4.6
    df_randn_spcs.loc['std','1.0']  = 0.85 
    
    df_randn_spcs.loc['mean','2.0'] = 0.6
    df_randn_spcs.loc['std','2.0']  = 0.2 
    
    df_randn_spcs.loc['mean','3.0'] = 1
    df_randn_spcs.loc['std','3.0']  = 0.15 
    
    df_randn_spcs.loc['mean','4.0'] = 0.5   
    df_randn_spcs.loc['std','4.0']  = 0.01
    
    df_randn_spcs.loc['mean','5/2'] = df_randn_spcs.loc['mean','7/2'] = df_randn_spcs.loc['mean','9/2'] = 0.08
    df_randn_spcs.loc['std','5/2']  = df_randn_spcs.loc['std','7/2']  = df_randn_spcs.loc['std','9/2']  = 0.15
       
    df_envelope.loc['0','1.0'] = 10
    df_envelope.loc['0','2.0'] = 0.9
    df_envelope.loc['0','3.0'] = 1.4
    df_envelope.loc['0','4.0'] = 0.9
    df_envelope.loc['0','5/2'] = df_envelope.loc['0','7/2']      = df_envelope.loc['0','9/2']      = 0.2
    
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values = FP_Extraction(signal,df_Values,counter)

    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k] 
            print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            N_picos_A = Number_PEAKS(0.02*df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0'],df_dado.iloc[0]['4.0'], 
                                  harm.iloc[n_golden]['RMS 5.0'],harm.iloc[n_golden]['RMS 6.0'],
                                  harm.iloc[n_golden]['RMS 7.0'],harm.iloc[n_golden]['RMS 8.0'],
                                  harm.iloc[n_golden]['RMS 9.0'],harm.iloc[n_golden]['RMS 10.0'])
            
            A         = N_picos_A >= 3 and  PK(df_dado.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(0.02*df_dado.iloc[0]['1.0'],df_dado.iloc[0]['5/2'],df_dado.iloc[0]['7/2'],df_dado.iloc[0]['9/2'],
                                  harm.iloc[n_golden]['RMS 11/2'],harm.iloc[n_golden]['RMS 13/2'],
                                  harm.iloc[n_golden]['RMS 15/2'],harm.iloc[n_golden]['RMS 17/2'],
                                  harm.iloc[n_golden]['RMS 19/2'])
            
            B         = N_picos_B >= 3 and  PK(df_dado.iloc[0]['1.0'] )
            C         = df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['1.0']
            print(l1,l2,l3,A,B,C)
            df_Values,l1,l2,l3 = decision_table( not A,l1,
                                                A or B,l2,
                                                A and B and C,l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

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
        df_Values = FP_Extraction(signal,df_Values,k + n_reales)

    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)

    return df_Values
#==============================================================================
#                                                                             2
#==============================================================================
def Synth_Loose_Bedplate(Process_var):
    df_Values,df_golden,df_randn_spcs,df_envelope,df_dado,n_reales = Process_var.__func__()    
    
    l1 = l2 = l3 = 0
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válido-
        bool_template = True
        for k in Process_var.Harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            A = 0   < df_dado.loc['0']['1.0'] < 2.0 
            B = 2.0 < df_dado.loc['0']['1.0'] < 5.0
            C = 5.0 < df_dado.loc['0']['1.0']
            D = (PK(E2,df_dado.loc['0']['2.0']) and PK(E2,df_dado.loc['0']['2.0'])) and df_dado.loc['0']['3.0'] > 1*df_dado.loc['0']['2.0']
            df_Values,l1,l2,l3 = decision_table(A,l1,
                                                (B ^ C),l2,
                                                (C and D),l3,
                                                df_Values,df_dado.loc['0'],Process_var.Harmonics,n_reales)                   
#    for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
#        for harm_nb,hrm_name in enumerate(Process_var.Harmonics):
#            inic                        = int(df_golden.loc['n_s',hrm_name])
#            fin                         = int(df_golden.loc['n_e',hrm_name])
#            fact                        = df_Values.iloc[k+n_reales][hrm_name]/df_golden.loc['RMS',hrm_name] 
#            Process_var.spec_rand[inic:fin]         = fact          * Process_var.spectrum[inic:fin]
#            Process_var.spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * Process_var.spectrum[l-fin+1:l-inic+1]  #----espectro de la señal sintetica => spec_rand   
#        signal_math                                            = l*np.fft.ifft(Process_var.spec_rand)
#        if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
#            print('Cuidado señal no valida!!!!!!!!!!!!!!!')               
#            
#        signal                                                 = np.real(signal_math[start:end])
#        signal                                                 = Rms(signal)          #--signal => señal sintetica en tiempo   ESTA ALBERTO!!!
#        df_Values.loc[ df_Values.index[k + n_reales] , 'Type'] = 'Synth'
#        df_Values                                              = FP_Extraction(signal,df_Values,k + n_reales)
        
    df_Values= Process_var.__func1__(df_Values,n_reales)     
    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',Process_var.FailureName)
    return df_Values
#==============================================================================
#                                                                             3    
#==============================================================================
def Synth_Surge_Effect(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Surge E. Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Surge_Effect(harm)
    harmonics    = ['Surge E. 0.33x 0.5x','Surge E. 12/20k']
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )
                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','Surge E. 0.33x 0.5x'] = 0.05
    df_randn_spcs.loc['std' ,'Surge E. 0.33x 0.5x'] = 0.1
    df_randn_spcs.loc['mean','Surge E. 12/20k']     = 0.05
    df_randn_spcs.loc['std' ,'Surge E. 12/20k']     = 0.1
       
    df_envelope.loc['0','Surge E. 0.33x 0.5x']      = 0.7
    df_envelope.loc['0','Surge E. 12/20k']          = 0.7
    
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values = FP_Extraction(signal,df_Values,counter)
    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            A = PK(df_dado.loc['0']['Surge E. 0.33x 0.5x'])
            B = PK(df_dado.loc['0']['Surge E. 12/20k'])
            df_Values,l1,l2,l3 = decision_table(not A,l1,
                                                (A or B),l2,
                                                (A and B),l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

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
        df_Values = FP_Extraction(signal,df_Values,k + n_reales)

    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)

    return df_Values

#==============================================================================
#                                                                             4    
#==============================================================================
def Synth_Oil_Whip(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Oil Whip Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Oil_Whip(harm)
    harmonics    = ['1/2','5/2']
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )
                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','1/2'] = 0.05
    df_randn_spcs.loc['std' ,'1/2'] = 0.5
    df_randn_spcs.loc['mean','5/2'] = 0.1
    df_randn_spcs.loc['std' ,'5/2'] = 0.5
       
    df_envelope.loc['0','1/2']      = 0.7
    df_envelope.loc['0','5/2']      = 0.7
    
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    time_sig      = df_speed_in.iloc[n_golden].values #-----me quedo con la primera 
    t_sig_rnd     = np.copy(time_sig)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values                             = FP_Extraction(signal,df_Values,counter)
    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            #print(k)
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            #print(df_dado)
            A =    PK(E2,df_dado.loc['0']['1/2']) and PK(E2,df_dado.loc['0']['5/2'])
            B = PEAKS(E2,df_dado.loc['0']['1/2'],harm.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['1/2'] > 0.02 * harm.iloc[n_golden]['RMS 1.0']
            C = PEAKS(E2,df_dado.loc['0']['5/2'],harm.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['5/2'] > 0.02 * harm.iloc[n_golden]['RMS 1.0']
#            print(l1,l2,l3,A,B,C)
            df_Values,l1,l2,l3 = decision_table(not A and ( not(B and C)),l1,
                                                A ^ (B ^ C),l2,
                                                A and B and C,l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

            if l1 == n_random and l2 == n_random and l3 == n_random:
                print(l1,l2,l3)
                break                
    freqs = [0.5*harm.iloc[n_golden]['f 1.0'], 0.5*harm.iloc[n_golden]['f 1.0'] ]  
    for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
        for harm_nb,hrm_name in enumerate(harmonics):
            t_sig_rnd = Synth_feature(freqs[np.mod(harm_nb,2)],4.3,df_Values.iloc[k+n_reales][hrm_name],'triangle',t_sig_rnd)

        signal                                                 = t_sig_rnd[start:end]
        signal                                                 = Rms(signal) #--ALBERTO, esta es tu señal ya normalizada!!!!!!!        
        df_Values.loc[ df_Values.index[k + n_reales] , 'Type'] = 'Synth'
        df_Values                                              = FP_Extraction(signal,df_Values,k + n_reales)
    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)
    return df_Values
#==============================================================================
#                                                                             5
#==============================================================================
def Synth_Plain_Bearing_Clearance(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Plain Bearing Clearance Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Plain_Bearing_Clearance(harm)
    harmonics    = ['1.0','2.0','3.0','1/2','3/2','5/2']
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )

                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','1.0'] = 4.6
    df_randn_spcs.loc['std','1.0']  = 1    #______
    
    df_randn_spcs.loc['mean','2.0'] = 1
    df_randn_spcs.loc['std','2.0']  = 0.5  #______
    
    df_randn_spcs.loc['mean','3.0'] = 0.9
    df_randn_spcs.loc['std','3.0']  = 0.5  #______
    
    df_randn_spcs.loc['mean','1/2'] = 0.9
    df_randn_spcs.loc['std','1/2']  = 0.5 #______
    
    df_randn_spcs.loc['mean','3/2'] = 0.5
    df_randn_spcs.loc['std','3/2']  = 0.01 #______
    
    df_randn_spcs.loc['mean','5/2'] = 0.3
    df_randn_spcs.loc['std','5/2']  = 0.5
       
    df_envelope.loc['0','1.0'] = 10
    df_envelope.loc['0','2.0'] = 3
    df_envelope.loc['0','3.0'] = 2
    df_envelope.loc['0','1/2'] = 2
    df_envelope.loc['0','3/2'] = 1.5
    df_envelope.loc['0','5/2'] = 1
    
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values = FP_Extraction(signal,df_Values,counter)

    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k] 
            #print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            a1 = PEAKS(df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['1.0'] > df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['3.0'])
            a2 = PEAKS(df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['2.0'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3.0'] > 0.02 * df_dado.iloc[0]['1.0'])
            A         = a1 and a2

            b1 = PEAKS(df_dado.iloc[0]['1/2'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > df_dado.iloc[0]['3/2'] > df_dado.iloc[0]['5/2'])
            b2 = PEAKS(df_dado.iloc[0]['1/2'],df_dado.iloc[0]['1.0'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['5/2'] > 0.02 * df_dado.iloc[0]['1.0']) 
            B         = b1 and b2
            
            print(l1,l2,l3,A,B)
            df_Values,l1,l2,l3 = decision_table( (not A) and (not B),l1,
                                                A or B,l2,
                                                A and B,l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

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
        df_Values = FP_Extraction(signal,df_Values,k + n_reales)

    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)

    return df_Values

#==============================================================================
#                                                                             6
#==============================================================================
    
def Synth_Centrifugal_Fan_unbalance(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Blower Wheel Unbalance Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Blower_Wheel_Unbalance(harm)
    harmonics    = ['1.0','2th Max Value.'] 
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )

                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','1.0'] = 4.8
    df_randn_spcs.loc['std','1.0']  = 1.2    #______
    
    df_randn_spcs.loc['mean','2th Max Value.'] = 0.8
    df_randn_spcs.loc['std','2th Max Value.']  = 0.5  #______
    

    df_envelope.loc['0','1.0'] = 10
    df_envelope.loc['0','2th Max Value.'] = 3

    
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values = FP_Extraction(signal,df_Values,counter)

    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k] 
            #print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            
            A         = df_dado.iloc[0]['1.0'] * 0.15 < df_dado.iloc[0]['2th Max Value.']
            B         = df_dado.iloc[0]['1.0'] < 4
            
            print(l1,l2,l3,A,B)
            df_Values,l1,l2,l3 = decision_table( (A and B),l1,
                                                A or B,l2,
                                                (not A) and (not B),l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

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
        df_Values = FP_Extraction(signal,df_Values,k + n_reales)

    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)

    return df_Values
#==============================================================================
#                                                                             7
#==============================================================================
    
def Synth_Presure_Pulsations(df_speed_in,df_SPEED_in):
    
    n_reales     = df_speed.shape[0]
    label        = '$Pressure P. Failure'
    harm         = df_Harmonics(df_SPEED_in, fs,'blower')
    harm         = Pressure_Pulsations(harm)
    harmonics    = ['1/3','2/3','4/3','5/3','8/3','4.0'] 
    columnas_out = ['Type','Failure Type']+harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
    df_Values    = pd.DataFrame(index   = range( n_reales + 3*n_random), #---output
                                columns = columnas_out, 
                                data    = np.zeros(( n_reales + 3*n_random , np.size(columnas_out))) )

                        #--Dataframe con valores de partida señal GOLDEN
    n_golden      = 0                    
    df_golden     = pd.DataFrame(index = ['RMS','n_s','n_e'],
                                 columns = harmonics,
                                 data = np.zeros([3,np.size(harmonics)]) )
    for k in harmonics:
        df_golden.loc['RMS',k] = harm.iloc[n_golden]['RMS '+ k]
        df_golden.loc['n_s',k] = harm.iloc[n_golden]['n_s '+ k]
        df_golden.loc['n_e',k] = harm.iloc[n_golden]['n_e '+ k]
    
    df_randn_spcs = pd.DataFrame(index = ['lon','mean','std'],
                                 columns = harmonics, 
                                 data = np.zeros((3,np.size(harmonics))) )    
    
    df_envelope   = pd.DataFrame(index = ['0'], columns = harmonics, 
                                 data = np.zeros((1,np.size(harmonics))) )
    
    df_randn_spcs.loc['mean','1/3'] = 0.1
    df_randn_spcs.loc['std' ,'1/3'] = 0.5    #______
    
    df_randn_spcs.loc['mean','2/3'] = 0.3
    df_randn_spcs.loc['std' ,'3/3'] = 0.5    #______
    
    df_randn_spcs.loc['mean','4/3'] = 0.2
    df_randn_spcs.loc['std', '4/3'] = 0.5    #______
    
    df_randn_spcs.loc['mean','5/3'] = 0.2
    df_randn_spcs.loc['std', '5/3'] = 0.5    #______
    
    df_randn_spcs.loc['mean','8/3'] = 0.15
    df_randn_spcs.loc['std', '8/3'] = 0.5    #______
    
    df_randn_spcs.loc['mean','4.0'] = 0.1
    df_randn_spcs.loc['std', '4.0'] = 0.5    #______
    
#    df_envelope.loc['0','1/3'] = 10
#    df_envelope.loc['0','2/3'] = 10
#    df_envelope.loc['0','4/3'] = 10
#    df_envelope.loc['0','5/3'] = 10
#    df_envelope.loc['0','8/3'] = 10
#    df_envelope.loc['0','4.0'] = 10

        
    spectrum      = np.fft.fft(df_speed_in.iloc[n_golden].values)/l #-----me quedo con la primera 
    spec_rand     = np.copy(spectrum)
    
    for counter,k in enumerate (harm.index):
        signal                                = Rms(df_speed.iloc[counter].values[start:end])
        df_Values.loc[counter,'Type']         = 'Real'
        df_Values.loc[counter,'Failure Type'] = harm.iloc[counter][label]
        df_Values = FP_Extraction(signal,df_Values,counter)

    
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = harmonics, data = np.zeros((1,np.size(harmonics))) )
    
    while True: #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in harmonics:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_randn_spcs.loc['mean'][k] + df_randn_spcs.loc['std'][k] * np.random.randn(1)) 
            #bool_template      = bool_template and df_dado.loc['0'][k] < df_envelope.loc['0'][k] 
            #print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            A  = PEAKS(0.10,df_dado.iloc[0]['1/3'],df_dado.iloc[0]['2/3'],df_dado.iloc[0]['4/3'],df_dado.iloc[0]['5/3'])
            b1 = PEAKS(0.10,df_dado.iloc[0]['1/3'],df_dado.iloc[0]['4/3'],df_dado.iloc[0]['8/3'],df_dado.iloc[0]['4.0'])
            B = b1 and (df_dado.iloc[0]['4/3'] > df_dado.iloc[0]['1/3']) and (df_dado.iloc[0]['8/3'] > df_dado.iloc[0]['1/3']) and (df_dado.iloc[0]['4.0'] > df_dado.iloc[0]['1/3'])
            print(l1,l2,l3,A,B)
            df_Values,l1,l2,l3 = decision_table( not A,l1,
                                                A ,l2,
                                                A and B,l3,
                                                df_Values,df_dado.loc['0'],harmonics,n_reales)

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
        df_Values = FP_Extraction(signal,df_Values,k + n_reales)

    Plot_3D(df_Values, 'Kurtosis', 'Wnl', 'Entropy',label)

    return df_Values
#------------------------------------------------------------------------------
if __name__ == '__main__':
   
        
    pi       = np.pi
    E1       = 0.15
    E2       = 0.10
    fs       = 5120
    l        = 16384
    l_2      = np.int(l/2)
    t        = np.arange(l)/fs
    f        = np.arange(l)/(l-1)*fs
    A_noise  = 0*0.8
    
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
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-20T00:20:00.9988564Z',
        'FechaInicio'  : '2019-02-12T00:52:46.9988564Z',
        'NumeroTramas' : '10',
        'Parametros'   : 'waveform' ,
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '12',#'12'
        'Hour'         : ''
        }
    
    n_random = 10 #---Numeroseñales sintéticas de cada tipo (Red, Green, Yellow)
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
  
    #df_Values   = Synth_Severe_Misaligment(df_speed,df_SPEED)                  # 1. NOT WORKING
    #df_Values   = Synth_Loose_Bedplate(df_speed,df_SPEED)                      # 2. ok
    #df_Values   = Synth_Surge_Effect(df_speed,df_SPEED)                        # 3.CRAPPY PLOT
#    df_Values   =Synth_Oil_Whip(df_speed,df_SPEED)                              # 4    
    #df_Values   = Synth_Plain_Bearing_Clearance(df_speed,df_SPEED)             # 5. 
#    df_Values   = Synth_Centrifugal_Fan_unbalance(df_speed,df_SPEED)           # 6
#    df_Values   = Synth_Presure_Pulsations (df_speed,df_SPEED)                 # 7
    

    
    
    #harm         = df_Harmonics(df_SPEED, fs,'blower')
    Process_variable2 = FailureMode('Loose_Bedplate',df_speed,df_SPEED,0,
                                    ['1.0','2.0','3.0'],[4.8,0.9,0.9],[1.2,0.5,0.5],[10,1.2,2.4]) 
    df_Values   = Synth_Loose_Bedplate(Process_variable2)   
   

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
