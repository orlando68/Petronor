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
import time

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
#------------------------------------------------------------------------------
@jit 
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
@jit 
def FP_Extraction(sig,df_in,kounter):
    df_in.loc[kounter,'Kurtosis']     = stats.kurtosis(np.abs(sig),fisher = False)
    df_in.loc[kounter,'Skewness']     = stats.skew(sig)
    df_in.loc[kounter,'Wnl']          = Wnl(sig)
    df_in.loc[kounter,'Entropy']      = Entropy(sig)
    df_in.loc[kounter,'Nnl']          = Nnl(sig)
    return df_in
#------------------------------------------------------------------------------
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
#------------------------------------------------------------------------------
@jit 
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

#------------------------------------------------------------------------------

class FailureMode:
    def __init__(self,FailureName,df_TI_signal, df_SP_SIGNAL):

        self.FailureName         = FailureName
    
        self.df_TI_signal        = df_TI_signal             
        self.df_SP_SIGNAL        = df_SP_SIGNAL             
        
        self.spectrum            = 0
        self.spec_rand           = 0
        self.df_Values_OUT       = 0
        
    def __func__(self,n_golden,Harmonics,rand_mean,rand_std,template_specs):
        
        self.spectrum                = np.fft.fft(self.df_TI_signal.iloc[n_golden].values)/l #-----me quedo con la primera 
        self.spec_rand               = np.copy(self.spectrum) 
        n_real_OUT                   = self.df_SP_SIGNAL.shape[0]
        columnas_OUT                 = ['Type','Failure Type']+Harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
        print('n_reales',n_real_OUT)
        
        df_RD_specs_OUT              = pd.DataFrame(index = ['lon','mean','std'],columns = Harmonics, data = np.zeros((3,np.size(Harmonics))) )
        df_RD_specs_OUT.loc['mean']  = rand_mean
        df_RD_specs_OUT.loc['std']   = rand_std
        
        df_env_specs_OUT             = pd.DataFrame(index = ['0'], columns = Harmonics, data = np.zeros((1,np.size(Harmonics))) )
        df_env_specs_OUT.loc['0']    = template_specs
        
        self.df_Values_OUT           = pd.DataFrame(index   = range( n_real_OUT + 3*n_random),columns = columnas_OUT, 
                                                                    data    = np.zeros(( n_real_OUT + 3*n_random , np.size(columnas_OUT))) )
        
        df_FingerPrint               = df_Harmonics(self.df_SP_SIGNAL, fs,'blower')
        exec('self.df_FingerPrint = '+self.FailureName+'(df_FingerPrint)')
        start_time = time.time() 
        for counter,k in enumerate (df_FingerPrint.index):
            signal                                         = Rms(df_speed.iloc[counter].values[start:end])
            self.df_Values_OUT.loc[counter,'Type']         = 'Real'
            self.df_Values_OUT.loc[counter,'Failure Type'] = df_FingerPrint.iloc[counter]['$'+self.FailureName+'_Failure']
            self.df_Values_OUT                             = FP_Extraction(signal,self.df_Values_OUT,counter )       
        stop_time = time.time()
        print('Wasted time to calculate time FP values from real signals:',stop_time-start_time)
        
        df_gold_OUT                  = pd.DataFrame(index = ['RMS','n_s','n_e'], columns = Harmonics, data = np.zeros([3,np.size(Harmonics)]) )
        for k in Harmonics:
            df_gold_OUT.loc['RMS',k] = df_FingerPrint.iloc[n_golden]['RMS '+ k]
            df_gold_OUT.loc['n_s',k] = df_FingerPrint.iloc[n_golden]['n_s '+ k]
            df_gold_OUT.loc['n_e',k] = df_FingerPrint.iloc[n_golden]['n_e '+ k]
            
#        df_dice_OUT                  = pd.DataFrame(index = ['0'], columns = Harmonics, data = np.zeros((1,np.size(Harmonics))) )
    
#        print(self.df_Values_OUT)    
#        print(df_gold_OUT)
#        print(df_RD_specs_OUT)
#        print(df_env_specs_OUT)
#        print(df_dice_OUT)

        start_time = time.time()       
        exec('self.df_Values_OUT = DecissionTable_'+self.FailureName+'(df_FingerPrint,Harmonics,df_RD_specs_OUT,df_env_specs_OUT,self.df_Values_OUT,n_real_OUT,n_golden)')        
        stop_time = time.time()
        print('Wasted time to generate valid synthetic signals Spetral Values:',stop_time-start_time)

        start_time = time.time() 
        for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
            for harm_nb,hrm_name in enumerate(Harmonics):
                inic                             = int(df_gold_OUT.loc['n_s',hrm_name])
                fin                              = int(df_gold_OUT.loc['n_e',hrm_name])
                fact                             = self.df_Values_OUT.iloc[k+n_real_OUT][hrm_name]/df_gold_OUT.loc['RMS',hrm_name] 
                self.spec_rand[inic:fin]         = fact          * self.spectrum[inic:fin]
                self.spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * self.spectrum[l-fin+1:l-inic+1]  #----espectro de la señal sintetica => spec_rand   
            signal_math                                            = l*np.fft.ifft(self.spec_rand)
            if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
                print('Cuidado señal no valida!!!!!!!!!!!!!!!')               
                
            signal                                                                    = np.real(signal_math[start:end])
            signal                                                                    = Rms(signal) #--signal => señal sintetica en tiempo   ESTA ALBERTO!!!
            self.df_Values_OUT.loc[self.df_Values_OUT.index[k + n_real_OUT] , 'Type'] = 'Synth'
            self.df_Values_OUT                                                        = FP_Extraction(signal,self.df_Values_OUT,k + n_real_OUT)
        stop_time = time.time()
        print('Wasted time to calculate time FP values from synthetic signals:',stop_time-start_time)    
        
        Plot_3D(self.df_Values_OUT, 'Kurtosis', 'Wnl', 'Entropy',self.FailureName)
        return self.df_Values_OUT
    
    def __func_2__(self):
        b= 1
        return b
#------------------------------------------------------------------------------

#def DecissionTable_Loose_Bedplate(Harmonics_IN,df_Values_IN,df_dice_IN,n_reales_IN,l1_IN,l2_IN,l3_IN):
#    #print('====================ENTRO AQUI')
#    
#    A = 0   < df_dice_IN.loc['0']['1.0'] < 2.0 
#    B = 2.0 < df_dice_IN.loc['0']['1.0'] < 5.0
#    C = 5.0 < df_dice_IN.loc['0']['1.0']
#    D = (PK(E2,df_dice_IN.loc['0']['2.0']) and PK(E2,df_dice_IN.loc['0']['2.0'])) and df_dice_IN.loc['0']['3.0'] > 1*df_dice_IN.loc['0']['2.0']
#    df_Values_IN,l1_IN,l2_IN,l3_IN = decision_table(A,l1_IN,
#                                        (B ^ C),l2_IN,
#                                        (C and D),l3_IN,
#                                        df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)    
#    return df_Values_IN,l1_IN,l2_IN,l3_IN
#

#------------------------------------------------------------------------------
@jit 
def DecissionTable_Loose_Bedplate(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    l1_IN = l2_IN = l3_IN = 0
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    while not(l1_IN == n_random and l2_IN == n_random and l3_IN == n_random): #---------rellenamos df_random con "sucesos" aleatorios válido-
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template          = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            #self.df_Values_OUT,l1,l2,l3 = DecissionTable_Loose_Bedplate(Harmonics,self.df_Values_OUT,df_dice_OUT,n_real_OUT,l1,l2,l3)


            A = 0   < df_dice_IN.loc['0']['1.0'] < 2.0 
            B = 2.0 < df_dice_IN.loc['0']['1.0'] < 5.0
            C = 5.0 < df_dice_IN.loc['0']['1.0']
            D = (PK(E2,df_dice_IN.loc['0']['2.0']) and PK(E2,df_dice_IN.loc['0']['2.0'])) and df_dice_IN.loc['0']['3.0'] > 1*df_dice_IN.loc['0']['2.0']
            df_Values_IN,l1_IN,l2_IN,l3_IN = decision_table(A,l1_IN,
                                                (B ^ C),l2_IN,
                                                (C and D),l3_IN,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)    
    return df_Values_IN
#------------------------------------------------------------------------------    
def DecissionTable_Severe_Misaligment(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

    l1 = l2 = l3  = 0
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
            #print(       df_dice_IN.loc['0'][k] , df_env_specs_IN.loc['0'][k],'==>',df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            N_picos_A = Number_PEAKS(0.02*df_dice_IN.iloc[0]['1.0'],df_dice_IN.iloc[0]['2.0'],df_dice_IN.iloc[0]['3.0'],df_dice_IN.iloc[0]['4.0'], 
                                  SP_FingerPrint.iloc[n_golden]['RMS 5.0'],SP_FingerPrint.iloc[n_golden]['RMS 6.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 7.0'],SP_FingerPrint.iloc[n_golden]['RMS 8.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 9.0'],SP_FingerPrint.iloc[n_golden]['RMS 10.0'])
            
            A         = N_picos_A >= 3 and  PK(E2,df_dice_IN.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(0.02*df_dice_IN.iloc[0]['1.0'],df_dice_IN.iloc[0]['5/2'],df_dice_IN.iloc[0]['7/2'],df_dice_IN.iloc[0]['9/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 11/2'],SP_FingerPrint.iloc[n_golden]['RMS 13/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 15/2'],SP_FingerPrint.iloc[n_golden]['RMS 17/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 19/2'])
            
            B         = N_picos_B >= 3 and  PK(E2,df_dice_IN.iloc[0]['1.0'] )
            C         = df_dice_IN.iloc[0]['2.0'] > df_dice_IN.iloc[0]['1.0']
            print(l1,l2,l3,A,B,C)
            df_Values_IN,l1,l2,l3 = decision_table( not A,l1,
                                                A or B,l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#------------------------------------------------------------------------------
def DecissionTable_Surge_Effect(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    
    l1 = l2 = l3  = 0
    df_dado  = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_env_specs_IN.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            A = PK(E2,df_dado.loc['0']['Surge E. 0.33x 0.5x'])
            B = PK(E2,df_dado.loc['0']['Surge E. 12/20k'])
            df_Values_IN,l1,l2,l3 = decision_table(not A,l1,
                                                (A or B),l2,
                                                (A and B),l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#------------------------------------------------------------------------------    
def DecissionTable_Plain_Bearing_Lubrication_Whip(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            #print(k)
            df_dado.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_env_specs_IN.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            #print(df_dado)
            A =    PK(E2,df_dado.loc['0']['1/2']) and PK(E2,df_dado.loc['0']['5/2'])
            B = PEAKS(E2,df_dado.loc['0']['1/2'],SP_FingerPrint.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['1/2'] > 0.02 * SP_FingerPrint.iloc[n_golden]['RMS 1.0']
            C = PEAKS(E2,df_dado.loc['0']['5/2'],SP_FingerPrint.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['5/2'] > 0.02 * SP_FingerPrint.iloc[n_golden]['RMS 1.0']
#            print(l1,l2,l3,A,B,C)
            df_Values,l1,l2,l3 = decision_table(not A and ( not(B and C)),l1,
                                                A ^ (B ^ C),l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN) 
    return df_Values_IN
#------------------------------------------------------------------------------    

def DecissionTable_Plain_Bearing_Clearance(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
            #print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            a1 = PEAKS(E2,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['1.0'] > df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['3.0'])
            a2 = PEAKS(E2,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['2.0'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3.0'] > 0.02 * df_dado.iloc[0]['1.0'])
            A         = a1 and a2

            b1 = PEAKS(E2,df_dado.iloc[0]['1/2'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > df_dado.iloc[0]['3/2'] > df_dado.iloc[0]['5/2'])
            b2 = PEAKS(E2,df_dado.iloc[0]['1/2'],df_dado.iloc[0]['1.0'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['5/2'] > 0.02 * df_dado.iloc[0]['1.0']) 
            B         = b1 and b2
            
            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table( (not A) and (not B),l1,
                                                A or B,l2,
                                                A and B,l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------    

def DecissionTable_Centrifugal_Fan_Unbalance(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

    l1 = l2 = l3  = 0
    df_dado                         = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dado.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dado.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
            #print(       df_dado.loc['0'][k] , df_envelope.loc['0'][k],'==>',df_dado.loc['0'][k] < df_envelope.loc['0'][k])                    #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            A         = df_dado.iloc[0]['1.0'] * 0.15 < df_dado.iloc[0]['2th Max Value.']
            B         = df_dado.iloc[0]['1.0'] < 4
            
            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table( (A and B),l1,
                                                A or B,l2,
                                                (not A) and (not B),l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#------------------------------------------------------------------------------                 
      


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
    
    n_random = 3 #---Numeroseñales sintéticas de cada tipo (Red, Green, Yellow)
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
    
    
#    Process_variable1 = FailureMode('Severe_Misaligment',df_speed,df_SPEED) 
#    Process_variable1.__func__(0,['1.0','2.0','3.0','4.0','5/2','7/2','9/2'],
#                               [4.6  , 0.6 ,1.0  ,0.5  ,0.08 ,0.08 , 0.08],
#                               [0.85 , 0.2 ,0.15 ,0.01 ,0.15 ,0.15 , 0.15],
#                               [10   , 0.9 ,1.4  ,0.9  ,0.2  ,0.2  , 0.2])
    
    Process_variable2 = FailureMode('Loose_Bedplate',df_speed,df_SPEED) 
    Process_variable2.__func__(0,['1.0','2.0','3.0'],[4.8,0.9,0.9],[1.2,0.5,0.5],[10,1.2,2.4])

    Process_variable3 = FailureMode('Surge_Effect',df_speed,df_SPEED) 
    Process_variable3.__func__(0,['Surge E. 0.33x 0.5x','Surge E. 12/20k'],[0.05,0.05],[0.1,0.1],[0.7,0.7])

    Process_variable4 = FailureMode('Plain_Bearing_Lubrication_Whip',df_speed,df_SPEED) 
    Process_variable4.__func__(0,['1/2','5/2'],[0.05,0.1],[0.5,0.5],[0.7,0.7])

    Process_variable5 = FailureMode('Plain_Bearing_Clearance',df_speed,df_SPEED) 
    Process_variable5.__func__(0,['1.0','2.0','3.0','1/2','3/2','5/2'],[4.6,1,0.9,0.9,0.5,0.3],[1,0.5,0.5,0.5,0.01,0.5],[10,3,2,2,1.5,1])
    
    Process_variable6 = FailureMode('Centrifugal_Fan_Unbalance',df_speed,df_SPEED) 
    Process_variable6.__func__(0,['1.0','2th Max Value.'],[4.8,0.8],[1.2,0.5],[10,3])

