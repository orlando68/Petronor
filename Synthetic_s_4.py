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


freq_names = ['1/2'                 ,'5/2'            ,'7/2'       ,'9/2'     ,
              '1/3'                 ,'2/3'            ,'4/3'       ,'5/3'     ,'8/3'         ,
              '1/4'                 ,
              '5.0'                 ,'6.0'            ,'7.0'       ,'8.0'     ,'9.0'         ,'10.0'        ,
              '11/2'                ,'13/2'           ,'15/2'      ,'17/2'    ,'19/2'                       ,
              '12.0'                ,'24.0'           ,'11.0'      ,'13.0'    ,'23.0'        ,'25.0'        ,
              'BPFO1'               ,'2*BPFO1'        ,'3*BPFO1'   ,'4*BPFO1' ,
              'BPFO2'               ,'2*BPFO2'        ,'3*BPFO2'   ,'4*BPFO2' ,
              'BPFI1'               ,'BPFI1+f'        ,'BPFI1-f'   ,'2*BPFI1' ,'2*BPFI1+f'   ,'2*BPFI1-f'   ,
              'BPFI2'               ,'BPFI2+f'        ,'BPFI2-f'   ,'2*BPFI2' ,'2*BPFI2+f'   ,'2*BPFI2-f'   ,
              'BSF1'                ,'BSF1-FTF1'      ,'BSF1+FTF1' ,'2*BSF1'  ,'2*BSF1-FTF1' ,'2*BSF1+FTF1' ,
              'BSF2'                ,'BSF2-FTF2'      ,'BSF2+FTF2' ,'2*BSF2'  ,'2*BSF2-FTF2' ,'2*BSF2+FTF2' ,
              'FTF1'                ,'2*FTF1'         ,'3*FTF1'    ,'4*FTF1'  ,
              'FTF2'                ,'2*FTF2'         ,'3*FTF2'    ,'4*FTF2'  ,
              'Surge E. 0.33x 0.5x' ,'Surge E. 12/20k',
              'Oil Whirl',
              'Flow T.']
f = 24.7
freq_values = [f/2                  , f*5/2           , f*7/2      , f*9/2    ,
               f*1/3                , f*2/3           , f*4/3      , f*5/3    , f*8/3,
               f/4                  ,
               5*f                  , 6*f             , 7*f        , 8*f      , 9*f          , 10*f         ,
               11/2*f               , 13/2*f          , 15/2*f     , 17/2*f   , 19/2*f                      ,
               12*f                 , 24*f            , 11*f       , 13*f     , 23*f         , 25*f         ,
               BPFO1                , 2*BPFO1         , 3*BPFO1    , 4*BPFO1,
               BPFO2                , 2*BPFO2         , 3*BPFO2    , 4*BPFO2,
               BPFI1                , BPFI1+f         , BPFI1-f    , 2*BPFI1  , 2*BPFI1+f    , 2*BPFI1-f    ,
               BPFI2                , BPFI2+f         , BPFI2-f    , 2*BPFI2  , 2*BPFI2+f    , 2*BPFI2-f    ,
               BSF1                 , BSF1-FTF1       , BSF1+FTF1  , 2*BSF1   , 2*BSF1-FTF1  , 2*BSF1+FTF1  ,
               BSF2                 , BSF2-FTF2       , BSF2+FTF2  , 2*BSF2   , 2*BSF2-FTF2  , 2*BSF2+FTF2  ,
               FTF1                 , 2*FTF1          , 3*FTF1     , 4*FTF1   ,
               FTF2                 , 2*FTF2          , 3*FTF2     , 4*FTF2   ,
               0.47*f               ,14700,
               0.41*f               ,
               14]

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
def AddRMS(input_list):
    out_list =[]
    for counter,item in enumerate(input_list):
        elemento = 'RMS '+item
        out_list.append(elemento)
    return out_list

@jit
#------------------------------------------------------------------------------
def df_FFT(df_in):
    #---- devuleve el espectro con ventana de hanning y corregida en potencia
    #---- es decir multiplicado por el factor 1.63    
    l_row     = df_in.shape[1]
    hann      = np.hanning(l_row) #
    df_cmplx  = pd.DataFrame(np.zeros([np.size(df_in.index), l_row], dtype=complex),
                             index = df_in.index,
                             columns = df_in.columns.values)

    for counter,indice in enumerate(df_in.index):

        trace                   = 1.63 * np.fft.fft(df_in.iloc[counter].values* hann/l_row)
        df_cmplx.iloc[counter]  =  trace  
    return df_cmplx
#------------------------------------------------------------------------------
@jit 
def  decision_table(Bool_A,la,Bool_B,lb,Bool_C,lc,df_in,dice,harmonics,n_reales):  
    
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
@jit 
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
            
            ax.scatter(x,y,z, facecolors='w',edgecolor = color[0],marker='o')
            #ax.scatter(x,y,z,edgecolor = color,marker='o')
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
        self.df_Values_OUT       = 0
        
    def __func__(self,n_golden,Harmonics,rand_mean,rand_std,template_specs):
        
        df_special_freqs           = pd.DataFrame(np.nan,index = ['Hz','n_sample'],columns = freq_names)
        df_special_freqs.loc['Hz'] = freq_values
        for counter,k in enumerate(freq_values):
            df_special_freqs.iloc[1,counter] = (16384-1)* df_special_freqs.iloc[0][counter]/5120
        
        self.df_FingerPrint_Real     = df_Harmonics(self.df_SP_SIGNAL, fs,'blower')
        exec('self.df_FingerPrint_Real = '+self.FailureName+'(self.df_FingerPrint_Real)')
        self.spectrum                = np.fft.fft(self.df_TI_signal.iloc[n_golden].values)/l #-----me quedo con la primera 
        spec_rand                    = np.copy(self.spectrum) 
        n_real_OUT                   = self.df_SP_SIGNAL.shape[0]
        columnas_OUT                 = ['Type','Failure Type']+Harmonics+['Kurtosis','Skewness','Wnl','Entropy','Nnl']  
                                    #------------------------------------------
        df_RD_specs_OUT              = pd.DataFrame(index = ['lon','mean','std'],columns = Harmonics, data = np.zeros((3,np.size(Harmonics))) )
        df_RD_specs_OUT.loc['mean']  = rand_mean
        df_RD_specs_OUT.loc['std']   = rand_std
                                    #------------------------------------------        
        df_env_specs_OUT             = pd.DataFrame(index = ['0'], columns = Harmonics, data = np.zeros((1,np.size(Harmonics))) )
        df_env_specs_OUT.loc['0']    = template_specs
                                    #-------Creo y preparo --self.df_Values_OUT--------        
        self.df_Values_OUT           = pd.DataFrame(index   = range( n_real_OUT + 3*n_random),columns = columnas_OUT, 
                                                                    data    = np.zeros(( n_real_OUT + 3*n_random , np.size(columnas_OUT))) )
        for counter,k in enumerate (self.df_FingerPrint_Real.index):
            signal                                         = Rms(df_speed.iloc[counter].values[start:end])
            self.df_Values_OUT.loc[counter,'Type']         = 'Real'
            self.df_Values_OUT.loc[counter,'Failure Type'] = self.df_FingerPrint_Real.iloc[counter]['$'+self.FailureName+'_Failure']
            self.df_Values_OUT                             = FP_Extraction(signal,self.df_Values_OUT,counter )       
        
        #--------------------TOMO LOS VALORES DE PARTIDA DE LA SEÑAL GOLDEN-------------
        df_gold_OUT                  = pd.DataFrame(index = ['RMS','n_s','n_e'], columns = Harmonics, data = np.zeros([3,np.size(Harmonics)]) )
        for k in Harmonics:
            df_gold_OUT.loc['RMS',k] = self.df_FingerPrint_Real.iloc[n_golden]['RMS '+ k]
            df_gold_OUT.loc['n_s',k] = self.df_FingerPrint_Real.iloc[n_golden]['n_s '+ k]
            df_gold_OUT.loc['n_e',k] = self.df_FingerPrint_Real.iloc[n_golden]['n_e '+ k]
            
        #--------------------GENERO LOS VALORES ALEATORIOS DE LAS S. SYNTH-------------
        start_time = time.time()  
        exec('self.df_Values_OUT = DecissionTable_'+self.FailureName+'(self.df_FingerPrint_Real,Harmonics,df_RD_specs_OUT,df_env_specs_OUT,self.df_Values_OUT,n_real_OUT,n_golden)')        
        stop_time = time.time()
        print('Wasted time to generate valid synthetic signals Spetral Values (DECISSION TABLE):',stop_time-start_time)

                    #-----------------------la velocidad en tiempo sintetica CON recorte
        self.df_speed_g  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(end-start))
        self.df_speed_y  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(end-start))
        self.df_speed_r  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(end-start))
                    #-----------------------la velocidad en freqcuencia sintetica SIN recorte       
        self.df_CMPLX_G  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(l))
        self.df_CMPLX_Y  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(l))
        self.df_CMPLX_R  = pd.DataFrame(np.nan ,index = range(n_random),columns = range(l))
        
        
        #--------------CONSTRUCCION DE SEÑALES SYNTHETIC--(IFFT)----------------
        start_time = time.time()
        #---------NECESARIAMENTE  HAY DOS TIPOS
        #        |_____Todas los modos de fallo salvo Oil Whip
        #        |_____Oil Whip (fabricar un pico CON MODULACION)
        if self.FailureName != 'Plain_Bearing_Lubrication_Whip': #picos sin MODULACION
            for k in range(3*n_random): 
                spec_rand                    = np.copy(self.spectrum)
                for harm_nb,hrm_name in enumerate(Harmonics):
                    inic                        = int(df_gold_OUT.loc['n_s',hrm_name])
                    fin                         = int(df_gold_OUT.loc['n_e',hrm_name])
                    RMS_value                   = df_gold_OUT.loc['RMS',hrm_name] 
                    if RMS_value > 0:  #---------------------------------------se modifica un piko
                        fact                        = self.df_Values_OUT.iloc[k+n_real_OUT][hrm_name] / RMS_value
                        spec_rand[inic:fin]         = fact          * self.spectrum[inic:fin]
                        spec_rand[l-fin+1:l-inic+1] = np.conj(fact) * self.spectrum[l-fin+1:l-inic+1]   
                    else:              #---------------------------------------se crea un piko nuevo
                        f_point = int(df_special_freqs.loc['n_sample'][hrm_name])
                        #print(harm_nb,hrm_name,f_point)
                        spec_rand[f_point]         = self.df_Values_OUT.iloc[k+n_real_OUT][hrm_name]
                        spec_rand[l-f_point]       = self.df_Values_OUT.iloc[k+n_real_OUT][hrm_name]
                signal_math                  = l*np.fft.ifft(spec_rand)
                if np.max( np.abs( np.imag(signal_math) )  ) > 1e-10:
                    print('Cuidado señal no valida!!!!!!!!!!!!!!!')               
                signal                                                                    = np.real(signal_math[start:end])
                signal                                                                    = Rms(signal)
                self.df_Values_OUT.loc[self.df_Values_OUT.index[k + n_real_OUT] , 'Type'] = 'Synth'
                self.df_Values_OUT                                                        = FP_Extraction(signal,self.df_Values_OUT,k + n_real_OUT)
                
                if k < n_random:
                    self.df_speed_g.iloc[k]            = signal
                    self.df_CMPLX_G.iloc[k]            = spec_rand
                if n_random <= k < 2*n_random:
                    self.df_speed_y.iloc[k-n_random]   = signal
                if 2*n_random <= k < 3*n_random:
                    self.df_speed_r.iloc[k-2*n_random] = signal
        
        if self.FailureName == 'Plain_Bearing_Lubrication_Whip': #pico con MODULACION
            freqs         = [0.5*self.df_FingerPrint_Real.iloc[n_golden]['f 1.0'], 2.5*self.df_FingerPrint_Real.iloc[n_golden]['f 1.0'] ] 
            time_sig      = self.df_TI_signal.iloc[n_golden].values 
            t_sig_rnd     = np.copy(time_sig)
            print('-------------------no hago nada-----------------------')
            print(self.FailureName)
            for k in range(3*n_random): #----- IFFT de cada una de las señales sinteticas
                for harm_nb,hrm_name in enumerate(Harmonics):
                    t_sig_rnd = Synth_feature(freqs[np.mod(harm_nb,2)],4.3,self.df_Values_OUT.iloc[k+n_real_OUT][hrm_name],'triangle',t_sig_rnd)
        
                signal                                                 = t_sig_rnd[start:end]
                signal                                                 = Rms(signal)         
                self.df_Values_OUT.loc[ self.df_Values_OUT.index[k + n_real_OUT] , 'Type'] = 'Synth'
                self.df_Values_OUT                                              = FP_Extraction(signal,self.df_Values_OUT,k + n_real_OUT)
            
                if k < n_random:
                    self.df_speed_g.iloc[k]            = signal
                if n_random <= k < 2*n_random:
                    self.df_speed_y.iloc[k-n_random]   = signal
                if 2*n_random <= k < 3*n_random:
                    self.df_speed_r.iloc[k-2*n_random] = signal   
        #-----------------ls velocidad en freq CON recorte---------------------    
        self.df_cmplx_g = df_FFT(self.df_speed_g)
        self.df_cmplx_y = df_FFT(self.df_speed_y)
        self.df_cmplx_r = df_FFT(self.df_speed_r)
        
        self.df_SP_FingerPrint_g = df_Harmonics(self.df_cmplx_g, fs,'blower')
        self.df_SP_FingerPrint_y = df_Harmonics(self.df_cmplx_y, fs,'blower')
        self.df_SP_FingerPrint_r = df_Harmonics(self.df_cmplx_r, fs,'blower') 
        
        
        stop_time = time.time()
        print('Wasted time to synthesize synthetic signals:',stop_time-start_time) 
        Plot_3D(self.df_Values_OUT, 'Kurtosis', 'Wnl', 'Entropy',self.FailureName)
        return self.df_Values_OUT
    
    def __func_2__(self):
        b= 1
                
        plot_waterfall_lines('espectro de la señal normalizada en el tiempo',self.df_cmplx_g,self.df_SP_FingerPrint_g,fs,0,400)
        plot_waterfall_lines('espectro modificado',self.df_CMPLX_G,self.df_SP_FingerPrint_g,fs,0,400)
        plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',self.df_SP_SIGNAL,self.df_FingerPrint_Real,fs,0,400)
        return b

#-----------------------------------------------------------------------------1   
@jit
def DecissionTable_Severe_Misaligment(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    
    l1 = l2 = l3  = 0
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
#            print(       df_dice_IN.loc['0'][k] , df_env_specs_IN.loc['0'][k],'==>',df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k])                    #---------------------
#        print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            N_picos_A = Number_PEAKS(E1,df_dice_IN.iloc[0]['2.0'],df_dice_IN.iloc[0]['3.0'],df_dice_IN.iloc[0]['4.0'])
            
            A         = N_picos_A >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(E1,df_dice_IN.iloc[0]['5/2'],df_dice_IN.iloc[0]['7/2'],df_dice_IN.iloc[0]['9/2'])
            
            B         = N_picos_B >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )
            C         = df_dice_IN.iloc[0]['2.0'] > df_dice_IN.iloc[0]['1.0']
            print(l1,l2,l3,A,'(',N_picos_A,')',B,'(',N_picos_B,')',C)
            df_Values_IN,l1,l2,l3 = decision_table( not A,l1,
                                                A or B,l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
"""
#-----------------------------------------------------------------------------1   
@jit
def DecissionTable_Severe_Misaligment(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    
    l1 = l2 = l3  = 0
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
#            print(       df_dice_IN.loc['0'][k] , df_env_specs_IN.loc['0'][k],'==>',df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k])                    #---------------------
#        print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            N_picos_A = Number_PEAKS(E1,df_dice_IN.iloc[0]['2.0'],df_dice_IN.iloc[0]['3.0'],df_dice_IN.iloc[0]['4.0'], 
                                  SP_FingerPrint.iloc[n_golden]['RMS 5.0'],SP_FingerPrint.iloc[n_golden]['RMS 6.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 7.0'],SP_FingerPrint.iloc[n_golden]['RMS 8.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 9.0'],SP_FingerPrint.iloc[n_golden]['RMS 10.0'])
            
            A         = N_picos_A >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(E1,df_dice_IN.iloc[0]['5/2'],df_dice_IN.iloc[0]['7/2'],df_dice_IN.iloc[0]['9/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 11/2'],SP_FingerPrint.iloc[n_golden]['RMS 13/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 15/2'],SP_FingerPrint.iloc[n_golden]['RMS 17/2'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 19/2'])
            
            B         = N_picos_B >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )
            C         = df_dice_IN.iloc[0]['2.0'] > df_dice_IN.iloc[0]['1.0']
            print(SP_FingerPrint.iloc[n_golden]['RMS 5.0'],SP_FingerPrint.iloc[n_golden]['RMS 6.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 7.0'],SP_FingerPrint.iloc[n_golden]['RMS 8.0'],
                                  SP_FingerPrint.iloc[n_golden]['RMS 9.0'],SP_FingerPrint.iloc[n_golden]['RMS 10.0'])
            print(l1,l2,l3,A,'(',N_picos_A,')',B,'(',N_picos_B,')',C)
            df_Values_IN,l1,l2,l3 = decision_table( not A,l1,
                                                A or B,l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
"""
#-----------------------------------------------------------------------------1   
"""
@jit
def DecissionTable_Severe_Misaligment(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    print('entradoen tabla-----------------------')
    l1 = l2 = l3  = 0
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    
    while not(l1 == n_random and l2 == n_random and l3 == n_random): #---------rellenamos df_random con "sucesos" aleatorios válidos
                #----------------lanzamos el dado
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template      = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k] 
#            print(       df_dice_IN.loc['0'][k] , df_env_specs_IN.loc['0'][k],'==>',df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k])                    #---------------------
#        print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            N_picos_A = Number_PEAKS(E1,df_dice_IN.iloc[0]['2.0'],df_dice_IN.iloc[0]['3.0'],df_dice_IN.iloc[0]['4.0'], 
                                  df_dice_IN.iloc[0]['5.0'],df_dice_IN.iloc[0]['6.0'],
                                  df_dice_IN.iloc[0]['7.0'],df_dice_IN.iloc[0]['8.0'],
                                  df_dice_IN.iloc[0]['9.0'],df_dice_IN.iloc[0]['10.0'])
            
            A         = N_picos_A >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )

            N_picos_B = Number_PEAKS(E1,df_dice_IN.iloc[0]['5/2'],df_dice_IN.iloc[0]['7/2'],df_dice_IN.iloc[0]['9/2'],
                                  df_dice_IN.iloc[0]['11/2'],df_dice_IN.iloc[0]['13/2'],
                                  df_dice_IN.iloc[0]['15/2'],df_dice_IN.iloc[0]['17/2'],
                                  df_dice_IN.iloc[0]['19/2'])
            
            B         = N_picos_B >= 3 and  PK(E1,df_dice_IN.iloc[0]['1.0'] )
            C         = df_dice_IN.iloc[0]['2.0'] > df_dice_IN.iloc[0]['1.0']
            
            print(l1,l2,l3,A,'(',N_picos_A,')',B,'(',N_picos_B,')',C)
            df_Values_IN,l1,l2,l3 = decision_table( not A,l1,
                                                A or B,l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
"""

#-----------------------------------------------------------------------------2
@jit 
def DecissionTable_Loose_Bedplate(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):
    l1_IN = l2_IN = l3_IN = 0
#    Harmonics_INb = AddRMS(Harmonics_IN)
#    Harmonics_INb = (Harmonics_IN)
    df_dice_IN = pd.DataFrame(index = ['0'], columns = Harmonics_IN, data = np.zeros((1,np.size(Harmonics_IN))) )
    #print(df_dice_IN)
    while not(l1_IN == n_random and l2_IN == n_random and l3_IN == n_random): #---------rellenamos df_random con "sucesos" aleatorios válido-
        bool_template = True
        for k in Harmonics_IN:             #------lanzo el dado--
            df_dice_IN.loc['0',k] = np.abs(df_RD_specs_IN.loc['mean'][k] + df_RD_specs_IN.loc['std'][k] * np.random.randn(1)) 
            bool_template          = bool_template and df_dice_IN.loc['0'][k] < df_env_specs_IN.loc['0'][k]                            #---------------------
        #print(bool_template)
        if bool_template:#-----------------TEMPLATE
            
            A = 0   < df_dice_IN.loc['0']['1.0'] < 2.0 
            B = 2.0 < df_dice_IN.loc['0']['1.0'] < 5.0
            C = 5.0 < df_dice_IN.loc['0']['1.0']
            D = (PK(E1,df_dice_IN.loc['0']['2.0']) and PK(E1,df_dice_IN.loc['0']['2.0'])) and df_dice_IN.loc['0']['3.0'] > 1*df_dice_IN.loc['0']['2.0']
            #print(l1_IN,l2_IN,l3_IN,A,B,C)
            df_Values_IN,l1_IN,l2_IN,l3_IN = decision_table(A,l1_IN,
                                                (B ^ C),l2_IN,
                                                (C and D),l3_IN,
                                                df_Values_IN,df_dice_IN.loc['0'],Harmonics_IN,n_reales_IN)    
    return df_Values_IN
#-----------------------------------------------------------------------------3
@jit 
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
            A = PK(E1,df_dado.loc['0']['Surge E. 0.33x 0.5x'])
            B = PK(E1,df_dado.loc['0']['Surge E. 12/20k'])
            df_Values_IN,l1,l2,l3 = decision_table(not A,l1,
                                                (A or B),l2,
                                                (A and B),l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------4   
@jit 
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
            A =    PK(E1,df_dado.loc['0']['1/2']) and PK(E1,df_dado.loc['0']['5/2'])
            B = PEAKS(E1,df_dado.loc['0']['1/2'],SP_FingerPrint.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['1/2'] > 0.02 * SP_FingerPrint.iloc[n_golden]['RMS 1.0']
            C = PEAKS(E1,df_dado.loc['0']['5/2'],SP_FingerPrint.iloc[n_golden]['RMS 1.0']) and df_dado.loc['0']['5/2'] > 0.02 * SP_FingerPrint.iloc[n_golden]['RMS 1.0']
#            print(l1,l2,l3,A,B,C)
            df_Values,l1,l2,l3 = decision_table(not A and ( not(B and C)),l1,
                                                A ^ (B ^ C),l2,
                                                A and B and C,l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN) 
    return df_Values_IN
#-----------------------------------------------------------------------------5   
@jit 
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
            
            a1 = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['1.0'] > df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['3.0'])
            a2 = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0']) and (df_dado.iloc[0]['2.0'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3.0'] > 0.02 * df_dado.iloc[0]['1.0'])
            A         = a1 and a2

            b1 = PEAKS(E1,df_dado.iloc[0]['1/2'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > df_dado.iloc[0]['3/2'] > df_dado.iloc[0]['5/2'])
            b2 = PEAKS(E1,df_dado.iloc[0]['1/2'],df_dado.iloc[0]['1.0'],df_dado.iloc[0]['3/2'],df_dado.iloc[0]['5/2']) and (df_dado.iloc[0]['1/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['3/2'] > 0.02 * df_dado.iloc[0]['1.0']) and (df_dado.iloc[0]['5/2'] > 0.02 * df_dado.iloc[0]['1.0']) 
            B         = b1 and b2
            
#            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table( (not A) and (not B),l1,
                                                A or B,l2,
                                                A and B,l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------6    
@jit 
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
#            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table( (A and B),l1,
                                                A or B,l2,
                                                (not A) and (not B),l3,
                                                df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------7  
@jit 
def DecissionTable_Pressure_Pulsations(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            A       = PEAKS(E1,df_dado.iloc[0]['1/3'],df_dado.iloc[0]['2/3'],df_dado.iloc[0]['4/3'],df_dado.iloc[0]['5/3'])
            B_peaks = PEAKS(E1,df_dado.iloc[0]['1/3'],df_dado.iloc[0]['4/3'],df_dado.iloc[0]['8/3'],df_dado.iloc[0]['4.0']) 
            B       = B_peaks and (df_dado.iloc[0]['4/3']> df_dado.iloc[0]['1/3']) and (df_dado.iloc[0]['8/3'] > df_dado.iloc[0]['1/3']) and (df_dado.iloc[0]['4.0'] > df_dado.iloc[0]['1/3'])
#            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table(not A,l1,
                                                   A,l2,
                                                   A and B,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------8                
@jit 
def DecissionTable_Shaft_Misaligments(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            
            A_peaks = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'])
            A       = A_peaks and df_dado.iloc[0]['2.0'] < 0.5 *  df_dado.iloc[0]['1.0']
            B_peaks = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'])
            B       = B_peaks and 1.5 *df_dado.iloc[0]['1.0'] >        df_dado.iloc[0]['2.0'] > 0.5 *df_dado.iloc[0]['1.0']
            C_peaks = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'])
            C       = C_peaks and 1.5 *df_dado.iloc[0]['1.0'] <        df_dado.iloc[0]['2.0']
            D       = PEAKS(E1,df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0'],df_dado.iloc[0]['4.0'],df_dado.iloc[0]['5.0'])
                        
#            print(l1,l2,l3,A,B)
            df_Values_IN,l1,l2,l3 = decision_table(A or not D,l1,
                                                   B and D,l2,
                                                   C and D,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------9                
@jit 
def  DecissionTable_Plain_Bearing_Block_Looseness(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            
            A_peaks = PEAKS(E1,df_dado.iloc[0]['1.0'],df_dado.iloc[0]['2.0'],df_dado.iloc[0]['3.0'])
            A       = A_peaks and df_dado.iloc[0]['1.0'] <df_dado.iloc[0]['2.0'] > df_dado.iloc[0]['3.0']
            B       = PEAKS(E1,df_dado.iloc[0]['1/2'],df_dado.iloc[0]['1/3'],df_dado.iloc[0]['1/4'])
     
#            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(not A and not B,l1,
                                                   A ^ B ,l2,
                                                   A and B ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#----------------------------------------------------------------------------10a              
@jit 
def  DecissionTable_Blade_Faults(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            
            A = PK(E1,df_dado.loc['0']['12.0'])
            B = PEAKS(E1,df_dado.loc['0']['12.0'],df_dado.loc['0']['24.0']) 
            C = PK(E1,df_dado.loc['0']['12.0'])       and ( PK(E1,df_dado.loc['0']['11.0']) or PK(E1,df_dado.loc['0']['13.0']) )
            D = C and PK(E1,df_dado.loc['0']['24.0']) 
            E = C and PK(E1,df_dado.loc['0']['24.0']) and ( PK(E1,df_dado.loc['0']['23.0']) or PK(E1,df_dado.loc['0']['25.0']) )
            F = df_dado.loc['0']['12.0'] < E2 and df_dado.loc['0']['24.0'] < E2
     
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A or B or F,l1,
                                                   C or D     ,l2,
                                                   E          ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#-----------------------------------------------------------------------------10b               
@jit 
def  DecissionTable_Flow_Turbulence(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            
            A       = df_dado.iloc[0]['Flow T.'] <= 0.2
            B_peaks = PK(E1,df_dado.iloc[0]['1.0'])
            B       = B_peaks and (0.2 <= df_dado.iloc[0]['Flow T.'] <= df_dado.iloc[0]['1.0'])
            C_peaks = PK(E1,df_dado.iloc[0]['1.0'])
            C       = C_peaks and (df_dado.iloc[0]['Flow T.'] >  df_dado.iloc[0]['1.0'])
     
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A,l1,
                                                   B ,l2,
                                                   C ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN
#----------------------------------------------------------------------------11
@jit 
def  DecissionTable_Plain_Bearing_Lubrication_Whirl(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
                                                #-----------green-----------------
                                                # no detected Peak in '0.38-0.48'
            A = df_dado.iloc[0]['Oil Whirl'] > E2
                                                        #-----------yellow-----------------
                                                # Detected Peak in '0.38-0.48'
                                                #         but
                                                # Peak in '0.38-0.48' < 2% 1.0x
            B_peaks = PEAKS(E1,df_dado.iloc[0]['Oil Whirl'],df_dado.iloc[0]['1.0']) 
            B       = B_peaks and df_dado.iloc[0]['Oil Whirl'] > 0.02 * df_dado.iloc[0]['1.0']
     
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(not A         ,l1,
                                                   A and (not B) ,l2,
                                                   A and B       ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN    

#----------------------------------------------------------------------------12a
@jit 
def  DecissionTable_Ball_Bearing_Outer_Race_Defects_22217C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BPFO1'] < E2) 
            a2 = (df_dado.iloc[0]['BPFO1'] > E2) and (df_dado.iloc[0]['2*BPFO1'] < E2) and (df_dado.iloc[0]['3*BPFO1'] < E2) and (df_dado.iloc[0]['4*BPFO1'] < E2)
            A  = a1 or a2
            B  = PEAKS(E1,df_dado.iloc[0]['BPFO1'],df_dado.iloc[0]['2*BPFO1'])
            C  = PEAKS(E1,df_dado.iloc[0]['BPFO1'],df_dado.iloc[0]['2*BPFO1'],df_dado.iloc[0]['3*BPFO1'])
            D  = PEAKS(E1,df_dado.iloc[0]['BPFO1'],df_dado.iloc[0]['2*BPFO1'],df_dado.iloc[0]['3*BPFO1'],df_dado.iloc[0]['4*BPFO1'])
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   C     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   


#----------------------------------------------------------------------------12b
@jit 
def  DecissionTable_Ball_Bearing_Outer_Race_Defects_22219C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BPFO2'] < E2) 
            a2 = (df_dado.iloc[0]['BPFO2'] > E2) and (df_dado.iloc[0]['2*BPFO2'] < E2) and (df_dado.iloc[0]['3*BPFO2'] < E2) and (df_dado.iloc[0]['4*BPFO2'] < E2)
            A  = a1 or a2
            B  = PEAKS(E1,df_dado.iloc[0]['BPFO2'],df_dado.iloc[0]['2*BPFO2'])
            C  = PEAKS(E1,df_dado.iloc[0]['BPFO2'],df_dado.iloc[0]['2*BPFO2'],df_dado.iloc[0]['3*BPFO2'])
            D  = PEAKS(E1,df_dado.iloc[0]['BPFO2'],df_dado.iloc[0]['2*BPFO2'],df_dado.iloc[0]['3*BPFO2'],df_dado.iloc[0]['4*BPFO2'])
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   C     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   



#----------------------------------------------------------------------------13a
@jit 
def  DecissionTable_Ball_Bearing_Inner_Race_Defects_22217C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BPFI1'] < E2)
            a2 = (df_dado.iloc[0]['BPFI1'] > E2) and (df_dado.iloc[0]['2*BPFI1'] < E2)
            A  = a1 or a2 
            B  = (df_dado.iloc[0]['BPFI1']   > E2) and (df_dado.iloc[0]['2*BPFI1']  > E2)
            C  = (df_dado.iloc[0]['BPFI1']   > E2) and (df_dado.iloc[0]['BPFI1+f']   > E2) and (df_dado.iloc[0]['BPFI1-f']   > E2)
            D  = (df_dado.iloc[0]['2*BPFI1'] > E2) and (df_dado.iloc[0]['2*BPFI1+f'] > E2) and (df_dado.iloc[0]['2*BPFI1-f'] > E2) 
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   D     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   

#----------------------------------------------------------------------------13b
@jit 
def  DecissionTable_Ball_Bearing_Inner_Race_Defects_22219C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BPFI2'] < E2)
            a2 = (df_dado.iloc[0]['BPFI2'] > E2) and (df_dado.iloc[0]['2*BPFI2'] < E2)
            A  = a1 or a2 
            B  = (df_dado.iloc[0]['BPFI2']   > E2) and (df_dado.iloc[0]['2*BPFI2']  > E2)
            C  = (df_dado.iloc[0]['BPFI2']   > E2) and (df_dado.iloc[0]['BPFI2+f']   > E2) and (df_dado.iloc[0]['BPFI2-f']   > E2)
            D  = (df_dado.iloc[0]['2*BPFI2'] > E2) and (df_dado.iloc[0]['2*BPFI2+f'] > E2) and (df_dado.iloc[0]['2*BPFI2-f'] > E2) 
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   D     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   
#----------------------------------------------------------------------------14a
@jit 
def  DecissionTable_Ball_Bearing_Ball_Defect_22217C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BSF1']   < E2)
            a2 = (df_dado.iloc[0]['BSF1']   > E2) and (df_dado.iloc[0]['2*BSF1'] < E2)
            A  = a1 or a2 
            B  = PEAKS(E1,df_dado.iloc[0]['BSF1'],df_dado.iloc[0]['2*BSF1'])
            C  = PEAKS(E1,df_dado.iloc[0]['BSF1']  ,df_dado.iloc[0]['BSF1+FTF1']  ,df_dado.iloc[0]['BSF1-FTF1'])
            D  = PEAKS(E1,df_dado.iloc[0]['2*BSF1'],df_dado.iloc[0]['2*BSF1+FTF1'],df_dado.iloc[0]['2*BSF1-FTF1'])
            print(l1,l2,l3,A,B)

            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   D     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN       

#----------------------------------------------------------------------------14b
@jit 
def  DecissionTable_Ball_Bearing_Ball_Defect_22219C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = (df_dado.iloc[0]['BSF2']   < E2)
            a2 = (df_dado.iloc[0]['BSF2']   > E2) and (df_dado.iloc[0]['2*BSF2'] < E2)
            A  = a1 or a2 
            B  = PEAKS(E1,df_dado.iloc[0]['BSF2'],df_dado.iloc[0]['2*BSF2'])
            C  = PEAKS(E1,df_dado.iloc[0]['BSF2']  ,df_dado.iloc[0]['BSF2+FTF2']  ,df_dado.iloc[0]['BSF2-FTF2'])
            D  = PEAKS(E1,df_dado.iloc[0]['2*BSF2'],df_dado.iloc[0]['2*BSF2+FTF2'],df_dado.iloc[0]['2*BSF2-FTF2'])
            print(l1,l2,l3,A,B)
            
            df_Values_IN,l1,l2,l3 = decision_table(A     ,l1,
                                                   B ^ C ,l2,
                                                   D     ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN       

#----------------------------------------------------------------------------15a
@jit 
def  DecissionTable_Ball_Bearing_Cage_Defect_22217C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = PK(E1,SP_FingerPrint.iloc[n_golden]['RMS 1.0'])  
            a2 = df_dado.iloc[0]['FTF1'] < SP_FingerPrint.iloc[n_golden]['RMS 1.0']  
            b2 = df_dado.iloc[0]['FTF1'] > df_dado.iloc[0]['2*FTF1'] < SP_FingerPrint.iloc[n_golden]['RMS 1.0'] 
            c2 = df_dado.iloc[0]['FTF1'] > df_dado.iloc[0]['2*FTF1'] > SP_FingerPrint.iloc[n_golden]['RMS 1.0'] > df_dado.iloc[0]['3*FTF1'] > df_dado.iloc[0]['4*FTF1']
            
            A  = a1 and a2
            B  = a1 and b2
            C  = a1 and c2
            print(l1,l2,l3,A,B,C)

            df_Values_IN,l1,l2,l3 = decision_table(A ,l1,
                                                   B ,l2,
                                                   C ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   

#----------------------------------------------------------------------------15a
@jit 
def  DecissionTable_Ball_Bearing_Cage_Defect_22219C(SP_FingerPrint,Harmonics_IN,df_RD_specs_IN,df_env_specs_IN,df_Values_IN,n_reales_IN,n_golden):

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
            a1 = PK(E1,SP_FingerPrint.iloc[n_golden]['RMS 1.0'])  
            a2 = df_dado.iloc[0]['FTF2'] < SP_FingerPrint.iloc[n_golden]['RMS 1.0']  
            b2 = df_dado.iloc[0]['FTF2'] > df_dado.iloc[0]['2*FTF2'] < SP_FingerPrint.iloc[n_golden]['RMS 1.0'] 
            c2 = df_dado.iloc[0]['FTF2'] > df_dado.iloc[0]['2*FTF2'] > SP_FingerPrint.iloc[n_golden]['RMS 1.0'] > df_dado.iloc[0]['3*FTF2'] > df_dado.iloc[0]['4*FTF2']
            
            A  = a1 and a2
            B  = a1 and b2
            C  = a1 and c2
            print(l1,l2,l3,A,B,C,'=(',a1,c2,')')
            print(df_dado.iloc[0]['FTF2'] , df_dado.iloc[0]['2*FTF2'] , SP_FingerPrint.iloc[n_golden]['RMS 1.0'] , df_dado.iloc[0]['3*FTF2'] , df_dado.iloc[0]['4*FTF2'])
            df_Values_IN,l1,l2,l3 = decision_table(A ,l1,
                                                   B ,l2,
                                                   C ,l3,
                                                   df_Values_IN,df_dado.loc['0'],Harmonics_IN,n_reales_IN)
    return df_Values_IN   
#------------------------------------------------------------------------------    

if __name__ == '__main__':   
    pi       = np.pi
    E1       = 0.10
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
        'NumeroTramas' : '3',
        'Parametros'   : 'waveform' ,
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '12',#'12'
        'Hour'         : ''
        }
    
    n_random = 100 #---Numeroseñales sintéticas de cada tipo (Red, Green, Yellow)
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
    
    
    Process_variable1 = FailureMode('Severe_Misaligment',df_speed,df_SPEED)    #------tarda mucho en generar señales verdes
    Process_variable1.__func__(0,['1.0','2.0','3.0','4.0','5/2','7/2','9/2'],
                                 [4.6  ,4.6   ,1.0  ,0.5  ,0.1  ,0.1  , 0.1],
                                 [0.85 ,0.85  ,0.5  ,0.5  ,0.5  ,0.5  , 0.5],
                                 [10   ,10    ,1.4  ,0.9  ,0.2  ,0.2  , 0.2])
    
#
#    Process_variable2 = FailureMode('Loose_Bedplate',df_speed,df_SPEED) 
#    Process_variable2.__func__(0,['1.0','2.0','3.0'],
#                                 [4.8  ,0.9  ,0.9],
#                                 [1.2  ,0.5  ,0.5],
#                                 [10   ,1.2  ,2.4])
#    Process_variable2.__func_2__()
#    
#    Process_variable3 = FailureMode('Surge_Effect',df_speed,df_SPEED) 
#    Process_variable3.__func__(0,['Surge E. 0.33x 0.5x','Surge E. 12/20k'],
#                                  [0.05                 ,0.05],
#                                  [0.1                  ,0.1],
#                                  [0.7                  ,0.7])
#
#    Process_variable4 = FailureMode('Plain_Bearing_Lubrication_Whip',df_speed,df_SPEED) 
#    Process_variable4.__func__(0,['1/2','5/2'],
#                                 [0.05 ,0.1],
#                                 [0.5  ,0.5],
#                                 [0.7  ,0.7])
#
#    Process_variable5 = FailureMode('Plain_Bearing_Clearance',df_speed,df_SPEED) 
#    Process_variable5.__func__(0,['1.0','2.0','3.0','1/2','3/2','5/2'],
#                                 [4.6  ,1    ,0.9  ,0.9  ,0.5  ,0.3],
#                                 [1    ,0.5  ,0.5  ,0.5  ,0.01 ,0.5],
#                                 [10   ,3    ,2    ,2    ,1.5  ,1])
#    
#    Process_variable6 = FailureMode('Centrifugal_Fan_Unbalance',df_speed,df_SPEED) 
#    Process_variable6.__func__(0,['1.0','2th Max Value.'],[4.8,0.8],[1.2,0.5],[10,3])
#
#    Process_variable7 = FailureMode('Pressure_Pulsations',df_speed,df_SPEED) 
#    Process_variable7.__func__(0,['1/3','2/3','4/3','5/3','8/3','4.0'],
#                                 [0.1  ,0.3  ,0.2  ,0.2  ,0.15 ,0.1],
#                                 [0.5  ,0.5  ,0.5  ,0.5  ,0.5  ,0.5],
#                                 [0.5  ,1.2  ,1    ,1    ,0.7  ,0.5])
#    
#    Process_variable8 = FailureMode('Shaft_Misaligments',df_speed,df_SPEED) 
#    Process_variable8.__func__(0,['1.0','2.0','3.0','4.0','5.0'],
#                                 [3.5  ,5.2  ,1    ,0.5  ,0.1],
#                                 [0.85 ,0.85 ,0.5  ,0.5  ,0.5],
#                                 [6    ,10   ,1.4  ,0.9  ,0.2])
#    
#    Process_variable9 = FailureMode('Plain_Bearing_Block_Looseness',df_speed,df_SPEED) 
#    Process_variable9.__func__(0,['1.0','2.0','3.0','1/4','1/3','1/2'],
#                                 [3.5  ,5.2  ,1    ,0.2  ,0.15 ,0.1],
#                                 [0.85 ,0.85 ,0.5  ,0.5  ,0.5  ,0.5],
#                                 [6    ,10   ,1.4  ,0.4  ,0.3  ,0.2])    
#
#  
#    Process_variable10a = FailureMode('Blade_Faults',df_speed,df_SPEED) 
#    Process_variable10a.__func__(0,['1.0','12.0','24.0','11.0','13.0','23.0','25.0'],    #OK modificado
#                                   [3.5  ,0.3   ,0.2   ,0.1   ,0.1    ,0.1  ,0.1],
#                                   [0.85 ,0.5   ,0.5   ,0.6   ,0.6    ,0.6   ,0.6],
#                                   [6    ,1     ,0.8   ,0.2  ,0.2   ,0.2  ,0.2])
#    
#    Process_variable10b = FailureMode('Flow_Turbulence',df_speed,df_SPEED)  # OK modificada
#    Process_variable10b.__func__(0,['Flow T.','1.0'],
#                                   [3        ,  3.5],
#                                   [3        , 0.85],
#                                   [8        , 6])
#  
#    Process_variable11 = FailureMode('Plain_Bearing_Lubrication_Whirl',df_speed,df_SPEED)  # OK modificada
#    Process_variable11.__func__(0,['Oil Whirl','1.0'],
#                                  [0.1        ,3.5],
#                                  [0.5        ,2],
#                                  [0.7        ,6])
#
#    
#    Process_variable12a = FailureMode('Ball_Bearing_Outer_Race_Defects_22217C',df_speed,df_SPEED)  
#    Process_variable12a.__func__(0,['BPFO1','2*BPFO1','3*BPFO1','4*BPFO1'],
#                                  [1.5     ,1        ,0.5      , 0.35],
#                                  [1       ,0.85     ,0.5      ,0.5],
#                                  [4       ,2        ,1        ,0.8])  
#    Process_variable12a.__func_2__()
#    
#    Process_variable12b = FailureMode('Ball_Bearing_Outer_Race_Defects_22219C',df_speed,df_SPEED)  
#    Process_variable12b.__func__(0,['BPFO2','2*BPFO2','3*BPFO2','4*BPFO2'],
#                                  [1.5     ,1        ,0.5      ,0.35],
#                                  [1       ,0.85     ,0.5      ,0.5],
#                                  [4       ,2        ,1        ,0.8])       
##                   #------------------------------------------------------------
#    Process_variable13a = FailureMode('Ball_Bearing_Inner_Race_Defects_22217C',df_speed,df_SPEED)  
#    Process_variable13a.__func__(0,['BPFI1','2*BPFI1','BPFI1-f','BPFI1+f','2*BPFI1-f','2*BPFI1+f'],
#                                  [0.9     ,0.5      ,0.08     ,0.08     ,0.08       ,0.08],
#                                  [1       ,0.85     ,0.5      ,0.5      ,0.5        ,0.5],
#                                  [3       ,1.5      ,0.5      ,0.5      ,0.5        ,0.5])     
#    
#    Process_variable13b = FailureMode('Ball_Bearing_Inner_Race_Defects_22219C',df_speed,df_SPEED)  
#    Process_variable13b.__func__(0,['BPFI2','2*BPFI2','BPFI2-f','BPFI2+f','2*BPFI2-f','2*BPFI2+f'],
#                                  [0.9     ,0.5      ,0.08     ,0.08     ,0.08       ,0.08],
#                                  [1       ,0.85     ,0.5      ,0.5      ,0.5        ,0.5],
#                                  [3       ,1.5      ,0.5     ,0.5     ,0.5       ,0.5])  
#                   #------------------------------------------------------------    
#    Process_variable14b = FailureMode('Ball_Bearing_Ball_Defect_22217C',df_speed,df_SPEED)   #esta
#    Process_variable14b.__func__(0,['BSF1','2*BSF1','BSF1-FTF1','BSF1+FTF1','2*BSF1-FTF1','2*BSF1+FTF1'],
#                                  [1.5    ,1       ,0.1        ,0.1        ,0.1          ,0.1],
#                                  [1      ,0.85    ,1          ,1          ,1            ,1],
#                                  [4      ,2       ,0.5        ,0.5        ,0.5          ,0.5]) 
#
#    
#    Process_variable14b = FailureMode('Ball_Bearing_Ball_Defect_22219C',df_speed,df_SPEED)  
#    Process_variable14b.__func__(0,['BSF2','2*BSF2','BSF2-FTF2','BSF2+FTF2','2*BSF2-FTF2','2*BSF2+FTF2'],
#                                  [1.5    ,1       ,0.1        ,0.1        ,0.1          ,0.1],
#                                  [1      ,0.85    ,0.5        ,0.5        ,0.5          ,0.5],
#                                  [4      ,2       ,0.5       ,0.5       ,0.5         ,0.5])  
#                   #------------------------------------------------------------
#
#
#    Process_variable15a = FailureMode('Ball_Bearing_Cage_Defect_22217C',df_speed,df_SPEED)  # OK
#    Process_variable15a.__func__(0,['FTF1','2*FTF1','3*FTF1','4*FTF1'],
#                                   [2     ,2       ,1       ,0.8],
#                                   [2     ,2       ,0.5     ,0.5],
#                                   [4     ,4       ,2       ,1.5])  
#    
#    Process_variable15b = FailureMode('Ball_Bearing_Cage_Defect_22219C',df_speed,df_SPEED)  # OK
#    Process_variable15b.__func__(0,['FTF2','2*FTF2','3*FTF2','4*FTF2'],
#                                   [2     ,2       ,1       ,0.8],
#                                   [2     ,2       ,0.5     ,0.5],
#                                   [4     ,4       ,2       ,1.5]) 
#    
