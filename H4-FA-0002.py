import requests
from PETRONOR_lyb import *
import pandas as pd


#--------------------------------------------------------------------------------' 
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Localizacion' : 'SH3', #SH3/4
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-05-20T00:00:00.00Z',
        'FechaInicio'  : '2019-02-13T00:52:46.9988564Z',
        'NumeroTramas' : '1',
        'Parametros'   : 'waveform',
        
#        'Path'         : 'C://OPG106300//TRABAJO//Proyectos//Petronor-075879.1 T 20000//Trabajo//data//Petronor//data//vibrations//2018',
        'Path'         : '//home//instalador//Mantenimiento//data//2018',
        'Month'        : '10',
        'Day'          : '01',#'12'
        'Hour'         : '01' 
    }

    df_speed,df_SPEED           = Load_Vibration_Data_Global(parameters)
    harm                        = df_Harmonics(df_SPEED, fs,'blower')

    harm                        = Centrifugal_Fan_Unbalance(harm)
    harm                        = Plain_Bearing_Clearance(harm)
    harm                        = Plain_Bearing_Lubrication_Whirl(harm)
    harm                        = Plain_Bearing_Lubrication_Whip (harm)
    harm                        = Blade_Faults(harm)
    harm                        = Flow_Turbulence(harm)
    harm                        = Plain_Bearing_Block_Looseness(harm)
    harm                        = Shaft_Misaligments(harm)
    harm                        = Pressure_Pulsations(harm)
    harm                        = Surge_Effect(harm)
    harm                        = Severe_Misaligment(harm)
    harm                        = Loose_Bedplate(harm)
    
    check_test_results(harm)
    save_files(parameters,df_speed,df_SPEED,harm)
        
    #find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    
    Plot_Spectrum(0,df_SPEED,harm)
    #Plot_Spectrum_log(0,df_SPEED,harm)
    #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak')
    #color,vertices = plot_waterfall(df_SPEED_abs,harm,fs,0,400) 
    #plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',df_SPEED,harm,fs,0,400) #---Este
    #plot_waterfall27(parameters,df_SPEED_abs,harm,fs,0,400)
    
    """
    AVG = df_SPEED.abs().sum().div(df_SPEED.shape[0]).values
    df_SPEED_AVG = pd.DataFrame(data     = AVG,
                                 index   = df_SPEED.columns.values, 
                                 columns = [df_SPEED.index[0]]  )
    df_SPEED_AVG = df_SPEED_AVG.T
    harm_AVG                        = df_Harmonics(df_SPEED_AVG, fs,'blower')

    harm_AVG                        = Centrifugal_Fan_Unbalance(harm_AVG)
    harm_AVG                        = Plain_Bearing_Clearance(harm_AVG)
    harm_AVG                        = Plain_Bearing_Lubrication_Whirl(harm_AVG)
    harm_AVG                        = Plain_Bearing_Lubrication_Whip (harm_AVG)
    harm_AVG                        = Blade_Faults(harm_AVG)
    harm_AVG                        = Flow_Turbulence(harm_AVG)
    harm_AVG                        = Plain_Bearing_Block_Looseness(harm_AVG)
    harm_AVG                        = Shaft_Misaligments(harm_AVG)
    harm_AVG                        = Pressure_Pulsations(harm_AVG)
    harm_AVG                        = Surge_Effect(harm_AVG)
    harm_AVG                        = Severe_Misaligment(harm_AVG)
    harm_AVG                        = Loose_Bedplate(harm_AVG)
    Plot_Spectrum(0,df_SPEED_AVG,harm_AVG)
    print('-------------------------------FIN----------------------------------')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
    """