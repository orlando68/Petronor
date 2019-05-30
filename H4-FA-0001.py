import requests
from PETRONOR_lyb import *


#------------------------------------------------------------------------------
Path_out = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\outputs\\'
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0001',
        'Localizacion' : 'MH2', #SH4/MH2
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-20T00:00:00.00Z',
        'FechaInicio'  : '2019-02-14T00:00:00Z',
        'NumeroTramas' : '10',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '',#'12'
        'Hour'         : ''    
    }

    
    df_speed,df_SPEED = Load_Vibration_Data_Global(parameters)
    harm              = df_Harmonics(df_SPEED, fs,'blower')
    
    if parameters['Localizacion'] == 'SH4':
        harm                  = Pillow_Block_Loseness(harm)           # not for MH2
        harm                  = Blower_Wheel_Unbalance(harm)          # not for MH2
        harm                  = Blade_Faults(harm)                    # not for MH2
        harm                  = Flow_Turbulence(harm)                 # not for MH2
        harm                  = Pressure_Pulsations(harm)             # not for MH2
        harm                  = Surge_Effect(harm)                    # not for MH2
    
    harm                  = Severe_Misaligment(harm)
    harm                  = Loose_Bedplate(harm) 
    harm                  = Ball_Bearing_Outer_Race_Defects_22217C(harm) #ok
    harm                  = Ball_Bearing_Outer_Race_Defects_22219C(harm) #ok
    
    harm                  = Ball_Bearing_Inner_Race_Defects_22217C(harm) #ok
    harm                  = Ball_Bearing_Inner_Race_Defects_22219C(harm) #ok
    
    harm                  = Ball_Bearing_Ball_Defect_22217C(harm)        #ok
    harm                  = Ball_Bearing_Ball_Defect_22219C(harm)        #ok
    
    harm                  = Ball_Bearing_Cage_Defect_22217C(harm)        #ok
    harm                  = Ball_Bearing_Cage_Defect_22219C(harm)        #ok
    
    check_test_results(harm)
    
    save_files(parameters,df_speed,df_SPEED,harm)
    
    #find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    #Plot_rapido(df_speed,harm)
    #Plot_Spectrum(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    
    Plot_Spectrum(0,df_SPEED,harm)
    
    #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak')
    #color,vertices = plot_waterfall(df_SPEED_abs,harm,fs,0,400) 
    
    plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',df_SPEED,harm,fs,0,400)

    print('-------------------------------FIN----------------------------------')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
