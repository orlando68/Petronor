import requests
from PETRONOR_lyb import *


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
        'Localizacion' : 'SH3', #SH3/4
        'Source'       : 'Local Database', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-02-20T00:20:00.9988564Z',
        'FechaInicio'  : '2019-02-12T00:52:46.9988564Z',
        'NumeroTramas' : '10',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '11',
        'Day'          : '',#'12'
        'Hour'         : '' 
    }

    
    df_speed,df_SPEED_abs       = Load_Vibration_Data_Global(parameters)
    harm                        = df_Harmonics(df_speed,df_SPEED_abs, fs)
   
    harm                        = Blower_Wheel_Unbalance(harm)
    harm                        = Plain_Bearing_Clearance(harm)
    harm                        = Oil_Whirl(harm)
    harm                        = Oil_Whip(harm)
    harm                        = Blade_Faults(harm)
    harm                        = Flow_Turbulence(harm)
    harm                        = Plain_Bearing_Block_Looseness(harm)
    harm                        = Shaft_Misaligments(harm)
    harm                        = Pressure_Pulsations(harm)
    harm                        = Surge_Effect(harm)
    harm                        = Severe_Misaligment(harm)
    harm                        = Loose_Bedplate(harm)
   
    if parameters['Source'] == 'Petronor Server':
        fecha_label = parameters['Fecha'].split('-')[2].split('T')[0]+'_'+ parameters['Fecha'].split('-')[2].split('T')[0]+'_'+parameters['Fecha'].split('-')[0]
        print('>>>>>>>>>>>>>>>>>>',fecha_label)
        writer      = pd.ExcelWriter(Path_out+'Spectral_FP_Server_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+fecha_label+'.xlsx')
    else:
        fecha_label = parameters['Day']+'_'+parameters['Month']+'_2018'
        writer      = pd.ExcelWriter(Path_out+'Spectral_FP_Local_DB_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+fecha_label+'.xlsx')
        
    
    
    harm.to_excel(writer, 'DataFrame')
    writer.save()

    #find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    
    Plot_Spectrum(0,df_SPEED_abs,harm)
    #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak')
    #color,vertices = plot_waterfall(df_SPEED_abs,harm,fs,0,400)
    #plot_waterfall2(parameters,df_SPEED_abs,harm,fs,0,400)
    
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FINNNNNNNNNNNNNNNN')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
