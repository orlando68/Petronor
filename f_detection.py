import requests
from PETRONOR_lyb import *


#--------------------------------------------------------------------------------
Path_out = '/home/instalador/Mantenimiento/data/outputs/'
#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
#        'IdAsset'      : 'H4-FA-0002',
#        'Localizacion' : 'SH3', #SH3/4
        'IdAsset'      : 'U3-P-0006-B',
        'Localizacion' : 'BV4', #BH3 (horizontal), BA4 (axial) y BV4 (vertical)
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-04-13T00:00:00.9988564Z',
        'FechaInicio'  : '2019-02-12T00:52:46.9988564Z',
        'NumeroTramas' : '30',
        'Parametros'   : 'waveform',
        
        'Path'         : '/home/instalador/Mantenimiento/data/2018',
        'Month'        : '10',
        'Day'          : '',#'12'
        'Hour'         : '' 
    }

    df_speed,df_SPEED           = Load_Vibration_Data_Global(parameters)
    harm                        = df_shortcut(df_SPEED, fs,'pump')
#    harm                        = df_Harmonics(df_SPEED, fs,'pump')
#    Plot_Spectrum(0,df_SPEED,harm)
    

    #plot_waterfall27(parameters,df_SPEED_abs,harm,fs,0,400)
    print('-------------------------------FIN----------------------------------')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
