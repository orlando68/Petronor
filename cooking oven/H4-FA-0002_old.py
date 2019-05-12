import requests
from PETRONOR_lyb import *


#------------------------------------------------------------------------------

#--------------------------------------------------------------------------------
if __name__ == '__main__':

    # input parameters for API call
    # Funciona de tal modo que se obtienen el número de tramas o valores (si hay) especificados en 'NumeroTramas' desde 'Fecha' hacia atrás y hasta 'FechaInicio'.
    # NumeroTramas prioridad sobre FechaInicio
    parameters = {
        'IdPlanta'     : 'BPT',
        'IdAsset'      : 'H4-FA-0002',
        'Fecha'        : '2018-10-12T00:52:46.9988564Z',
        'FechaInicio'  : '2018-10-12T00:52:46.9988564Z',
        'NumeroTramas' : '1',
        'Parametros'   : 'waveform',
        'Localizacion' : 'SH4' #SH3/4
    }

    #path      = '/Users/inigo/Desktop/petronor'
    path      = 'C:/OPG106300/TRABAJO/Proyectos/Petronor-075879.1 T 20000/Trabajo/python'

    # extract month from key 'Fecha' in parameters object
    month     = parameters['Fecha'].split('-')[1]
    # extract day from key 'Fecha' in parameters object
    day       = parameters['Fecha'].split('-')[2].split('T')[0]
    hour      = parameters['Fecha'].split('-')[2].split('T')[1]
    
    # construct API endpoint
    api_endpoint = ('http://predictivepumpsapi.azurewebsites.net/api/Models/GetInfoForModel?IdPlanta=' + parameters['IdPlanta'] + \
    '&IdAsset=' + parameters['IdAsset'] + '&Fecha=' + parameters['Fecha'] + '&FechaInicio=' + parameters['FechaInicio'] + \
    '&NumeroTramas=' + parameters['NumeroTramas'] + '&Parametros=' + parameters['Parametros'])

    # make GET request to API endpoint
    response = requests.get(api_endpoint)
    # convert response from server into json object
    response_json = response.json()

    # check if server is up or not. They usually restart the server every day...
    # check if there are Nones in 'ValoresTrama'
    flag = response_json[0]['waveform'][0]['ValoresTrama']
    
    # if there are Nones
    if None in flag:
        print("Server is down...")
    # if there are not, run script functions
    else:

        df_accel,l_devices          = Load_Vibration_Data_From_Get(response_json, parameters)
        df_SPEED_abs = velocity(df_accel)
        
        harm                        = df_Harmonics(df_SPEED_abs, fs)
       
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
       

        writer = pd.ExcelWriter('result_'+parameters['IdAsset']+'_'+parameters['Localizacion']+'_'+month+'_'+day+'.xlsx')
        harm.to_excel(writer, 'DataFrame')
        writer.save()

        find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)

        #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak')
        #color,vertices = plot_waterfall(df_SPEED,fs,0,400)
        plot_waterfall2(df_SPEED_abs,fs,0,400)

        print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<FINNNNNNNNNNNNNNNN')

        ####POST

        # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
        #requests.post('/api/Models/SetResultModel', output=OUTPUT)
