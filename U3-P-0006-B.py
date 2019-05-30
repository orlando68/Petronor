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
        'IdAsset'      : 'U3-P-0006-B',
        'Localizacion' : 'BV4', #BH3 (horizontal), BA4 (axial) y BV4 (vertical)
        'Source'       : 'Petronor Server', # 'Petronor Server'/'Local Database'
        
        'Fecha'        : '2019-05-21T20:00:46.9988564Z',
        'FechaInicio'  : '2018-10-12T00:52:46.9988564Z',
        'NumeroTramas' : '5',
        'Parametros'   : 'waveform',
        
        'Path'         : 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data\\Petronor\\data\\vibrations\\2018',
        'Month'        : '10',
        'Day'          : '11',#'12'
        'Hour'         : ''    
    }

    df_speed_BH3,df_SPEED_BH3,df_speed_BA4,df_SPEED_BA4,df_speed_BV4,df_SPEED_BV4 = Load_Vibration_Data_Global_Pumps(parameters)
    
    if parameters['Localizacion'] == 'BH3':    #-------horizontal Radial
       
        harm_BH3                  = df_Harmonics(df_SPEED_BH3, fs,'pump')
        
        harm_BH3                  = Recirculation_in_pump(harm_BH3)
        harm_BH3                  = Impeller_Rotor_Unbalance(harm_BH3)
        harm_BH3                  = Shaft_misaligment_Radial(harm_BH3)
        harm_BH3                  = Hydraulic_Instability(harm_BH3)
        harm_BH3                  = Ball_Bearing_Outer_Race_Defects_7310BEP(harm_BH3)
        harm_BH3                  = Ball_Bearing_Inner_Race_Defects_7310BEP(harm_BH3)
        harm_BH3                  = Ball_Bearing_Defect_7310BEP(harm_BH3)
        harm_BH3                  = Ball_Bearing_Cage_Defect_7310BEP(harm_BH3)
        harm_BH3                  = R_Rotating_Stall(harm_BH3)
        harm_BH3                  = Rotating_Cavitation(harm_BH3)
        harm_BH3                  = Cavitation_Noise(harm_BH3)
        harm_BH3                  = Piping_Vibration(harm_BH3)
        harm_BH3                  = Plain_Bearing_Clearance_pumps(harm_BH3)
        harm_BH3                  = Vane_Failure(harm_BH3)
        harm_BH3                  = Oil_Whirl_pumps(harm_BH3)
        harm_BH3                  = Oil_Whip_pumps(harm_BH3)
        harm_BH3                  = Auto_Oscillation(harm_BH3)
        
        check_test_results(harm_BH3)
        save_files(parameters,df_speed_BH3,df_SPEED_BH3,harm_BH3)
        plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',df_SPEED_BH3,harm_BH3,fs,45,55)
        
        Plot_Spectrum(0,df_SPEED_BH3,harm_BH3)
    
    if parameters['Localizacion'] == 'BV4':    #-------vertical Radial
        
        harm_BV4                  = df_Harmonics(df_SPEED_BV4, fs,'pump')
        
        harm_BV4                  = Recirculation_in_pump(harm_BV4)
        harm_BV4                  = Impeller_Rotor_Unbalance(harm_BV4)
        harm_BV4                  = Shaft_misaligment_Radial(harm_BV4)
        harm_BV4                  = Hydraulic_Instability(harm_BV4)
        harm_BV4                  = Ball_Bearing_Outer_Race_Defects_7310BEP(harm_BV4)
        harm_BV4                  = Ball_Bearing_Inner_Race_Defects_7310BEP(harm_BV4)
        harm_BV4                  = Ball_Bearing_Defect_7310BEP(harm_BV4)
        harm_BV4                  = Ball_Bearing_Cage_Defect_7310BEP(harm_BV4)
        harm_BV4                  = R_Rotating_Stall(harm_BV4)
        harm_BV4                  = Rotating_Cavitation(harm_BV4)
        harm_BV4                  = Cavitation_Noise(harm_BV4)
        harm_BV4                  = Piping_Vibration(harm_BV4)
        harm_BV4                  = Plain_Bearing_Clearance_pumps(harm_BV4)
        harm_BV4                  = Vane_Failure(harm_BV4)
        harm_BV4                  = Oil_Whirl_pumps(harm_BV4)
        harm_BV4                  = Oil_Whip_pumps(harm_BV4)
        harm_BV4                  = Auto_Oscillation(harm_BV4)
        
        check_test_results(harm_BV4)
        
        plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',df_SPEED_BV4,harm_BV4,fs,0,400)
        
        save_files(parameters,df_speed_BV4,df_SPEED_BV4,harm_BV4)
        Plot_Spectrum(0,df_SPEED_BV4,harm_BV4)
    
    if parameters['Localizacion'] == 'BA4':    #--------Axial
        harm_BA4                  = df_Harmonics(df_SPEED_BA4, fs,'pump')
        harm_BH3                  = df_Harmonics(df_SPEED_BH3, fs,'pump')
        harm_BV4                  = df_Harmonics(df_SPEED_BV4, fs,'pump')
        
        harm_BA4                  = Recirculation_in_pump(harm_BA4)
        harm_BA4                  = Impeller_Rotor_Unbalance(harm_BA4)
        harm_BA4                  = Shaft_misaligment_Axial(harm_BH3,harm_BV4,harm_BA4)
        harm_BA4                  = Hydraulic_Instability(harm_BA4)
        harm_BA4                  = Loosness_pumps(harm_BH3,harm_BA4)
        harm_BA4                  = Dynamic_instability(harm_BH3,harm_BA4)
        
        check_test_results(harm_BA4)
        
        plot_waterfall_lines(parameters['IdAsset']+' '+parameters['Localizacion']+' mm/sg RMS',df_SPEED_BA4,harm_BA4,fs,45,55)
        
        save_files(parameters,df_speed_BA4,df_SPEED_BA4,harm_BA4)
        #Plot_Spectrum(0,df_SPEED_BH3,harm_BH3)
        #Plot_Spectrum(0,df_SPEED_BV4,harm_BV4)
        Plot_Spectrum(0,df_SPEED_BA4,harm_BA4)

#   
    #find_closest(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    #Plot_rapido(df_speed,harm)
    #Plot_Spectrum(datetime.datetime(2018, int(month), int(day), 0, 0),df_SPEED_abs,harm)
    
    
    
    #PETROspectro(df_speed.iloc[0], fs,'Velocidad','mm/s',Detection = 'Peak') 
    #color,vertices = plot_waterfall(df_SPEED_abs,harm,fs,0,400)
    
    #plot_waterfall2(parameters,df_SPEED_BV4,harm_BV4,fs,0,400)
   
    #plot_waterfall27(parameters,df_SPEED_abs,fs,0,400)
    
    print('-------------------------------FIN----------------------------------')

    ####POST

    # WE ARE NOT ALLOWED TO POST DATA TO THE SERVER YET
    #requests.post('/api/Models/SetResultModel', output=OUTPUT)
