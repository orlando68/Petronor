# -*- coding: utf-8 -*-
"""
Editor de Spyder

Bueno,
1. me he fijado en la fase para que me ayude a posicionar armonicos, y no estaba claro,
2. he promediado la fase con el filtro de savgol => nada interesante
3. he correlado el espectro con el pico maximo para ver si asi se detectaban más calmente los 
    otros pico, pero nada de nada.
4. si resto al modulo de la FFT su promediado con el filtro de savgol, tengo un 
    espectro mas plano, mas consecuente para buscar picos. Pero no concluyo nada
"""

import numpy as np
import scipy as sp
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
import json

#------------------------------------------------------------------------------ 
def US_corr(signal,in_signal):
    out    = np.zeros(np.size(signal))
    l_in_s = np.size(in_signal)
    length = np.size(signal)-l_in_s
    print (length)
    for i in np.arange(length):

        out[i] = np.sum(signal[i:i+l_in_s] * in_signal)
    return out

   
def pyt_espectro(waveform, fs,titulo,ylabel,**options):
   
    l           = np.size(waveform)  
    media       = np.mean(waveform)
    power       = np.std(waveform-media)
    
    wave_fft    = np.fft.fft(waveform)/l
    mod_sptrm   = np.abs(wave_fft)/np.sqrt(2)
    
    wave_fft_b  = np.fft.fft(waveform-media)/l
    mod_sptrm_b = np.abs(wave_fft_b)/np.sqrt(2)
    
    f           = np.arange(l)/l*fs
    f_rpm       = f * 60
    RBW         = fs/l
    if options.get("plot") == True:
        print ("mean :", media,";  power AC",power**2)
        maximo    = np.max(mod_sptrm_b)
        peaks,_   = find_peaks(mod_sptrm_b, height=maximo-10)
        #y         = signal.savgol_filter(mod_sptrm, 99, 1,mode='interp')
        l_mitad   = int(l/2)
        plt.figure()
        ax1 = plt.subplot(1,1,1)
        ax1.plot(f_rpm[0:l_mitad],mod_sptrm[0:l_mitad])
        ax1.set_xlabel('RPM')
        ax1.set_ylabel(ylabel)
        ax1.set_title(titulo+' RBW='+str(RBW)+'Hz')
        #plt.plot(f[0:l_mitad],y[0:l_mitad])
        #plt.plot(f[peaks], mod_sptrm[peaks],'o')
        
        ax2 = ax1.twiny()
        newlabel =np.arange(0,250,25)
        hz2rpm = lambda t: t*60 # convert function: from Kelvin to Degree Celsius
        newpos   = [hz2rpm(t) for t in newlabel] 
        ax2.set_xticks(newpos)
        ax2.set_xticklabels(newlabel)
        ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
        ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
        ax2.spines['bottom'].set_position(('outward', 36))
        ax2.set_xlabel('Hz')
        ax2.set_xlim(ax1.get_xlim())
        ax1.grid(True)
        plt.show()

    return mod_sptrm, f
  

#------------------------------------------------------------------------------    
def getKey(item):
    return item['AssetId']
#------------------------------------------------------------------------------
def unwrap_phase(phase):

    end = np.size(phase)
    xu  = phase
    
    
    for i in range(2, end):
        difference = phase[i]-phase[i-1]
        if difference > pi:
            xu[i:end] = xu[i:end] - 2*pi
        else:
            if difference < -pi:
                xu[i:end] = xu[i:end] + 2*pi
    return xu


#------------------------------------------------------------------------------
def STFT_waterfall(waveform,f_sampling):
    n          = 2**9
    f, t, Zxx  = signal.stft(waveform, fs=f_sampling, window = 'hann', nperseg=n,noverlap = n-1,nfft=1*n)
    
#    fmax       = 1120
#    fmin       = 1080
#    n_fmax     = np.int(np.size(f)*fmax/(f_sampling/2))
#    n_fmin     = np.int(np.size(f)*fmin/(f_sampling/2))
    
    f_DC       = 10
    n_f_DC     = np.int(np.size(f)*f_DC/(f_sampling/2))
    color_mesh = np.abs(Zxx)
    Vmin = np.min(color_mesh)
    Vmax = np.max(color_mesh[n_f_DC:,:]) #tomamos el valor mas grande que no sea DC
    plt.figure()
    #ax1 = plt.subplot(211)
    ax1 = plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=3)
    #plt.pcolormesh(t, f[n_fmin:n_fmax], np.abs(Zxx[n_fmin:n_fmax,:]), vmin=0, vmax=30)
    plt.pcolormesh(t, f, color_mesh, vmin=Vmin, vmax=Vmax)
    
    #plt.setp(ax1.get_xticklabels(), fontsize=6)
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    
    
    #ax2 = plt.subplot(212, sharex=ax1)
    ax2 = plt.subplot2grid((4,4), (3,0), sharex=ax1, colspan=4, rowspan=1)
    plt.plot(tiempo,acceleration)
    #plt.title('Captured signal')
    plt.ylabel('amplitude')
    #plt.xlabel('Time [sec]')
    plt.subplots_adjust(hspace=0)
    plt.tight_layout()
    plt.show()
    
    return f, t, Zxx








pi = np.pi

path = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\data'
file = 'sh3_json3.json'
file = 'Json2018.09.25.json'


with open(path+'//'+file, "r") as read_file:
    data = json.load(read_file)

counter      = 10
a            = data[counter]
print (counter,a['AssetId'],a['AssetName'])
acceleration = np.asarray(a['Value'])
l            = np.size(acceleration) 

b          = a['Props']
c          = b[0]
fs_m       = np.float(c['Value'])

c          = b[4]
cali       = np.float(c['Value'])
print ('calibracion',cali)
acceleration = acceleration * cali * 9.8 
speed      = 1000*np.cumsum(acceleration -np.mean(acceleration))
tiempo     = np.arange(l)  / fs_m

#STFT_waterfall(acceleration, fs_m)
titulo = 'Acceleration'
ylabel = 'RMS(m/s2)/RBW'
pyt_espectro(acceleration,fs_m,titulo,ylabel,plot = True)
titulo = 'Velocity'     
ylabel = 'RMS(mm/s)/RBW'       
pyt_espectro(speed       ,fs_m,titulo,ylabel,plot = True)



"""
xu1 =unwrap_phase(np.angle(Zxx[10, :]))
xu2 =unwrap_phase(np.angle(Zxx[15, :]))
plt.figure(3)
plt.plot(xu1)
plt.plot(xu2)
plt.show()

"""

#mod_sptrm, f = pyt_espectro(acceleration, fs_m)
#maximo = np.max(mod_sptrm)
#minimo = np.min(mod_sptrm)





#pota = sorted(data, key=getKey)
#for counter,a in enumerate(pota,0):
#    print (counter,a['AssetId'],a['AssetName'])
    
"""    

sensor_n    = 96#int(input('numero de sensor: '))
sensor_data = data[sensor_n]
y           = np.asarray(sensor_data['Value'])
l           = np.size(y)
media       = np.mean(y)
print ('longitud de muestra :',l,'media :', media)
y           = y - media


b           = a['Props']
c           = b[0]
fs          = np.float(c['Value'])
pyt_espectro(y, fs)






f = fs*np.arange(l)/l
Y = 20 *np.log10(np.abs( np.fft.fft(y/l)))

plt.figure(1,figsize=(16, 10), dpi=80)
peaks, _ = find_peaks(Y, height=5)
plt.subplot2grid((4,4), (0,0), colspan=4, rowspan=1)
plt.plot(y)
plt.title('Time domain')

plt.subplot2grid((4,4), (1,0), colspan=4, rowspan=3)
#plt.subplot(2, 1, 2)
plt.plot(f,Y,linewidth =0.4)
plt.plot(f[peaks],Y[peaks],'x')



plt.title('Freq domain')
plt.ylabel('Modulo')
plt.xlabel('Hz')
#plt.axis([0,fs/2,-10,60])

#plt.subplots_adjust(top=0)
plt.tight_layout()


plt.grid()
plt.show()
"""