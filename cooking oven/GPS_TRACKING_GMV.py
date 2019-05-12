#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 17:25:32 2018

@author: instalador
ESTE ES EL MAS MODERNO
Se detecta el fin del chip, para detectar exactamente el flanco de los bits
"""

import time
from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/instalador/GNSS/anaconda lib')
from orlando_lyb_anaconda import *
import CA_code_gen as CA
from time import gmtime, strftime

#------------------------------------------------------------------------------

def dayOfWeek(year, month, day): 
    "returns day of week: 0=Sun, 1=Mon, .., 6=Sat" 
    hr = 12  #make sure you fall into right day, middle is save 
    t = time.mktime((year, month, day, hr, 0, 0, 0, 0, -1)) 
    pyDow = time.localtime(t)[6] 
    gpsDow = (pyDow + 1) % 7 
    print ("day of week",gpsDow)
    return gpsDow

def timeOfWeek(year,month,day,hour,minute,seconds):
    day_week = dayOfWeek(year, month, day)
    out      = ( ( (day_week)*24 + hour )*60 + minute )*60 + seconds
    return out

def loop_filter(Bl,Kd,Ko):
    T      = 0.001
    dseta  = np.sqrt(2)/2
    wn     = 8*dseta*Bl/(4*dseta**2+1)
    k      = (1/Kd/Ko) * (8*dseta*wn*T) /(4+4*dseta*wn*T+(wn*T)**2)
    k_inte = (1/Kd/Ko) * (4*(wn*T)**2)  /(4+4*dseta*wn*T+(wn*T)**2)
    return wn,k,k_inte

#-----------------------------------------------------------------------------
@jit
def Sec_order_filter(k,k_inte,B,x):
    A   = x  * k_inte + B
    y   = (B + A)/2
    B   = A     
    out = (k * x  + y)
    return out,B 
#-----------------------------------------------------------------------------

def pyt_GPS_double (Nsat):#------SOLO se usa 1 vez principio TRACKING----------
        g=CA.CA_code_gen(Nsat)
        codigo=np.zeros(2046)
        for i in range(0,1023):
            codigo[2*i]   = g[i]
            codigo[2*i+1] = g[i]
      
        return codigo
"""
#------------------------------------------------------------------------------
@jit
def str2array(signal, output):    
    msb = [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,
        2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  3.,  3.,  3.,  3.,
        3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  4.,
        4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,
        4.,  4.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,
        5.,  5.,  5.,  5.,  5.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,
        6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  7.,  7.,  7.,  7.,  7.,
        7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7.,  7., -8., -8.,
       -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8., -8.,
       -8., -7., -7., -7., -7., -7., -7., -7., -7., -7., -7., -7., -7.,
       -7., -7., -7., -7., -6., -6., -6., -6., -6., -6., -6., -6., -6.,
       -6., -6., -6., -6., -6., -6., -6., -5., -5., -5., -5., -5., -5.,
       -5., -5., -5., -5., -5., -5., -5., -5., -5., -5., -4., -4., -4.,
       -4., -4., -4., -4., -4., -4., -4., -4., -4., -4., -4., -4., -4.,
       -3., -3., -3., -3., -3., -3., -3., -3., -3., -3., -3., -3., -3.,
       -3., -3., -3., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1.]

    lsb = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6., -5., -4.
           ,-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7.
           ,-6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.
           ,7., -8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.
           ,4.,  5.,  6.,  7., -8., -7., -6., -5., -4., -3., -2., -1.,  0.
           ,1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6., -5., -4., -3.
           ,-2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6.
           ,-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.
           ,-8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.
           ,5.,  6.,  7., -8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.
           ,2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6., -5., -4., -3., -2.
           ,-1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6., -5.
           ,-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8.
           ,-7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.
           ,6.,  7., -8., -7., -6., -5., -4., -3., -2., -1.,  0.,  1.,  2.
           ,3.,  4.,  5.,  6.,  7., -8., -7., -6., -5., -4., -3., -2., -1.
           ,0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7., -6., -5., -4.
           ,-3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.,  7., -8., -7.
           ,-6., -5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.,  6.
           ,7., -8., -7., -6., -5., -4., -3., -2., -1.]
   
    k=0
    for pointer in signal: 
        output[k]  = msb[pointer]
        k = k + 1
        output[k]  = lsb[pointer]
        k = k + 1
    return output
#------------------------------------------------------------------------------    
"""
@jit
def Inte_dump(sat,wei):   
     
    sat.CAR_f             = sat.CAR_f_i + sat.CAR_f_e 
    phase_carrier_array_r = sat.CAR_ph_i + 2*pi* tiempo * sat.CAR_f / fs 
    sat.CAR_ph_i          = np.mod(phase_carrier_array_r[l_1ms-1],2*pi)
    seno_r                = np.sin(phase_carrier_array_r)
    coseno_r              = np.cos(phase_carrier_array_r)
    seno_r                = np.multiply(array_samples , seno_r )
    coseno_r              = np.multiply(array_samples , coseno_r)
    #---------------------CODE---------------------
    f_code_r           = (2*1.023 + sat.COD_f_e + (sat.CAR_f - abs(1575.42-f_lo))/1540)/fs
    phase_code_array_r = sat.COD_ph_i + tiempo * f_code_r 
    sat.COD_ph_i       = np.mod(phase_code_array_r[l_1ms-1],1)
    indice_codigo_r    = np.mod(sat.COD_ind + phase_code_array_r,2046)
    index_cod_array_r  = np.int0( indice_codigo_r )            
    sat.COD_ind        = index_cod_array_r[l_1ms-1] 
    
    kk_early_r         = sat.COD_e[index_cod_array_r]  
    kk_promt_r         = sat.COD_p[index_cod_array_r]  
    kk_late_r          = sat.COD_l[index_cod_array_r] 
    #print('longutud', np.size(sat.COD_l),sat.COD_ind)
    #--------------------CALCULATIONS--------------
    
    Ip_array           = np.multiply(seno_r , kk_promt_r)
    chip_end           = np.argmin(index_cod_array_r)-1 # --acaba el bit  
    sec_p_OLD_bit      = np.sum(Ip_array[0:chip_end])
    fst_p_NEW_bit      = np.sum(Ip_array[chip_end:l_1ms])           
    bits_r             = (sat.COD_I_old + sec_p_OLD_bit) / l_1ms
    sat.COD_I_old      = fst_p_NEW_bit
    bit_end_time_r     = sat.l_OLD + chip_end /90000
    sat.l_OLD          = sat.l_OLD + 1
    
    Ip_r                  = (sec_p_OLD_bit + fst_p_NEW_bit) /l_1ms
    #Ip_r               = np.mean(np.multiply(seno_r  ,kk_promt_r))
    Qp_r                  = np.mean(np.multiply(coseno_r,kk_promt_r))
    Ie_r                  = np.mean(np.multiply(seno_r  ,kk_early_r))
    Qe_r                  = np.mean(np.multiply(coseno_r,kk_early_r))
    Il_r                  = np.mean(np.multiply(seno_r  ,kk_late_r ))
    Ql_r                  = np.mean(np.multiply(coseno_r,kk_late_r ))
    #--------------------NOISE---------------------
    noise_r               = np.std(array_samples)**2
    noise_dB_Hz_r         = 10*np.log10(noise_r)-10*np.log10(fs*1000000)
    CN_r                  = 10*np.log10(Ip_r**2+Qp_r**2) - noise_dB_Hz_r
    #--------------------CARRIER LOOP-------------- 
    disc_carrier_r        = np.sign(Ip_r*wei) * Qp_r*wei * Kd_carrier
    sat.CAR_f_e,sat.B_CAR = Sec_order_filter(k_carrier, k_inte_carrier, sat.B_CAR, disc_carrier_r)
    #--------------------CODE--- LOOP--------------
    disc_code_r           = ( (Ie_r-Il_r)*Ip_r + (Qe_r-Ql_r)*Qp_r)* Kd_code * wei**2
    sat.COD_f_e,sat.B_COD = Sec_order_filter(k_code,k_inte_code, sat.B_COD,disc_code_r)
        
    return sat, Ip_r,Qp_r, CN_r, bits_r,bit_end_time_r
        
#------------------------------------------------------------------------------
class sat_correlator:
    def __init__(self,        Nsat,COD_e,COD_p,COD_l,COD_f_e,COD_ph_i,COD_ind,B_COD,
                              l_OLD,COD_I_old,
                              CAR_f_i,CAR_f_e,CAR_f,CAR_ph_i,
                              B_CAR,a_e,a_p,a_l):
        self.Nsat   = Nsat        
        self.COD_e   = COD_e        
        self.COD_p   = COD_p 
        self.COD_l   = COD_l
        self.COD_f_e = COD_f_e 
        self.COD_ph_i  = COD_ph_i
        self.COD_ind = COD_ind
        self.B_COD   = B_COD
        
        self.l_OLD = l_OLD
        self.COD_I_old = COD_I_old
        
        self.CAR_f_i = CAR_f_i       
        self.CAR_f_e = CAR_f_e
        self.CAR_f   = CAR_f
        self.CAR_ph_i  = CAR_ph_i
        self.B_CAR   = B_CAR       
        self.a_e     = a_e
        self.a_p     = a_p
        self.a_l     = a_l
        self.l_OLD = l_OLD
       
    def __str__(self):
        s = ''.join(['Sat_Number     : ', str(self.Nsat), '\n',
                     'Doppler inic   : ', str(self.CAR_f_i), '\n',
                     'cod_freq error : ', str(self.COD_f_e),  '\n', 
                     'COD_ph         : ', str(self.COD_ph_i),  '\n', 
                                       
                     'carrier_freq e : ', str(self.CAR_f_e),  '\n',  
                     'carrier_freq i : ', str(self.CAR_f_i),  '\n',                
                     'CAR_ph         : ', str(self.CAR_ph_i),  '\n', 
                     '\n'])        
        return s        

class data_out:
    def __init__(self,        Nsat,bits,bit_end_time,Ip,CAR_f_e,COD_f_e,CN): 
        self.Nsat   = Nsat  
        self.bits   = bits 
        self.bit_end_time   = bit_end_time        
        self.Ip   = Ip 
        self.CAR_f_e   = CAR_f_e
        self.COD_f_e = COD_f_e 
        self.CN  = CN

    def __str__(self):
        s = ''.join(['Nsat     : ', str(self.Nsat), '\n', 
                     '\n'])        
        return s        


#------------------------------------------------------------------------------
strftime("%Y-%m-%d %H:%M:%S", gmtime())
inicio        = time.clock()

file_name     = 'CH1_1581_90'               ;fs = 90.0;f_lo = 1581.0;fi = f_lo-1575.42
file_name     = 'E1_15545_90'               ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo
file_name     = 'E1_15545_90_saw_narrow'    ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo 
file_name     = 'e1_15545_90_saw'           ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo;time_cap_UTC = "10:40:00";sat_list     = [8,10,16,18,21,26,27]
#file_name     = 'e1_15545_90_25_oct_11_03'  ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo;time_cap_UTC = "10:03:00";sat_list     = [8,10,16,27]
#file_name     = 'e1_15545_90_25_oct_11_01'  ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo 
#file_name     = 'e1_15545_90_saw_13_oct_12_11'  ;fs= 90;f_lo = 1554.5; fi = 1575.42- f_lo 

#path          ="D:\\Trabajo\\Proyectos\\043770-HORUS\\Phyton\\Windows\\capturas\\"
path          = "/disco/capture_store/"
pi            = np.pi

#sat_info      = np.load(path+file_name+'.npy')
sat_info      = np.load(path+'SEARCH_'+file_name+'_50Hz'+'.npy')
#=====================Rx Parameters============================================
#========================Wide loop settings-===================================
sat_list                 = [8,10,11,16,18,21,26,27] #[7,8,10,11,16,18,21,26,27][8,10,16,18,21,26,27]
inte_time                = 1   #ms
wide_loop_dewlling       = 0.1*1000 #ms
integrations_wide_loop   = np.int(wide_loop_dewlling/inte_time)
l_1ms                    = np.int(fs*1000000*inte_time/1000)
integrations             = integrations_wide_loop

tiempo                  = np.arange(1,l_1ms+1)
#-----------------Discriminator Gain & NCO Gain--------------------------------

Kd_carrier              = 0.05 
Ko_carrier              = 1000 # 1.0=>1000000Hz
Kd_code                 = 1
Ko_code                 = 1    #np.ones(integrations)
#----------------------LOOP FILTERS--------------------------------------------

Bl_carrier = 380
Bl_code    = 20
wn_carrier,k_carrier,k_inte_carrier = loop_filter(Bl_carrier,Kd_carrier,Ko_carrier)
wn_code,k_code,k_inte_code          = loop_filter(Bl_code,Kd_code,Ko_code)
print (wn_carrier,k_carrier,k_inte_carrier)
print (wn_code,k_code,k_inte_code)


#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#----------------------LOOP FILTERS--------------------------------------------
"""

k_carrier               =  0.002
k_inte_carrier           =  0.0002
k_code                  =  0.01 #0.001
k_inte_code             =  0.001 #0.0001


T                       = 0.001
dseta                   = np.sqrt(2)/2

Bl                      = 60
wn                      = 8*dseta*Bl/(4*dseta**2+1)
k_carrier               = (1/0.05/1000) * (8*dseta*wn*T) /(4+4*dseta*wn*T+(wn*T)**2)
k_carrier_inte          = (1/0.05/1000) * (4*(wn*T)**2)  /(4+4*dseta*wn*T+(wn*T)**2)
print ('<<<<<<<<<<<',wn,k_carrier,k_carrier_inte)
Bl                      = 60
wn                      = 8*dseta*Bl/(4*dseta**2+1)
k_code                  = (1/Kd_code/Ko_code) * (8*dseta*wn*T) /(4+4*dseta*wn*T+(wn*T)**2)
k_code_inte             = (1/Kd_code/Ko_code) * (4*(wn*T)**2)  /(4+4*dseta*wn*T+(wn*T)**2)
print ('<<<<<<<<<<<<<<',wn,k_code,k_code_inte) 
"""
#------------------------------------------------------------------------------
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#------------------------------------------------------------------------------

number_sats   = np.size(sat_list) 
offset_plot   = 0.3

S_EN          = []
Data_out_EN   = []
idx           = 0
for Nsat in sat_list:
    #print (Nsat,sat_info[0,Nsat-1])
    sat_ins            = sat_correlator(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    #======================Sat to receive==========================================
    sat_ins.Nsat      = Nsat
    sat_ins.COD_p      = 2.0*pyt_GPS_double(Nsat)-1
    sat_ins.COD_e      = np.roll(sat_ins.COD_p,-1)
    sat_ins.COD_l      = np.roll(sat_ins.COD_p, 1)
    sat_ins.COD_f_e    = 0
    sat_ins.COD_ph_i   = 0
    sat_ins.COD_ind    = np.mod(np.round(sat_info[3,Nsat-1]),2046)
    
    sat_ins.CAR_f_i    = (sat_info[2,Nsat-1]+0)/1000000 + fi 
    sat_ins.CAR_f_e    = 0
    sat_ins.CAR_f      = sat_ins.CAR_f_i + sat_ins.CAR_f_e
    sat_ins.CAR_ph_i   = 0
    f_carrier          = sat_ins.CAR_f_i + sat_ins.CAR_f_e
    
    f_code             = (2*1.023+ sat_ins.COD_f_e+ (f_carrier - abs(1575.42-f_lo)) /1540)/fs  
    sat_ins.a_e        = np.zeros(l_1ms)
    sat_ins.a_p        = np.zeros(l_1ms)
    sat_ins.a_l        = np.zeros(l_1ms)    
    S_EN.append(sat_ins)
    
    data_out_ins              = data_out(0,0,0,0,0,0,0)
    data_out_ins.Nsat        = Nsat
    data_out_ins.bits         = np.zeros(integrations)
    data_out_ins.bit_end_time = np.zeros(integrations)
    data_out_ins.Ip           = np.zeros(integrations)
    data_out_ins.CAR_f_e      = np.zeros(integrations)
    data_out_ins.COD_f_e      = np.zeros(integrations)
    data_out_ins.CN           = np.zeros(integrations)
    Data_out_EN.append(data_out_ins)
    
    idx                = idx+1

#print (S_EN[0])
#-----------------------ABRIR FICHERO--------------------------------------
weight               = np.ones(number_sats)
CN                   = np.zeros(number_sats)
f                    = open(path+'e1_15545_90_saw.bin', 'rb')

array_samples        = np.zeros (l_1ms)
inicio               = time.clock()
integrations_counter = 0

ccc                  = time.clock()
while integrations_counter<integrations:
    
    portion       = f.read(45000)
    array_samples = str2array(portion, array_samples) 
    for idx in range(number_sats):
       
        S_EN[idx],Ip,Qp,CN,bits,bit_end_time   = Inte_dump(S_EN[idx],weight[idx])
        Data_out_EN[idx].bits[integrations_counter]  = bits
        Data_out_EN[idx].bit_end_time[integrations_counter]  = bit_end_time
        Data_out_EN[idx].Ip[integrations_counter]  = Ip
        Data_out_EN[idx].CAR_f_e[integrations_counter]  = S_EN[idx].CAR_f_e
        Data_out_EN[idx].COD_f_e[integrations_counter] = S_EN[idx].COD_f_e
        Data_out_EN[idx].CN[integrations_counter]  = CN # phase_code = 1 => one chip
    
    
    if integrations_counter == 200:
        Bl_carrier = 20
        Bl_code    = 1
        for index,sat in enumerate(Data_out_EN):
            print (sat.Nsat)
            value = np.mean(np.abs(sat.bits))
            weight[int(index)]= value
        print (weight)
        maxi = np.max(weight)
        weight = maxi/weight
        wn_carrier,k_carrier,k_inte_carrier = loop_filter(Bl_carrier,Kd_carrier,Ko_carrier)
        wn_code,k_code,k_inte_code          = loop_filter(Bl_code,Kd_code,Ko_code)  
    """
    if integrations_counter == 300:
        Bl_carrier = 25
        Bl_code    = 1
        wn_carrier,k_carrier,k_inte_carrier = loop_filter(Bl_carrier,Kd_carrier,Ko_carrier)
        wn_code,k_code,k_inte_code          = loop_filter(Bl_code,Kd_code,Ko_code)
       
    if integrations_counter == 400:
        k_carrier               =  k_carrier     * 0.25
        k_inte_carier           =  k_inte_carier * 0.25
        k_code                  =  k_code        * 0.25 
        k_inte_code             =  k_inte_code   * 0.25
        
    if integrations_counter == 600:
        k_carrier               =  k_carrier     * 0.1
        k_inte_carier           =  k_inte_carier * 0.1
        k_code                  =  k_code        * 0.1
        k_inte_code             =  k_inte_code   * 0.1
    """
    integrations_counter                        = integrations_counter+1 
    print
ddd = time.clock()
print (ddd-ccc)

f.close()

fin = time.clock()
print ("tiempo :", fin-inicio)  
tiempo_int = np.arange(integrations)

plt.figure(1,figsize=(16, 8), dpi=80)

for idx in range(number_sats):
    
    
    plt.axes([0.03,0.70,0.96,0.3 ]) 
    #plt.plot(tiempo_int,data_RX[6*idx+2,:]+idx*0.2       ,label=str(sat_list[idx])+' '+str(np.round(sat_info[0,sat_list[idx]-1],4)))
    plt.plot(Data_out_EN[idx].bit_end_time,Data_out_EN[idx].bits-idx*0.25,label=str(sat_list[idx])+' '+str(np.round(sat_info[0,sat_list[idx]-1],4)) )
    plt.legend(bbox_to_anchor=(0.85, 1), loc=2, borderaxespad=0.)
    #plt.plot(data_RX[6*idx+5],data_RX[6*idx+0,:]+idx*0.2,'o')
    plt.grid(True)
    
    plt.axes([0.03,0.42,0.96,0.20 ]) 
    plt.title("k_carrier="+str(k_carrier)+"          " + "k_inte_carrier="+str(k_inte_carrier))
    plt.plot(Data_out_EN[idx].bit_end_time,1000000*Data_out_EN[idx].CAR_f_e)
    plt.grid(True)
    
    plt.axes([0.03,0.17,0.96,0.20 ])
    plt.title("k_code="+str(k_code)+"          " + "k_inte_code="+str(k_inte_code))
    plt.plot(Data_out_EN[idx].bit_end_time,1000000*Data_out_EN[idx].COD_f_e)
    plt.grid(True)
    
    plt.axes([0.03,0.03,0.96,0.10 ])
    plt.plot(Data_out_EN[idx].bit_end_time,Data_out_EN[idx].CN)
    
plt.show()

#np.save (path+file_name+'6',Data_out_EN)
