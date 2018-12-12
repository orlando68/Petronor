#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 19:06:39 2018

@author: instalador
Mas evolucionado que 
NAVIGATION
NAVIGATION_vatiable_rate
Introduce correccion Ionosferica
Calcula la posicion para n puntos a partir del primer TOW
El incremento de tiempo entre solución PVT es 1 bit
El error/offset de tiempo en el punto inicial de aproximación es cero. 'tu_xyz= 0'
El error/offset de tiempo en el punto a calcular es cero.              'dtime = 0'
En cada calculo de posición 
    - el punto inicial de aproximación es la solución anterior
    - el offset de tiempo en el punto a calcular es el delat de tiempo caculado
    por mínmo cuadrados de la solución anterior                        'dtime = dtime  + d_t'

"""

import numpy as np
#import scipy.optimize as opt
#from scipy import ndimage
#import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import datetime as dt
import time
#from sympy import Point3D, Line3D, Plane

from gmplot import gmplot
import webbrowser
#------------------------------------------------------------------------------
def expand_array(input):
    Npoints = 10
    output = np.array([])
    word    = np.zeros(Npoints)
    for i in range(Npoints):
        word[i]= i
    for k in input:
        output = np.append(output,word+k)
        
    return output.astype(int)

def sats_2_take(original_tuple, sats_to_take):
    new_tuple = []
    for s in list(original_tuple):
        if s.Nsat  in sats_to_take:
            #print(s.Nsat)
            new_tuple.append(s)
    return tuple(new_tuple)

def Rn(phi):
    Rn = a_wgs84 / np.sqrt(1- (e_earth*np.sin(phi))**2 )
    return Rn

def ECEF_2_geodetic(point):
    
    point_out          = coord_geo(0,0,0)
    point_out.longitud = np.arctan2(point.y,point.x) /pi *180
    r          = np.sqrt(point.x**2+point.y**2+point.z**2)
    p          = np.sqrt(point.x**2+point.y**2)
    phi_now      = np.arctan2(p,r)
    for inx in np.arange(12):
        
        RN = Rn(phi_now)
        h = p /np.cos(phi_now) - RN
        phi_now = np.arctan(OP_xyz.z/( p*(1- e_earth**2 *( RN/(RN+h) ) ) ))
        #print (h, phi_now)
    
    #print(phi_now/pi*180,point_out.longitud,h)    
    point_out.latitud = phi_now/pi*180
    point_out.altitud       = h    
    return point_out

def geodetic_2_ECEF(point_in):
    point_out   = coord_ECEF(0,0,0)
    lat         = point_in.latitud /180*np.pi
    lon         = point_in.longitud/180*np.pi
    h           = point_in.altitud
    point_out.x = ( a_wgs84 * np.cos(lon) / np.sqrt( 1 + (1-e_earth**2)*np.tan(lat)**2 ) )                + h * np.cos(lon) * np.cos(lat)
    point_out.y = ( a_wgs84 * np.sin(lon) / np.sqrt( 1 + (1-e_earth**2)*np.tan(lat)**2 ) )                + h * np.sin(lon) * np.cos(lat)
    point_out.z = ( a_wgs84 * (1-e_earth**2) * np.sin(lat) / np.sqrt( 1-(e_earth**2) * np.sin(lat)**2 ) ) + h * np.sin(lat)
    return point_out

def dayOfWeek(year, month, day): 
    "returns day of week: 0=Sun, 1=Mon, .., 6=Sat" 
    hr     = 12  #make sure you fall into right day, middle is save 
    t      = time.mktime((year, month, day, hr, 0, 0, 0, 0, -1)) 
    pyDow  = time.localtime(t)[6] 
    gpsDow = (pyDow + 1) % 7 
    print ("day of week",gpsDow)
    return gpsDow

def Zcount_to_seconds(tow):
    #calcula los segundos desde las 00:00 del ultimo dia
    day_s       = 24*60*60
    hour_s      = 60*60
    minute_s    = 60
    day_week    = np.int(np.floor(tow/day_s)) # de 0 a 6
    hour        = np.int(np.floor( (tow-day_week*day_s)  / hour_s))
    minute      = np.int(np.floor( ( tow-((day_week*day_s) + hour*hour_s) ) /minute_s))
    seconds     = np.int(tow-  (day_week*day_s + hour*hour_s + minute* minute_s )) 
    sg_last_day = tow-day_week * day_s
    #print day+1,' ',hour,':',minute,':',seconds
    #print tow-day*day_s, hour*hour_s+minute*minute_s+seconds
    return day_week,hour, minute,seconds,sg_last_day

def distance_from_sats(sat_list,point):
    
    distances =np.array([])    
    for sat in sat_list:
        distance  = np.sqrt((sat.x-point.x)**2+(sat.y-point.y)**2+(sat.z-point.z)**2)
        distances = np.append(distances,distance)
    return distances
#-------------------------------------------------------------------------------   
#-------------------------------------------------------------------------------  
def parity(SF_entrada,sign): #---------word tiene valores +/-1
    word_names      = ('TLM','HOW','WD3','WD4','WD5','WD6','QD7','WD8','WD9','W10')
    par_check_total = 0
    Zcount          = 0
    if sign ==1:
        texto = '  >>>>>>>>>>>>NO INVERTIDOS'
    else:
        texto = '  <<<<<<<<<<<<INVERTIDOS'
    SF = np.zeros(302,dtype=int)
    
    for index in range(0,302):
        SF[index] = int((sign*SF_entrada[index]+1)/2) #tiene valores 0 y 1

    d       = np.zeros(30+1,dtype=int) # source bits (Elemento cero NO se usa)
    D       = np.zeros(30+1,dtype=int) # transmitted bits
    D29o    = np.int(SF[0])            # previous TX bits
    D30o    = np.int(SF[1])            # previous TX bits 

    #print '<<<<<<<<<<<<>>>>>>>>>>>>>>',SF[2:2+8]
    for word_i in range(10):
        d       = np.zeros(30+1,dtype=int)
        D[1:31] = SF[2 +30*word_i:2 +30*(word_i+1)]
        #print 'limit',2 +30*word_i,2 +30*(word_i+1)-1
        #print 'd[1:31]',d[1:31]
        #print 'D[1:31]',D[1:31]
        #----------------------------------------
        for i in range(1,24+1):
            d[i] =D[i] ^ D30o
            
        d[25] = D29o^d[1]^d[2]^d[3]^d[5]^d[6]^d[10]^d[11]^d[12]^d[13]^d[14]^d[17]^d[18]^d[20]^d[23]
        d[26] = D30o^d[2]^d[3]^d[4]^d[6]^d[7]^d[11]^d[12]^d[13]^d[14]^d[15]^d[18]^d[19]^d[21]^d[24]
        d[27] = D29o^d[1]^d[3]^d[4]^d[5]^d[7]^d[8] ^d[12]^d[13]^d[14]^d[15]^d[16]^d[19]^d[20]^d[22]
        d[28] = D30o^d[2]^d[4]^d[5]^d[6]^d[8]^d[9] ^d[13]^d[14]^d[15]^d[16]^d[17]^d[20]^d[21]^d[23]
        d[29] = D30o^d[1]^d[3]^d[5]^d[6]^d[7]^d[9] ^d[10]^d[14]^d[15]^d[16]^d[17]^d[18]^d[21]^d[22]^d[24]
        d[30] = D29o^d[3]^d[5]^d[6]^d[8]^d[9]^d[10]^d[11]^d[13]^d[15]^d[19]^d[22]^d[23]^d[24]
                
        par_check = 0    
        for k in range(25,1+30): 
            par_check   = par_check + (2*d[k]-1) * (2*D[k]-1)
        par_check_total = par_check_total + par_check
        #print  word_names[word_i],par_check
        if word_i ==1:
            Zcount =''
            for i in range(1,18):
                Zcount = Zcount + str(d[i])
            Zcount = int(Zcount,2)
        D29o    = np.int(SF[0 +30*(word_i+1)])
        D30o    = np.int(SF[1 +30*(word_i+1)])
        
      
    #if par_check_total == 60: # hemos recibido SF buena
    #    print 'Zcount',Zcount, 'SF ID', SF[51:54]
    return par_check_total,Zcount
#-------------------------------------------------------------------------------  
def find_Zcount(Sat_index, BS,data_matrix,data_RX):
    cont_bits = 0
    
    # 25 frames. 1 frame = 30 seg => 25*30= 12.5m 
    #  1 frame = 5 subframes. 1 subframe (6sg)= 10 words. 10*30 = 300
    #  1 word  = 30 bits.
    word_bits    = 30
    samples_bit  = 20 #1miliseg
    SF_l         = 10 *word_bits * samples_bit
    SubF_length  = 10 *word_bits
    prble_l      = 8*samples_bit
    TLM_HOW_l    = int(2*word_bits*samples_bit)
                    
    bit_counter  = 0
    preamble_p   = [ 1,-1,-1,-1, 1,-1, 1 ,1]
    correlation  = 0
    tiempo_GPS   = 0
    kv           = 0.5
    SF           = np.zeros(300+2,dtype=np.integer) # buffer de 300 + 2 pq necesito los 2 precentes para parity. Aqui meto 1 SF 
    
    polarity     = 1
    
    for index in range(integrations): 

        BS.int_dump = BS.int_dump + data_RX[Sat_index].bits[index] 
        #--------------------------------detectar + edge en datos------------------    
        if data_RX [Sat_index].bits[index]-BS.value_prev > 0.055:
            BS.data_p_edge = True
        #-------------------------------phase del VCO------------------------------
        BS.phase_VCO      = 2 * np.pi * (45 +kv*BS.f_error)/fs_i  + BS.phase_VCO    
        if BS.phase_VCO > 2*np.pi:#----edge en el VCO
            BS.CLK_p_edge    = True
            BS.phase_VCO     = BS.phase_VCO - 2*np.pi   
            BS.cont2edge     = 0            
        #----------------------------RF FF-----------------------------------------
        if BS.data_p_edge  == True:
            BS.RS_FF         = 1 
            BS.s_armed       = True
        if BS.CLK_p_edge == True and BS.s_armed == True:
            BS.RS_FF         = 0
            BS.time2volt_out = BS.time2volt
            BS.time2volt     = 0
            BS.s_armed       = False    
        #----------------------------Time 2 Voltage -------------------------------
        BS.time2volt              = BS.time2volt + BS.RS_FF   
        BS.f_error                = BS.time2volt_out
        #==============================        
        if BS.cont2edge == int(BS.time2volt_out): #-------detectamos un flanco+ = 1 bit-------------------
            #print ('flanco')            
            bit_counter                        = bit_counter +1
            bit                                = int(np.sign(BS.int_dump))
            #print BS.int_dump
            
            data_matrix [Sat_index].bits       = np.append(data_matrix [Sat_index].bits     , bit)
            data_matrix [Sat_index].bits_time  = np.append(data_matrix [Sat_index].bits_time,data_RX[Sat_index].bit_end_time[index])
            # aqui tengo los bits y el momento en el que inicia cada bit            
            
            data_matrix [Sat_index].clk[index]  = 0.03
            SF          = np.roll(SF,-1)     #-------desplazamos el resgistro hacia a izda
            SF[301]     = bit                #------metemos el bit en el registro por la derecha de [0 a 301]
            cont_bits   = cont_bits+1
            operador    = np.dot(SF[2:10],preamble_p ) #-----correlacion
            if np.abs(operador) == 8: #-----posible comienzo de TLM------------
                #print ('sat',Sat_index,'cont_bits',cont_bits)
                correlation,Zcontador = parity(SF,np.sign(operador))
                
                if correlation == 60:
                    polarity                                                       = np.sign(operador)
                    tiempo_GPS                                                     = Zcontador * 6
                    print('Sat',sat_list[Sat_index],'HOW word found. TOW=',tiempo_GPS)
                    back_1SF                                                       = index- SF_l #retrocedo 10Word_bits = 1subframe
                    data_matrix[Sat_index].preambl_AR[back_1SF:back_1SF+prble_l]   = polarity * data_RX[Sat_index].bits[back_1SF : back_1SF+prble_l]
                    data_matrix[Sat_index].events[index]                           = back_1SF
                    data_matrix[Sat_index].Z_count_AR[index]                       = tiempo_GPS
                    data_matrix[Sat_index].sbf_limits_AR[back_1SF]                 = polarity * data_RX [Sat_index].bits[back_1SF]                    
                    data_matrix[Sat_index].TLM_HOW_AR[back_1SF:back_1SF+TLM_HOW_l] = polarity * data_RX [Sat_index].bits[back_1SF:back_1SF+TLM_HOW_l]                   
                    
                    data_matrix[Sat_index].SF_s_index                              = np.append(data_matrix[Sat_index].SF_s_index, np.size(data_matrix [Sat_index].bits )-1 - SubF_length)
                    data_matrix[Sat_index].TOW                                     = np.append(data_matrix[Sat_index].TOW,tiempo_GPS)
                    data_matrix[Sat_index].TOW_index                               = np.append(data_matrix[Sat_index].TOW_index, np.size(data_matrix [Sat_index].bits)-1)
                    
            BS.int_dump = 0 
        
        BS.cont2edge                             = BS.cont2edge +1
        data_matrix [Sat_index].indump_AR[index] = BS.int_dump
        BS.value_prev                            = data_RX[Sat_index].bits[index]
        BS.data_p_edge                           = False
        BS.CLK_p_edge                            = False
         
    data_RX [Sat_index].bits = data_RX [Sat_index].bits * polarity
    data_matrix [Sat_index].bits = data_matrix [Sat_index].bits * polarity
    print('------------------------------------------')
    
    return data_matrix,data_RX

#------------------------------------------------------------------------------- 

def seek_ephem_in_file(path,n_fichero,N_sat,TOW_24h_sg):
    #TOW_24h_sg  : en sg ultimo dia UTC
    ephemeris      = np.zeros(32+5)
    time_epoch_old = 0  
    #print 'TOW_24h_sg',TOW_24h_sg
    #---solo usamos h, m y segundos para buscar el ephemerides en el fichero
    s_captura_UTC  = TOW_24h_sg
    new_path       = path+n_fichero+'/'
    fichero        = open(new_path+n_fichero,'r')
    
    #li = linea.split(' ')
    #print (li)
    #-----------------------------CABECERA FICHERO-----------------------------
    linea = fichero.readline()
    linea = fichero.readline()
    linea = fichero.readline()
    linea = fichero.readline()
    l     = linea.split(' ')
    A     = l[4],l[6],l[7],l[8]
    for k in np.arange(4):
        a            = A[k]
        a            = a.split('D')
        ephemeris[k] =  np.float(a[0])*10**np.int(a[1]) 
        
    linea = fichero.readline()
    l     = linea.split(' ')
    B     = l[4],l[6],l[7],l[8]
    for k in np.arange(4):
        b = B[k]
        b = b.split('D')
        ephemeris[k+4] =  np.float(b[0])*10**np.int(b[1]) 
               
    linea = fichero.readline()
    linea = fichero.readline()
    linea = fichero.readline()
 
    #--------primera interacion    
    eof = False
    while eof == False:
        linea  = fichero.readline()
        sat    = np.int(linea[0:2])
        
        if sat == N_sat:
            #date       = linea[3:11]
            hour       = np.int(linea[12:14])
            minute     = np.int(linea[15:18])
            time_epoch = hour*3600 + minute *60 #segundos pasados desde las 00:00
            
            if s_captura_UTC < time_epoch:
                """
                print ('Sat:',N_sat,',',"Hora de Ephem inmediatamente posterior a captura: ",str(dt.timedelta(seconds=time_epoch)), '>>>>>', time_epoch,"sg")
                """
                break
            
            ephemeris[1+5] = np.float(linea[3+1*19:18+1*19]) * 10 ** np.int(linea[19+1*19:22+1*19])                          
            ephemeris[2+5] = np.float(linea[3+2*19:18+2*19]) * 10 ** np.int(linea[19+2*19:22+2*19])                
            ephemeris[3+5] = np.float(linea[3+3*19:18+3*19]) * 10 ** np.int(linea[19+3*19:22+3*19])          
            k = 4+5
            for i in range(7):                 
                linea = fichero.readline() #------------------Linea 1-------------
                ephemeris[k] = np.float(linea[3+0*19:18+0*19]) * 10 ** np.int(linea[19+0*19:22+0*19])                
                k            = k + 1
                ephemeris[k] = np.float(linea[3+1*19:18+1*19]) * 10 ** np.int(linea[19+1*19:22+1*19])                
                k            = k + 1
                ephemeris[k] = np.float(linea[3+2*19:18+2*19]) * 10 ** np.int(linea[19+2*19:22+2*19])                
                k            = k + 1 
                ephemeris[k] = np.float(linea[3+3*19:18+3*19]) * 10 ** np.int(linea[19+3*19:22+3*19])
                k            = k + 1
                time_epoch_old = time_epoch
        else: #-----------avanzo 7 lines al siguiente sat
            for k in range(7):
                fichero.readline()
        if linea == "":
            eof = True
    fichero.close()
    print ('HOW subframe T_GPS >>>>>>>>>>',str(dt.timedelta(seconds=s_captura_UTC)) ,'>>>>>',"%.3f" % s_captura_UTC, 'sg/day')
    print ('Ephemeris time from file:>>>>',str(dt.timedelta(seconds=time_epoch_old)),'>>>>>',"%.3f" % time_epoch_old,'sg/day')
    #print (ephemeris)
    
    return time_epoch_old,ephemeris
#------------------------------------------------------------------------------    

def ECCF_sat(path,nombre_f, sat_info_EN, OP,TOW_r):
    """
    sat_list     : lista de sats
    t_2_cal      : instante exacto para calcular la posicion del satelite
                   puede ser un array en formato TOW_r para varios sats 
                   o una hora comun para varios sats ss en el ultimo dia
    OP           : punto de observacion
    """
    #------------------------------OUTPUTS-------------------------------------
        
    nu          = 3986005 * 1e+8       #  m3/sec2
    OMEGA_DOT_e = 7.2921151467 * 1e-5  #  rad/sg

    lat         = OP.latitud /180*np.pi
    lon         = OP.longitud/180*np.pi
    
    OP_xyz      = geodetic_2_ECEF(OP)
    x_OP        = OP_xyz.x 
    y_OP        = OP_xyz.y
    z_OP        = OP_xyz.z
    #print ('Punto inicial',x_OP,y_OP,z_OP)
    #------------------------------------VECTOR unitario normal a tierra en OP-
    ux          = np.cos(lat)*np.cos(lon)
    uy          = np.cos(lat)*np.sin(lon)
    uz          = np.sin(lat)
    #------------------------------------VECTOR unitario apuntando al norte----
    d_plane     = -1*(ux*x_OP + uy*y_OP + uz*z_OP) # ecuacion del plano tg a GOIDE en OP  
    z1          = -1*d_plane/  uz  # vector apuntando desde OP a el punto de inteseccion plano con ejej Z????
    modulo      = np.sqrt(x_OP**2 + y_OP**2 + (z1-z_OP)**2)    
    vx          = (0 -x_OP)/modulo
    vy          = (0 -y_OP)/modulo
    vz          = (z1-z_OP)/modulo
    #print 'perpen', ux*vx+uy*vy+uz*vz
    
    sat_index     = 0
    for N_sat in sat_list:
    
        TOW_24h_sg   = np.mod(TOW_r[sat_index],24*3600)
        t_epoch,ephem = seek_ephem_in_file(path,nombre_f,N_sat,TOW_24h_sg)
        """
        print('TOW >>>>>>>>>>>>>>>>>>>>',TOW_r[sat_index],'>>>>',str(dt.timedelta(seconds=TOW_r[sat_index])))
        print('TOW => seg en el dia>>>>',TOW_24h_sg,'>>>>>>>>>>>>>',str(dt.timedelta(seconds=TOW_24h_sg)))
        print('Hora sg ephem >>>>>>>>>>',t_epoch,'>>>>>>>>>>>>>>>',str(dt.timedelta(seconds=t_epoch)))
        """
        sat_info_EN[sat_index].CLK_bias              = ephem[1+5]# + ephem[2] *t +ephem[3]*t**2
        sat_info_EN[sat_index].CLK_drift             = ephem[2+5]
        sat_info_EN[sat_index].CLK_drift_rate        = ephem[3+5]
        sat_info_EN[sat_index].CLK_correction        = ephem[1+5] + ephem[2+5] *TOW_24h_sg +ephem[3+5]*TOW_24h_sg**2
        sat_info_EN[sat_index].Crs       = Crs       = ephem[5+5]
        sat_info_EN[sat_index].Delta_n   = Delta_n   = ephem[6+5]
        sat_info_EN[sat_index].M_0       = M_0       = ephem[7+5]
        sat_info_EN[sat_index].Cuc       = Cuc       = ephem[8+5]
        sat_info_EN[sat_index].e         = e         = ephem[9+5] 
        sat_info_EN[sat_index].Cus       = Cus       = ephem[10+5] 
        sat_info_EN[sat_index].sqrt_A    = sqrt_A    = ephem[11+5]
        sat_info_EN[sat_index].t_oe      = t_oe      = ephem[12+5] #(sec of GPS week) 7*24*60*60 = 604800
        sat_info_EN[sat_index].Cic       = Cic       = ephem[13+5] 
        sat_info_EN[sat_index].OMEGA     = OMEGA     = ephem[14+5]
        sat_info_EN[sat_index].Cis       = Cis       = ephem[15+5]
        sat_info_EN[sat_index].i0        = i0        = ephem[16+5]
        sat_info_EN[sat_index].Crc       = Crc       = ephem[17+5] 
        sat_info_EN[sat_index].omega     = omega     = ephem[18+5]
        sat_info_EN[sat_index].OMEGA_DOT = OMEGA_DOT = ephem[19+5]
        sat_info_EN[sat_index].IDOT      = IDOT      = ephem[20+5]


        A         = sqrt_A**2
        n         = np.sqrt(nu/A**3) + Delta_n                            # Corrected mean motion
        t_k       = (TOW_24h_sg + 1*sat_info_EN[sat_index].CLK_correction ) -t_epoch # Time from ephemeris ephoc
        #print ('TOW_24h_sg-t_epoch >>>>',t_k,'>>>>>>>>>>>>>>',str(dt.timedelta(seconds=t_k)))
        M_k       = M_0 + n * t_k                                         # Mean anomaly
        
        E_k_ante  = M_k
        E_k       = 0
        k         = 0
        while M_k+ e * np.sin(E_k_ante) != E_k_ante: 
            E_k      = M_k+ e * np.sin(E_k_ante)                          # Excentric Anomaly 
            E_k_ante = E_k
            #print (k,E_k_ante,E_k)
            k        = k+1 
            if k == 10:
                print ('<<<<<<<<<<<<<<<<<<<<<<<Numero iteraciones ',k, E_k)
                break
        sin_v_k = np.sqrt(1-e**2) * np.sin(E_k) / (1-e*np.cos(E_k)) 
        cos_v_k = (np.cos(E_k) - e)             / (1-e*np.cos(E_k)) 
        v_k     = np.arctan2( sin_v_k,cos_v_k)                             # Ture anomaly
        
        arg_lat = omega + v_k                                              # Argument of Latitude
        duk     = Cus * np.sin(2*arg_lat) + Cuc * np.cos(2*arg_lat)        # Arg. of Lat. correc
        drk     = Crs * np.sin(2*arg_lat) + Crc * np.cos(2*arg_lat)        # Radius correc
        dik     = Cis * np.sin(2*arg_lat) + Cic * np.cos(2*arg_lat)        # Incli. correc
        
        uk      = arg_lat               + duk                              # Correc. Argu Lat.
        rk      = A * (1-e*np.cos(E_k)) + drk                              # Correc. Radius
        ik      = i0 + IDOT * t_k       + dik                              # Correc. Inclination
        
        OMEGA_k = OMEGA + (OMEGA_DOT-OMEGA_DOT_e)*t_k - OMEGA_DOT_e*t_oe   # Cor. Long Node
        xp      = rk * np.cos(uk)                                          # In plane x position
        yp      = rk * np.sin(uk)                                          # In plane y position
        
        xk      = xp * np.cos(OMEGA_k) - yp * np.cos(ik) * np.sin(OMEGA_k) # ECCF X-coord
        yk      = xp * np.sin(OMEGA_k) + yp * np.cos(ik) * np.cos(OMEGA_k) # ECCF Y-coord
        zk      = yp * np.sin(ik)                                          # ECCF y-coord
                
        sat_info_EN[sat_index].x  = xk
        sat_info_EN[sat_index].y  = yk
        sat_info_EN[sat_index].z  = zk
        
        sat_info_EN[sat_index].distance = np.sqrt((x_OP-xk)**2+(y_OP-yk)**2+(z_OP-zk)**2)
        #print ('distancia',distance[sat_index])
        
        #-----------------VECTOR_unit OP => proyeccion_sat sobre d_plane 
        #-----------------AZIMUT: angulo medido desde el NORTE clockwise-------
        tpam    = (-1*d_plane-1*(xk*ux + yk*uy + zk*uz)) / (ux**2+uy**2+uz**2)
        #---------------------coordenadas punto interseccion-------------------
        xp      = xk + tpam*ux  
        yp      = yk + tpam*uy
        zp      = zk + tpam*uz
        mod1    = np.sqrt((xp-x_OP)**2 + (yp-y_OP)**2 +(zp-z_OP)**2) 
        sat_info_EN[sat_index].di_in_plane = mod1                          # distancia OP => Sat_porject in d_plane 
                #--------------------------------------------------------------        
        v_xp    = (xp-x_OP)/mod1
        v_yp    = (yp-y_OP)/mod1
        v_zp    = (zp-z_OP)/mod1
        #print 'vetor', v_xp,v_yp,v_zp
        azimut  = v_xp*vx + v_yp*vy + v_zp*vz 
        #----------------------------------CORRECCION del azimut---------------
        az_OP_P  = np.arctan2(y_OP,x_OP)                                   # Longitud del OP
        az_s_p   = np.arctan2(yk,xk)                                       # longitud del Satelite
        
        if  az_s_p < az_OP_P:
            aviso = 'el satelite esta a OESTE de OP, => azimut > PI'
            azimut_o = 2*np.pi-np.arccos(azimut)
            #print (aviso)
        else:                            
            aviso = 'el satelite esta a ESTE de OP, => azimut < PI'   
            azimut_o = np.arccos(azimut)    
            #print (aviso)
        
        #----------------vector unitario en direccion al sat-------------------
        v_module                         = np.sqrt( (xk-x_OP)**2 + (yk-y_OP)**2 +(zk-z_OP)**2 )
        ux_sat                           = (xk-x_OP) /v_module
        uy_sat                           = (yk-y_OP) /v_module
        uz_sat                           = (zk-z_OP) /v_module
    
        elevation                        = ux_sat*ux + uy_sat*uy + uz_sat*uz 
        elevation                        = np.pi/2-np.arccos(elevation)
        
        sat_info_EN[sat_index].elevation = elevation                       # radiands
        sat_info_EN[sat_index].azimut    =  azimut_o                       # radiands

        #------------------IONOSPHERE------------------------------------------4        
        psi                              = 0.0137/(elevation + 0.11) - 0.022
        if np.abs(psi) > 0.416:
            psi  = 0.416*np.sign(psi)
        phi_i                            = OP.latitud/180*np.pi + psi * np.cos(azimut_o)
        l_i                              = OP.longitud/180*np.pi +psi*np.sin(elevation)/np.cos(phi_i)     
        phi_m                            = phi_i + 0.064*np.cos(l_i-1.617)       
        t_ion                            = 4.32*10**4  * l_i + TOW_24h_sg    
        F                                = 1.0 + 16.0 *(0.53-elevation)**3        
        AMP                              = 0
        for indx in np.arange(4):
            AMP = AMP + ephem[indx+0]*phi_m**indx          
        if AMP < 0:
            AMP = 0
        PER                              = 0
        for indx in np.arange(4):
            PER = PER + ephem[indx+4]*phi_m**indx        
        if PER >72000:
            PER = 72000
        X                                = 2*pi*(t_ion-50400)/PER        
        if np.abs(X) < 1.57:
            T_iono = F * (5*10**-9 + AMP*(1-X*X/2 + X**4 /24) )
        if np.abs(X) >= 1.57:
            T_iono = F * (5*10**-9  )        
        sat_info_EN[sat_index].T_iono    = T_iono
        """
        print('psi   ',psi)
        print('phi_i ',phi_i)
        print('l_i   ',l_i)
        print('phi_m ',phi_m)
        print('t_ion ',t_ion)
        print('F     ',F)
        print('AMP,PER,X',AMP,PER,X)
        print('T_iono',T_iono)
        """
        #---------------------------------------------------------------------
        sat_index = sat_index+1
    
    
    #fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
    """
    ax = plt.subplot(111, projection='3d')
    #----------------PLOT 3D---------------------------------------------------
    
    #----------------EJES------------------------------------------------------
    ax.quiver(0, 0, 0, 1.5*a_wgs84, 0, 0,linewidth=2,arrow_length_ratio= 0.1)
    ax.quiver(0, 0, 0, 0, 1.5*a_wgs84, 0,linewidth=2,arrow_length_ratio= 0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.5*a_wgs84,linewidth=2,arrow_length_ratio= 0.1)
    
    coefs = (a_wgs84**2, a_wgs84**2, b_wgs84**2)  
    # Coefficients in  x**2 / (a0*c)+ y**2 (a1*c)+ z**2 (a2*c) = 1 
    # Radii corresponding to the coefficients:
    rx, ry, rz = np.sqrt(coefs)
    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    # Plot:
    #ax.plot_surface(x, y, z,  rstride=10, cstride=10, color='Y')
    ax.plot_wireframe(x, y, z,  rstride=10, cstride=10, color='y',linewidth=0.1, antialiased=False)
    sat_index     = 0
    for  N_sat  in sat_list:
        azim          = sat_info_EN[sat_index].azimut    * 180/np.pi 
        elev          = sat_info_EN[sat_index].elevation * 180/np.pi
        ax.scatter(sat_info_EN[sat_index].x,sat_info_EN[sat_index].y,sat_info_EN[sat_index].z)
        plt.quiver(x_OP,y_OP,z_OP, sat_info_EN[sat_index].x-x_OP, sat_info_EN[sat_index].y-y_OP, sat_info_EN[sat_index].z-z_OP, linewidth=.5,arrow_length_ratio= 0.01)
        ax.text(sat_info_EN[sat_index].x, sat_info_EN[sat_index].y, sat_info_EN[sat_index].z, 'Sat:'+str(N_sat)+'\n A: %.2f' % azim+'º'+'\n E: %.2f' % elev+'º',size=8,color='black')
        sat_index = sat_index +1 

    #ax.scatter(Xs,Ys,Zs,c='r')
    ax.scatter(x_OP,y_OP,z_OP,c='b')
    ax.view_init(elev=45, azim=45)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    #----------------PLOT polar------------------------------------------------
    ax = plt.subplot(111, projection='polar')        
    ax.set_theta_direction(-1)
    #ax.set_rmax(maximo)
    #ax.set_rticks(np.arange(maximo))  # less radial ticks
    sat_index     = 0
    for N_sat in sat_list:
        azim                             = sat_info_EN[sat_index].azimut    * 180/np.pi  
        elev                             = sat_info_EN[sat_index].elevation * 180/np.pi
        ax.plot(sat_info_EN[sat_index].azimut,  sat_info_EN[sat_index].di_in_plane,'o')
        ax.text(sat_info_EN[sat_index].azimut,  sat_info_EN[sat_index].di_in_plane,'Sat:'+str(N_sat)+'\n A: %.2f' % azim+'º'+'\n E: %.2f' % elev+'º',size=8,color='red')
        sat_index = sat_index+1
    ax.set_theta_zero_location("N")
    #ax.set_rlabel_position(0)  # get radial labels away from plotted line
    ax.grid(True)

    ax.set_title("GPS constellation", va='bottom')
    plt.show()   
    """
    return sat_info_EN,OP_xyz

#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
class BIT_synchronizer:
    
    def __init__(self, value_prev, data_p_edge, CLK_p_edge, time2volt, 
                 time2volt_out, phase_VCO, f_error,  RS_FF, s_armed,
                 cont2edge,int_dump):
                     
        self.value_prev    = value_prev
        self.data_p_edge   = data_p_edge
        self.CLK_p_edge    = CLK_p_edge
        self.time2volt     = time2volt
        self.time2volt_out = time2volt_out
        self.phase_VCO     = phase_VCO
        self.f_error       = f_error
        self.RS_FF         = RS_FF
        self.s_armed       = s_armed
        self.cont2edge     = cont2edge
        self.int_dump      = int_dump

class coord_geo:
    def __init__(self, latitud,longitud,altitud):
        self.longitud = longitud
        self.latitud  = latitud
        self.altitud  = altitud
        
class coord_ECEF:
    def __init__(self, x,y,z):
        self.x = x
        self.y = y
        self.z = z

class data_process:
    def __init__(self,        Nsat,clk,indump_AR,Z_count_AR,events,events_pre
                             ,TLM_HOW_AR,preambl_AR,sbf_limits_AR
                             ,bits,bits_time,SF_s_index,TOW,TOW_index): 
        
        self.Nsat          = Nsat  
        self.clk           = clk            # RELOJ DEL BIT SINCHRONIZER
        self.indump_AR     = indump_AR      # RESULTADO INTERNO DEL BIT SINCHRONIZER  
        self.Z_count_AR    = Z_count_AR     # TIEMPO GPS                                      ---> equiposicionado con events_
        self.events        = events         # IMARCADOR => INDICE DEL COMIENZO DE LA SUBFRAME ---> equiposicionado con Z_count_AR_ 
        self.events_pre    = events_pre     # NO SE USA      
        self.TLM_HOW_AR    = TLM_HOW_AR     # PORCION DE BIT QUE INDICAN EL TIEMPO GPS  
        self.preambl_AR    = preambl_AR     # PORCION DE BIT QUE PUEDEN SER UN PREAMBLE
        self.sbf_limits_AR = sbf_limits_AR  # UN PUNTO MARCADOR=>VALOR DE LOS BIT EN EL MOMENTO DE INICIO DE SUBFRAME [events_]
        
        self.bits          = bits
        self.bits_time     = bits_time
        self.SF_s_index    = SF_s_index     # INDEX DENTRO DEL ARRAY DE BIT
        self.TOW           = TOW
        self.TOW_index     = TOW_index
        
    def __str__(self):
        s = ''.join(['Nsat     : ', str(self.Nsat), '\n', 
                     '\n'])        
        return s        
    
class data_out:
    def __init__(self,        Nsat,bits,bit_end_time,Ip,CAR_f_e,COD_f_e,CN): 
        self.Nsat         = Nsat  
        self.bits         = bits 
        self.bit_end_time = bit_end_time        
        self.Ip           = Ip 
        self.CAR_f_e      = CAR_f_e
        self.COD_f_e      = COD_f_e 
        self.CN           = CN

    def __str__(self):
        s = ''.join(['Nsat     : ', str(self.Nsat), '\n', 
                     '\n'])        
        return s        
        
class sat_info:
    def __init__(self,        Nsat,x,y,z,CLK_bias,elevation,azimut,distance,di_in_plane
                             ,CLK_drift,CLK_drift_rate,CLK_correction
                             ,Crs,Delta_n,M_0,Cuc,e,Cus,sqrt_A,t_oe,Cic
                             ,OMEGA,Cis,i0,Crc,omega,OMEGA_DOT,IDOT,T_iono): 
        self.Nsat           = Nsat  
        self.x              = x 
        self.y              = y        
        self.z              = z 
        self.elevation      = elevation
        self.azimut         = azimut
        self.distance       = distance
        self.di_in_plane    = di_in_plane
        
        self.CLK_bias       = CLK_bias
        self.CLK_drift      = CLK_drift 
        self.CLK_drift_rate = CLK_drift_rate
        self.CLK_correction = CLK_correction
        
        self.Crs            = Crs
        self.Delta_n        = Delta_n
        self.M_0            = M_0
        self.Cuc            = Cuc
        self.e              = e
        self.Cus            = Cus
        self.sqrt_A         = sqrt_A
        self.t_oe           = t_oe #(sec of GPS week) 7*24*60*60 = 604800
        self.Cic            = Cic
        
        self.OMEGA          = OMEGA
        self.Cis            = Cis
        self.i0             = i0
        self.Crc            = Crc
        self.omega          = omega
        self.OMEGA_DOT      = OMEGA_DOT
        self.IDOT           = IDOT
        self.T_iono         = T_iono
        

    def __str__(self):
        s = ''.join(['Nsat     : ', str(self.Nsat), '\n',
                     '\n'])        
        return s       
#xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    

fs           = 90.0
fs_i         = 1000
pi           = np.pi 
c            = 2.99792458 * 10**8  #km/s
a_wgs84      = 6378137
b_wgs84      = 6356752.3142
e_earth      = np.sqrt(1-(b_wgs84/a_wgs84)**2)
#path            ="D:\\Trabajo\\Proyectos\\043770-HORUS\\Phyton\\Windows\\capturas\\" 
path         = "/disco/capture_store/"
#---------------------------datos de partida-----------------------------------
#----------------ROll over between Saturady and Sunday-------------------------
hora_cap     = dt.time(10, 40, 00);time_cap = 3*24*3600+10*3600+40*60+0 #--UTC- seia a las 10:30 sino no salen las cuentas
date         = '21/09/2016';nombre_f= "brdc2650.16n"
#Data_out_EN     = np.load(path+"e1_15545_90_sawBITS_RX_ana_8_10_16_18_21_27.npy")

#Data_out_EN     = np.load(path+"e1_15545_90_sawBITS_RX_ana.npy")#------ESTE
#Data_out_EN     = np.load(path+"e1_15545_90_sawBITS_RX_ana_long.npy")
Data_out_EN     = np.load(path+"e1_15545_90_saw3.npy")
Data_out_EN     = np.load(path+"e1_15545_90_saw5.npy")

#Data_out_EN     = np.load(path+"e1_15545_90_sawBITS_RX_ana_60_new.npy")
#Data_out_EN     = np.load(path+"e1_15545_90_sawBITS_RX_ana_30_wideloop.npy")

#data_in       = np.load(path+"E1_15545_90_saw_RX_6sats.npy")
#data_in       = np.load(path+"e1_15545_90_sawBIS_RX_cython.npy")
#data_in       = np.load(path+"e1_15545_90_sawBITS_RX_ana.npy")

sat_list        = [8,10,16,18,21,26,27]
sat_list        = [8,10,16,18,21,26,27]
Data_out_EN     = sats_2_take(Data_out_EN, sat_list)
print(Data_out_EN)
#-------------------------------Pasar de ms a sg-------------------------------
Sat_index       = 0
for sat in sat_list:
    Data_out_EN[Sat_index].bit_end_time = Data_out_EN[Sat_index].bit_end_time /1000
    Sat_index                    = Sat_index +1

integrations    = np.size(Data_out_EN[0].bits)
number_sats     = np.size(Data_out_EN)

word_bits       = 30
samples_bit     = 20
n_SF            = 0       #S---------elected SubFramedata_process_ins
data_process_EN = []
sat_info_EN     = []
idx             = 0
for idx in np.arange(number_sats):
    #print (Nsat,sat_info[0,Nsat-1])
    data_process_ins               = data_process(0,0,0,0,0,0,0,0,0
                                                  ,0,0,0,0,0)
    data_process_ins.Nsat          = Data_out_EN[idx].Nsat
    data_process_ins.clk           = np.zeros(integrations)
    data_process_ins.indump_AR     = np.zeros(integrations)
    data_process_ins.Z_count_AR    = np.zeros(integrations)
    data_process_ins.events        = np.zeros(integrations)
    data_process_ins.events_pre    = np.zeros(integrations)
    data_process_ins.TLM_HOW_AR    = np.zeros(integrations)
    data_process_ins.preambl_AR    = np.zeros(integrations)
    data_process_ins.sbf_limits_AR = np.zeros(integrations)
    
    data_process_ins.bits          = np.array([])
    data_process_ins.bits_time     = np.array([])
    data_process_ins.SF_s_index    = np.array([])
    data_process_ins.TOW           = np.array([])
    data_process_ins.TOW_index     = np.array([])
    data_process_EN.append(data_process_ins)
    sat_info_ins                   = sat_info(0,0,0,0,0,0,0,0,0,0,0,0,0,0
                                             ,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
    sat_info_ins.Nsat              = Data_out_EN[idx].Nsat
    sat_info_EN.append(sat_info_ins)
    idx                            = idx+1

#E700         = coord_geo(43.296806,-2.870631,0) #E700 N43,2842 O2,8632
OP           = coord_geo(43 + 17  *0.5/30 + 49.656 *0.01/36, -1*(2 + 52 *0.5/30 + 13.708 *0.01/36),20) #E700 N43.29712668215136 O-2.8704744547571863
OP_xyz       = geodetic_2_ECEF(OP)
Rotonda      = coord_geo(43.29663,-2.87068,63.8)
Paris        = coord_geo(48.864716,2.349014,36)
Init_P       = Paris
P_init_xyz   = geodetic_2_ECEF(Init_P)
print('Observation Point',OP_xyz.x,OP_xyz.y,OP_xyz.z)
#print('Initial Point    ',P_init_xyz.x,P_init_xyz.y,P_init_xyz.z)
# https://www.fcc.gov/media/radio/dms-decimal
# https://www.gps-coordinates.net/

gmap         = gmplot.GoogleMapPlotter(43.29712668215136,-2.8704744547571863,13)
hidden_gem_lat, hidden_gem_lon = OP.latitud	, OP.longitud
gmap.marker(hidden_gem_lat, hidden_gem_lon)



#---------------------DATA EXTRACTION------------------------------------------
BS_init      = BIT_synchronizer(0, False, False, 0, 0, 0, 0, 0, False, 0, 0 )    
Sat_index    = 0
for sat in sat_list:
    data_process_EN,Data_out_EN  = find_Zcount (Sat_index,BS_init,data_process_EN,Data_out_EN)
    Sat_index                    = Sat_index +1
"""
#----------------------------------------PLOTEADO--------------------------        
plt.figure(2,figsize=(12, 4), dpi=100)
inc       = 0.94/number_sats
start     = (number_sats-1)/number_sats
salto     = -2.2    
Sat_index = 0
for sat in sat_list:           
    plt.step(data_process_EN[Sat_index].bits_time ,  salto*Sat_index+data_process_EN[Sat_index].bits ,linewidth = 0.5, label = 'Sat :'+str(sat)+' '+str("%.1f" % np.mean(Data_out_EN[Sat_index].CN))+'dB '+str("%.1f" % sat_info_EN[Sat_index].elevation)+'º '+str("%.1f" %  sat_info_EN[Sat_index].distance)+ 'm')     #----los bits 
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.9)
    inicio = data_process_EN[Sat_index].SF_s_index.astype(int)
    plt.plot(data_process_EN[Sat_index].bits_time[inicio]  , salto*Sat_index+data_process_EN[Sat_index].bits[inicio],'o',linewidth = 4,color='red')        #-----comienzo SF        
    plt.plot(Data_out_EN[Sat_index].bit_end_time  ,  salto*Sat_index+10*Data_out_EN[Sat_index].bits,linewidth = 0.5)     #----los bits 
    plt.plot(Data_out_EN[Sat_index].bit_end_time  ,  salto*Sat_index+10*data_process_EN[Sat_index].TLM_HOW_AR ,linewidth = 0.2,color='yellow')# TLM&TOW
    plt.plot(Data_out_EN[Sat_index].bit_end_time  ,  salto*Sat_index+10*data_process_EN[Sat_index].preambl_AR,linewidth = 0.5,color='black') # preambles
    plt.plot(Data_out_EN[Sat_index].bit_end_time  ,  salto*Sat_index+10*data_process_EN[Sat_index].clk,linewidth = 1.0)      #----el clk        
    for tindice,indice in enumerate(data_process_EN[Sat_index].TOW_index.astype(int)):
        plt.text(data_process_EN[Sat_index].bits_time[indice],salto*Sat_index,str("%.0f" % data_process_EN[Sat_index].TOW[tindice])+'\n'+str(data_process_EN[Sat_index].bits_time[ data_process_EN[Sat_index].TOW_index[tindice].astype(int)   ] ) ,size=7,bbox=dict(facecolor='red', alpha=0.1))       
    Sat_index          = Sat_index +1 
plt.grid()
plt.show()
"""
#-------nos posicionamos en el primer TOW
TOW_initial    = np.zeros(number_sats)


Sat_index      = 0
for sat in sat_list:
    TOW_initial[Sat_index] = data_process_EN[Sat_index].TOW[0]
    Sat_index              = Sat_index +1
print(TOW_initial)

#--------------REPETIR POR N PUNTOS--------------------------------------------
P_evol         = P_init_xyz
tu_xyz         = 0                      # t offset en el punto inicial
T_increment_s  = 0.020                  # duracion de un bit
N_puntos       = 10
lats_puntos    = np.zeros(N_puntos)
lons_puntos    = np.zeros(N_puntos)
n_SF           = 0                      # Selected SubFramedata_process_ins
H              = np.ones([number_sats,4])
RX_time        = np.zeros(number_sats)
dtime          = 0
for N_iter in np.arange(N_puntos): #-----por cada calculo de PVT---------------
    
    TOW = TOW_initial + N_iter * T_increment_s
    
    #---------------comienzo de calculo coordenadas satelites------------------
    sat_info_EN,OP_xyz = ECCF_sat(path,nombre_f,sat_info_EN,OP,TOW)
    Sat_index          = 0
    for sat in sat_list:                   
        RX_time[Sat_index] = data_process_EN[Sat_index].bits_time[ data_process_EN[Sat_index].TOW_index[0].astype(int)+N_iter]  + sat_info_EN[Sat_index].CLK_correction - 1* sat_info_EN[Sat_index].T_iono
        Sat_index          = Sat_index +1   
    
    #-------------------------CALCULO DE POSICION------------------------------  
    pseudo_r           = c * (RX_time-np.min(RX_time) + dtime)
    #print (pseudo_r)
    #print(TOW[0])
    #print(RX_time[0])
    for indx in np.arange(5):    #inetramos 5 veces por cada punto
        
        dist_xyz_2_sats = distance_from_sats(sat_info_EN,P_evol)    
        pseudo_r_xyz    = dist_xyz_2_sats + c * tu_xyz
        d_p             = pseudo_r_xyz  - pseudo_r
        #print(dist_xyz_2_sats)
        #print('d_p',d_p)
        Sat_index       = 0
        for sat in sat_list:
            H[Sat_index,0]  = (sat_info_EN[Sat_index].x-P_evol.x ) / dist_xyz_2_sats[Sat_index]
            H[Sat_index,1]  = (sat_info_EN[Sat_index].y-P_evol.y ) / dist_xyz_2_sats[Sat_index]
            H[Sat_index,2]  = (sat_info_EN[Sat_index].z-P_evol.z ) / dist_xyz_2_sats[Sat_index]
            #print (np.sqrt ( (sat_XYZ[Sat_index,0]-x_OP)**2 + (sat_XYZ[Sat_index,1]-y_OP)**2 + (sat_XYZ[Sat_index,2]-z_OP)**2))
            #print (Sat_index, sat, d_p [Sat_index])
            #print ('Sat. N ',"%.10f" % (1000*(dist[Sat_index]-min_dist)/c),' ms',RX_time[Sat_index] ,RX_time[Sat_index]- RX_time[min_dist_i])
            #init_psd_ranges = 
            #delta_psd_range[Sat_index] = 
            Sat_index          = Sat_index +1
        
        H_t      = H.transpose()
        d_x      = np.dot ( np.dot(np.linalg.inv(np.dot(H_t,H)),H_t), d_p)
        P_evol.x = P_evol.x + d_x[0]
        P_evol.y = P_evol.y + d_x[1]
        P_evol.z = P_evol.z + d_x[2]
        d_t      = d_x[3]/c
        error    = np.sqrt( (OP_xyz.x-P_evol.x)**2 + (OP_xyz.y-P_evol.y)**2 + (OP_xyz.z-P_evol.z)**2)
        #print (H)
        #print ('X[]              ',d_x[0],d_x[1],d_x[2],d_x[3]/c)
        #print ('Estimated Point  ',P_evol.x,',',P_evol.y,',',P_evol.z)
        #print ('(sg) con respeto a tu en Init_P',d_t)
        #print ('Error (m)',error)
    
    dtime              = dtime + d_t
    #tu_xyz = tu_xyz - d_t 
    print('Observation Point          :','x=',"%.1f" % OP_xyz.x,'y=',"%.1f" % OP_xyz.y,'z=',"%.1f" % OP_xyz.z)
    print('Estimated Point            :','x=',"%.1f" % P_evol.x,'y=',"%.1f" % P_evol.y,'z=',"%.1f" % P_evol.z)
    print('la hora de captura         :',str(hora_cap)) #-----creo que esta mal, deberia ser 10:30:00
    print('Hora del primer TOW        :',str(dt.timedelta(seconds=TOW_initial[0])))
    print('Hora de cálculo            :',str(dt.timedelta(seconds=TOW[0]+np.max(RX_time) + dtime)))
    print('Dist. Error                :',"%.1f" % error,'meters')
    print('Time                       :',"%.3f" % np.float(dtime*1000),'msg')
    print('Time error (from prev. p.) :',"%.3f" % np.float(d_t*1e9),'nsg=',"%.3f" % np.float(d_t*c),'m' )
    
    print('===================================================================')
    
    P_evol_geo                     = ECEF_2_geodetic(P_evol)    
    lats_puntos[N_iter]            = P_evol_geo.latitud     
    lons_puntos[N_iter]            = P_evol_geo.longitud   
    hidden_gem_lat, hidden_gem_lon = P_evol_geo.latitud	, P_evol_geo.longitud
    gmap.marker(hidden_gem_lat, hidden_gem_lon, 'cornflowerblue')

# Draw
gmap.plot   (lats_puntos, lons_puntos, 'cornflowerblue', edge_width=2)
gmap.scatter(lats_puntos, lons_puntos, '#3B0A39', size=.5, marker=False)    
gmap.draw("my_map.html") 
#webbrowser.open_new_tab('my_map.html')
