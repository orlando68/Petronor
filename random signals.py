# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:01:15 2019

@author: 106300
"""
import numpy as np
import matplotlib.pyplot as plt
from PETRONOR_lyb import *
from scipy.signal import find_peaks

fs     = 5120
l      = 16384
l_2    = np.int(l/2)
t      = np.arange(l)/fs
f      = np.arange(l)/(l-1)*fs
signal = np.load('captura.npy')
#signal = 10*np.sin(2*np.pi*25*t)


S     = np.fft.fft(signal/l)
S_bis = S
S_abs = np.abs(S)
s     = l*np.fft.ifft(S)
s_r   = np.real(s)
s_i   = np.imag(s)

plt.figure()
plt.plot(t,signal,t,s_r,t,s_i)
plt.show()

index,properties = find_peaks(S_abs[0:l_2],height  = 0.2 ,prominence = 0.03 , width=1 , rel_height = 0.75)
#S_rev            = S_abs[::-1]


picos =np.array([4.5,6.5,4])

for i in range(np.size(index)):
    inic = np.round(properties["left_ips"]).astype(int)[i]
    fin = np.round(properties["right_ips"]).astype(int)[i]
    power = np.sqrt(np.sum(S_abs[inic:fin]**2))
    print(power,'mm/s')
    print(S_abs[inic:fin])
    print(S_abs[l-fin+1:l-inic+1])
    factor                  = picos[i]/power #* np.exp(np.pi/2*1j)
    S_bis[inic:fin]         = factor          * S_bis[inic:fin]
    S_bis[l-fin+1:l-inic+1] = np.conj(factor) * S_bis[l-fin+1:l-inic+1]
    power = np.sqrt(np.sum( np.abs(  S_bis[inic:fin] ) **2))
    print(power,'mm/s')
    print('---------------')
    


f_end = 300
n_end = int(f_end*l/fs)
plt.figure()
plt.plot(f[0:n_end],S_abs[0:n_end])
plt.plot(f[0:n_end],np.abs(S_bis[0:n_end]))
plt.plot(f[index],S_abs[index],'o')
plt.vlines(x=f[index], ymin=S_abs[index] - properties["prominences"],ymax = S_abs[index], color = "C1")
plt.hlines(y=properties["width_heights"], xmin=f[np.round(properties["left_ips"]).astype(int)],xmax=f[np.round(properties["right_ips"]).astype(int)], color = "C1")
plt.grid(True)
plt.show()
s_bis = l* np.fft.ifft(S_bis)


plt.figure()
plt.title('Imag parts')
plt.plot(t,s_i,t,np.imag(s_bis))
plt.show()

plt.figure()
plt.title('Real parts')
plt.plot(t,signal,t,np.real(s_bis))
plt.grid(True)
plt.show()


#
#plt.figure()
#plt.plot(t,s_i,t,np.imag(s_bis))
#plt.show()