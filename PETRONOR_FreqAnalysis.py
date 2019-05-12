#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:36:52 2018

@author: alberto
"""

#from pandas import DataFrame
#import numpy as np
#from scipy import stats
#import matplotlib.pyplot as plt
#import matplotlib.font_manager as font_manager
import pandas as pd
#from statsmodels.tsa.ar_model import AR
#from sklearn.metrics import mean_squared_error
#from sklearn.model_selection import TimeSeriesSplit
#from statsmodels.tsa.arima_model import ARIMA
import datetime
#from PETRONOR_FreqAnalysis_lyb import *
from PETRONOR_temporalAnalysis import *

Path_out  = 'C:\\OPG106300\\TRABAJO\\Proyectos\\Petronor-075879.1 T 20000\\Trabajo\\python\\outputs\\'


df_SPEED  = pd.read_pickle(Path_out+'SPEED_f_H4-FA-0002_SH3_10__.pkl')
df_speed  = pd.read_pickle(Path_out+'speed_t_H4-FA-0002_SH3_10__.pkl')
df_Frq_FP = pd.read_pickle(Path_out+'Freq_FP_H4-FA-0002_SH3_10__.pkl')

indice = []
for counter,k in enumerate(df_SPEED.index):
    #print(datetime.datetime.fromtimestamp(int(k)))
    indice.append(datetime.datetime.fromtimestamp(int(k)))

df_SPEED.index  = pd.to_datetime(indice)
df_speed.index  = pd.to_datetime(indice)
df_Frq_FP.index = pd.to_datetime(indice)


#p1 = temporal_analysis()
#o1 = p1.get_temporal_feats(df_speed)
#p1.draw_boxplots(o1, frequency='D')

p2 = temporal_analysis()
o2 = p2.get_frequency_feats(df_speed,df_Frq_FP)
p2.draw_boxplots(o2, frequency='D')