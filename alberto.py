#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:36:52 2018

@author: alberto
"""

from pandas import DataFrame
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#import matplotlib.font_manager as font_manager
import pandas as pd
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima_model import ARIMA
import datetime

"""
file_name = 'tfeats_05to12oct2018.csv'
#file_name = 'all_data.csv'
df=(pd.read_csv(file_name))
df.index=df['Unnamed: 0'].values 
df= df.drop(labels= 'Unnamed: 0',axis=1)
df.index = pd.to_datetime(df.index)
""" 

df_SPEED = pd.read_pickle('SPEED_H4-FA-0002_SH4_10__.pkl')


indice = []
for counter,k in enumerate(df_SPEED.index):
    #print(datetime.datetime.fromtimestamp(int(k)))
    indice.append(datetime.datetime.fromtimestamp(int(k)))

df_SPEED.index = pd.to_datetime(indice)

# ta = temporal_analysis();orl = ta.get_temporal_feats(df);ta.draw_boxplots(orl, frequency='D')
# ta2 = temporal_analysis();orl2 = ta2.get_temporal_feats(df_SPEED);ta2.draw_boxplots(orl2, frequency='D')