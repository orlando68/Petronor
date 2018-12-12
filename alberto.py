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

file_name = 'tfeats_05to12oct2018.csv'
file_name = 'all_data.csv'
df=(pd.read_csv(file_name))

df.index=df['Unnamed: 0'].values 
df= df.drop(labels= 'Unnamed: 0',axis=1)

df.index = pd.to_datetime(df.index)

