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




class temporal_analysis(object):
    
    
    '''
    Mehtod that computes the root mean square of an array.
        - Inputs: X
        - Outputs: rms of X
    '''
    def rms(self,
            X):
        # remove nans
        X.dropna(inplace=True)
        return np.sqrt(sum(n*n for n in X)/len(X))
    
    
    '''
    Mehtod that computes the peak value of an array.
        - Inputs: X
        - Outputs: peak value of X
    '''
    def peak_value(self,
                   X):
        # remove nans
        X.dropna(inplace=True)
        return (max(X)-min(X))/2
    
    
    '''
    Mehtod that computes the crest factor of an array.
        - Inputs: X
        - Outputs: crest factor of X
    '''
    def crest_factor(self,
                     X):
        # remove nans
        X.dropna(inplace=True)
        return self.peak_value(X)/self.rms(X)
    
    
    '''
    Mehtod that computes the kurtosis of an array.
        - Inputs: X
        - Outputs: kurtosis of X
    '''
    def kurtosis(self,
                 X):
        # remove nans
        X.dropna(inplace=True)
        return stats.kurtosis(X)
    
    '''
    Mehtod that computes the skewness of an array.
        - Inputs: X
        - Outputs: skewness of X
    '''
    def skewness(self,
                 X):
        # remove nans
        X.dropna(inplace=True)
        return stats.skew(X) 
    

    '''
    Mehtod that computes the clearance factor of an array.
        - Inputs: X
        - Outputs: clearance factor of X
    '''    
    def clearance_factor(self,
                         X):
        # remove nans
        X.dropna(inplace=True)
        return self.peak_value(X)/(sum(np.sqrt(abs(n)) for n in X)/len(X))
    
    
    '''
    Mehtod that computes the impulse factor of an array.
        - Inputs: X
        - Outputs: impulse factor of X
    '''
    def impulse_factor(self,
                       X):
        # remove nans
        X.dropna(inplace=True)
        return self.peak_value(X)/(sum(abs(n) for n in X)/len(X))
    
    
    '''
    Mehtod that computes the shape factor of an array.
        - Inputs: X
        - Outputs: shape factor of X
    '''
    def shape_factor(self,
                     X):
        # remove nans
        X.dropna(inplace=True)
        return self.rms(X)/(sum(abs(n) for n in X)/len(X))    
    
    
    '''
    Mehtod that computes the Shannon entropy of an array.
        - Inputs: X
        - Outputs: Shannon entropy of X
    '''
    def entropy(self,
                X):
        # remove nans
        X.dropna(inplace=True)
        pX = X / X.sum()
        return -np.nansum(pX*np.log2(pX))
    
    
    '''
    Mehtod that computes the Weibull negative log-likelihood of an array.
        - Inputs: X
        - Outputs: Wnl of X
    '''
    def Wnl(self,
            X):
        # remove nans
        X.dropna(inplace=True)
        params = stats.exponweib.fit(X, floc=0, f0=1)
        shape = params[1]
        scale = params[3] 
        weibull_pdf = (shape / scale) * (X / scale)**(shape-1) * np.exp(-(X/scale)**shape)
        return -np.nansum(np.log(weibull_pdf))
    
    
    '''
    Mehtod that computes the normal negative log-likelihood of an array.
        - Inputs: X
        - Outputs: Nnl of X
    '''
    def Nnl(self, 
            X):
        # remove nans
        X.dropna(inplace=True)
        mean = np.mean(X)
        std = np.std(X)
        normal_pdf = np.exp(-(X-mean)**2/(2*std**2)) / (std*np.sqrt(2*np.pi))
        return -np.nansum(np.log(normal_pdf))
    
    
    '''
    Mehtod that extracts temporal features from a set of data.
        - Inputs: data (DataFrame that contains data to be processed)
        - Outputs: DataFrame, indexed by time and containing computed features
    '''
    def get_temporal_feats(self,
                           data):
        df = DataFrame({"RMS": data.apply(self.rms, axis=1).values,
                        "PEAK_VALUE": data.apply(self.peak_value, axis=1).values,
                        "CREST_FACTOR": data.apply(self.crest_factor, axis=1).values,
                        "KURTOSIS": data.apply(self.kurtosis, axis=1).values,
                        "SKEWNESS": data.apply(self.skewness, axis=1).values,
                        "CLEARANCE_FACTOR": data.apply(self.clearance_factor, axis=1).values,
                        "IMPULSE_FACTOR": data.apply(self.impulse_factor, axis=1).values,
                        "SHAPE_FACTOR": data.apply(self.shape_factor, axis=1).values,
                        "AVG": data.apply(np.nanmean, axis=1).values,
                        "STDV": data.apply(np.nanstd, axis=1).values,
                        "MEDIAN": data.apply(np.nanmedian, axis=1).values,
                        "MIN": data.apply(np.nanmin, axis=1).values,
                        "MAX": data.apply(np.nanmax, axis=1).values,
                        "VARIANCE": data.apply(np.nanvar, axis=1).values,
                        "ENTROPY": data.apply(self.entropy, axis=1).values,
                        "WNL": data.apply(self.Wnl, axis=1).values,
                        "NNL": data.apply(self.Nnl, axis=1).values})
        df.index = data.index
        return df
            
    
    '''
    Mehtod that plots given temporal features.
        - Inputs: tfeats (DataFrame that contains temporal feature values)
        - Outputs: 
    '''    
    def plot_tfeats(self,
                    tfeats):
        for feat in tfeats.columns:
            plt.figure()
            plt.plot(tfeats[feat])
            plt.title(feat)
            plt.yticks()
            plt.xticks(rotation=45, rotation_mode='anchor', ha="right")
            plt.grid()
        tfeats.replace([np.inf, -np.inf], np.nan, inplace=True)
        tfeats.hist()
        
    
    '''
    Mehtod that draws the boxplots for a set of temporal feats, by week, and removes outliers.
        - Inputs: df (DataFrame that contains temporal feature values), frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''   
    def draw_boxplots(self,
                      df,
                      frequency='W'):
        df_out = []        
        feats_out = []
        for feat in df.columns:
            plt.figure()   
            plt.title(feat)
            plt.grid()     
            weeks = []
            cols = []
            weeks_out = []
            groups = df[feat].groupby(pd.Grouper(freq=frequency))
            for date, group in groups:                    
                cols.append(date.date())
                data = DataFrame(group.values)
                weeks.append(data)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                weeks_out.append(data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)])                
            weeks = pd.concat(weeks, axis=1)                
            weeks.columns=cols
            weeks.boxplot(vert=True, rot=45)
            weeks_out = pd.concat(weeks_out, axis=1)
            weeks_out.columns=cols
            df_out.append(weeks_out)
            feats_out.append(feat)
        return df_out, feats_out

    
    '''
    Mehtod that draw the boxplots for a set of temporal feats, by week.
        - Inputs: df (DataFrame that contains temporal feature values), 
                  frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''   
    def ar_model(self,
                 df,
                 frequency='W'):
        for feat in df.columns:
            residuals = []
            limits = [0]
            tmp = 0
            print(feat)
            groups = df[feat].groupby(pd.Grouper(freq=frequency))
            gp = groups.apply(lambda x: x.name)
            n = len(groups.apply(lambda x: x.name))
            c = 0
            for date, group in groups:
                print(gp[c])
                data = group.values
                data = data[~np.isnan(data)]
                data = data[np.isfinite(data)]
                c += 1
                if c >= n:
                    print('done!')
                    break
                data_sig = groups.get_group(gp[c])
                data_sig = data_sig[~np.isnan(data_sig)]
                data_sig = data_sig[np.isfinite(data_sig)]
                # train AR model after removing outliers
                data = DataFrame(data)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                train_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)].values
                #model = ARIMA(data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)].values, order=(len(data_sig)-1,1,0))
                #model_fit = model.fit(disp=0)
                #print(model_fit.summary())
                model = AR(train_data)
                model_fit = model.fit()                
                print('Lag: %s' % model_fit.k_ar)
                print('Coefficients: %s' % model_fit.params)
                # make predictions
                predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(data_sig)-1, dynamic=True)
                #residuals = DataFrame(model_fit.resid)
                error = mean_squared_error(data_sig, predictions)
                print(gp[c])
                print('Test MSE: %.3f \n' % error)
                residuals = np.concatenate((residuals, abs(data_sig - predictions)), axis=None)
                limits.append(tmp + len(data))
                tmp += len(data)
            # plot residuals        
            plt.figure()
            plt.title(feat)
            plt.plot(range(limits[1], limits[1]+len(residuals)), residuals, label='residuals')
            plt.xlim(0, limits[1]+len(residuals))
            plt.legend()
            plt.xticks(limits, [date.date() for date in gp], rotation=45) 
            plt.grid()        
            

    '''
    Mehtod that draws a lag plot (t vs t+1 values) for each temporal feature.
        - Inputs: data (DataFrames that contains temporal feature values)
        - Outputs: 
    ''' 
    def draw_lagplots(self,
                      data):
        for feat in data.columns: 
            print(feat)
            df = pd.concat([data[feat].shift(1), data[feat]], axis=1)
            df.columns = ['t-1', 't+1']
            result = df.corr()
            print(str(result)+'\n')
            plt.figure()   
            plt.title(feat)
            plt.grid()  
            pd.plotting.lag_plot(data[feat])                
            a = np.min(data[feat][np.where(data[feat]!=-np.inf)[0]])
            b = np.max(data[feat][np.where(data[feat]!=np.inf)[0]])
            plt.plot([a, b], [a, b], c='r')                    
            
            
    '''
    Mehtod that loads temporal feats given the paths
        - Inputs: path, file_names
        - Outputs: 
    '''  
    def load_temporal_feats(self,
                            path='/media/alberto/9AE6A9E8E6A9C53B/Users/104166/Data/PETRONOR/data/',
                            file_names=['tfeats_june2oct2018.csv',
                                        'tfeats_15to22oct2018.csv',
                                        'tfeats_22to29oct2018.csv',
                                        'tfeats_29oct5nov2018.csv',
                                        'tfeats_05to12oct2018.csv',
                                        'tfeats_12to19oct2018.csv']):
        dfs = []    
        for file_name in file_names:    
            dfs.append(pd.read_csv(path+file_name))
        data = pd.concat(dfs, sort=True)
        data.index = data['Unnamed: 0'].values
        data.index = pd.to_datetime(data.index)
        data = data.drop(labels='Unnamed: 0',axis=1)
        return data
        