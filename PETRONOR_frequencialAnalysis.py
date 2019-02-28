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
import matplotlib.dates as mdates
#import matplotlib.font_manager as font_manager
import pandas as pd
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.vector_ar.var_model import VAR
import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cross_validation import train_test_split 
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import datetime
from sklearn.model_selection import KFold




class temporal_analysis(object):
    
    
    '''
    Mehtod that computes the root mean square of an array.
        - Inputs: X
        - Outputs: rms of X
    '''
    def central_frequency(self,X):
        fs = 5120.0
        l = 16384.0
        f   = np.arange(l)/l*fs
        #print (np.argmax(X)/l*fs)
        return np.sum(f*f*X)/np.sum(X)

    '''
    Mehtod that computes the Root Mean Square Frequency of an array.
        - Inputs: X
        - Outputs: RMSF of X
    '''
    def RMSF(self,X):
        fs = 5120.0
        l = 16384.0
        f   = np.arange(l)/l*fs
        #print (np.argmax(X)/l*fs)
        return np.sqrt(np.sum(f*f*X)/np.sum(X))    
    

    '''
    Mehtod that computes Root Variance Frequency.
        - Inputs: X
        - Outputs: RVF of X
    '''    
    def RVF(self, X):

        l = 16384.0
        X_prima = np.diff(X)
        FC      = np.sum(X_prima*X[0:int(l)-1]) /(2*np.pi*np.sum(X*X))
        MSF     = np.sum(X_prima**2)     /(4*np.pi**2*np.sum(X*X))
       # RMSF    = np.sqrt(MSF)
        return np.sqrt(MSF-FC**2)
        
    
    
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
    Mehtod that computes the Shannon entropy of an array.
        - Inputs: X
        - Outputs: Shannon entropy of X
    '''
    def spectral_entropy(self,
                X):
        # remove nans
        X.dropna(inplace=True)
        pX = X / X.sum()
        return -np.nansum(pX*np.log2(pX)) / np.size(X)
    
    

    

    
    
    '''
    Mehtod that extracts temporal features from a set of data.
        - Inputs: data (DataFrame that contains data to be processed)
        - Outputs: DataFrame, indexed by time and containing computed features
    '''
    def get_temporal_feats(self,
                           data):
        df = DataFrame({"CENTRAL_FREQUENCY": data.apply(self.central_frequency, axis=1).values,
                        "ROOT MEAN SQUARE FREQUENCY": data.apply(self.RMSF, axis=1).values,
                        "ROOT VARIANCE FREQUENCY": data.apply(self.RVF, axis=1).values,
                        
                        "KURTOSIS": data.apply(self.kurtosis, axis=1).values,
                        "SKEWNESS": data.apply(self.skewness, axis=1).values,
                        #"CLEARANCE_FACTOR": data.apply(self.clearance_factor, axis=1).values,
                        #"IMPULSE_FACTOR": data.apply(self.impulse_factor, axis=1).values,
                        #"SHAPE_FACTOR": data.apply(self.shape_factor, axis=1).values,
                        #"AVG": data.apply(np.nanmean, axis=1).values,
                        "STDV": data.apply(np.nanstd, axis=1).values,
                        #"MEDIAN": data.apply(np.nanmedian, axis=1).values,
                        #"MIN": data.apply(np.nanmin, axis=1).values,
                        #"MAX": data.apply(np.nanmax, axis=1).values,
                        "VARIANCE": data.apply(np.nanvar, axis=1).values,
                        "SPECTRAL ENTROPY": data.apply(self.spectral_entropy, axis=1).values,
                        #"WNL": data.apply(self.Wnl, axis=1).values,
                        #"NNL": data.apply(self.Nnl, axis=1).values
                        })
        df.index = data.index
        return df
            
    
    '''
    Mehtod that plots given features.
        - Inputs: tfeats (DataFrame that contains feature values), frequency (date frequency for xticks)
        - Outputs: 
    '''    
    def plot_tfeats(self,
                    tfeats,
                    frequency='W'):
        if frequency == 'D':
            n = 1
        elif frequency == 'W':
            n = 7
        else:
            n = 30
        for feat in tfeats.columns:
            plt.figure()  
            plt.plot(tfeats[feat], lw=3)
            plt.title(feat,)
            ax = plt.gca()
            # set monthly locator
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=n))
            # set formatter
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
            # set font and rotation for date tick labels
            plt.gcf().autofmt_xdate()
            #plt.xticks([idx.strftime('%a\n%d\n%h\n%Y') for idx in tfeats.index[::n]], rotation=45, rotation_mode='anchor', ha="right")
            plt.grid()
            
            # histogram
            plt.figure()
            tfeats[feat].replace([np.inf, -np.inf], np.nan, inplace=True)
            tfeats[feat].astype(float).hist(figsize=(14,6))
            plt.title(feat)
        
    
    '''
    Mehtod that draws the boxplots for a set of feats, by week, and removes outliers.
        - Inputs: df (DataFrame that contains feature values), frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''   
    def draw_boxplots(self,
                      df,
                      frequency='W'):
        # boxplot config
        boxprops = dict(linestyle='-', linewidth=4, color='k')
        medianprops = dict(linestyle='-', linewidth=4, color='k')
        df_out = []
        feats_out = []
        for feat in df.columns:
            plt.figure()   
            plt.title(feat)
            lag = []
            cols = []
            data_out = []
            groups = df[feat].groupby(pd.Grouper(freq=frequency))
            for date, group in groups:                    
                cols.append(date)
                data = DataFrame(group.values)
                lag.append(data)
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                data_out.append(data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)])                
            lag = pd.concat(lag, axis=1)                
            lag.columns=cols
            lag.boxplot(vert=True, rot=45, showmeans=True, boxprops=boxprops, medianprops=medianprops)
            
            data_out = pd.concat(data_out, axis=1)
            data_out.columns=cols
            df_out.append(data_out)
            feats_out.append(feat)
        return df_out, feats_out


    '''
    Mehtod that computes the moving avg model for a set of feats.
        - Inputs: df (DataFrame that contains feature values), 
                  frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''   
    def moving_avg_model(self,
                         df,
                         frequency='W'):
        # last date, given as prediction
        add_time = 0
        if frequency == 'W':
            add_time = 7
        elif frequency == 'D':
            add_time = 1
        anomalies = []
        for feat in df.columns:
            residuals = []
            threshold = np.inf
            dates = []
            limits = [0]
            tmp = 0
            # print(feat)
            groups = df[feat].groupby(pd.Grouper(freq=frequency))
            gp = groups.apply(lambda x: x.name)
            anomalies_tmp = [0]*(len(gp))
            n = len(groups.apply(lambda x: x.name))
            c = 0
            filtered_data = []
            for date, group in groups:
                # print(gp[c])
                dates.append(gp[c])
                data = group.values
                data = data[~np.isnan(data)]
                data = data[np.isfinite(data)]
                data = DataFrame(data)
                if not data.shape[0] > 1:
                    c += 1
                    residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                    anomalies_tmp[c-1] = 0
                    limits.append(tmp + len(filtered_data))
                    tmp += len(filtered_data)
                    continue                
                # train AR model after removing outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)].values
                if c == 0:
                    limits.append(len(filtered_data))
                    tmp += len(filtered_data)    
                # make predictions
                c += 1
                if c < n:
                    try:
                        data_sig = groups.get_group(gp[c])
                    except Exception as ex:
                        #print('No data found for ' + str(gp[c]))
                        residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                        anomalies_tmp[c-1] = 0
                        limits.append(tmp + len(filtered_data))
                        tmp += len(filtered_data)
                        continue
                    #data_sig = data_sig[~np.isnan(data_sig)]
                    #data_sig = data_sig[np+.isfinite(data_sig)]
                else:
                    data_sig = filtered_data.ravel()
                limits.append(tmp + len(data_sig))
                tmp += len(data_sig)
                # moving average model
                moving_avg = np.mean(filtered_data)
                residuals_tmp = []
                for value in data_sig:
                    residuals_tmp.append(abs(moving_avg - value))
                    moving_avg = (moving_avg + value) / 2
                residuals = np.concatenate((residuals, residuals_tmp), axis=None)
                # check anomalies, with severity
                anomalies_tmp[c-1] = np.round(len(np.where(np.array(residuals_tmp)>threshold)[0])/len(residuals_tmp), decimals=3)
                threshold = np.mean(residuals_tmp)
            # add days to last prediction residuals
            if anomalies == []:
                anomalies = np.array(anomalies_tmp)
            else:
                anomalies = np.column_stack((anomalies, np.array(anomalies_tmp)))
            days = datetime.timedelta(days=add_time)
            dates.append(dates[-1]+days)
            # plot residuals
            residuals = self.normalize(residuals)
            plt.figure(figsize=(64,24))
            plt.title(feat, size=48)
            plt.plot(range(limits[1], limits[1]+len(residuals)), residuals, label='residuals', lw=3)
            plt.xlim(0, limits[1]+len(residuals))
            plt.legend(fontsize=48)
            plt.xticks(limits[::7], [date.date() for date in dates[::7]], rotation=45, ha="right", size=48) 
            plt.yticks(size=48)
            plt.grid()
        # print anomalies
        res = DataFrame(data=anomalies, index=dates[1:], columns=df.columns)
        return res


    ''''''''''''
    ''''TESTING
    '''''''''''
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1, look_forward=0, output_feat=0):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-look_forward-1):
            if output_feat > 1:
                a = dataset[i:(i+look_back), :-1]
                dataY.append(dataset[(i+look_back):(i+look_back+look_forward), output_feat]) 
            else:
                a = dataset[i:(i+look_back), 0]
                dataY.append(dataset[(i+look_back):(i+look_back+look_forward), 0]) 
            dataX.append(a)        
        return np.array(dataX), np.array(dataY)


    def mlp_model(self,
                  df,
                  target,
                  look_back = 30,
                  look_forward = 10):    
        from sklearn.preprocessing import MinMaxScaler
        from keras.models import Sequential
        from keras.layers import Dense
        # fix random seed for reproducibility
        np.random.seed(7)     
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_scaled = DataFrame(data=scaler.fit_transform(df.values), columns=df.columns, index=df.index)
        target_scaled = DataFrame(data=scaler.fit_transform(np.expand_dims(target.values.ravel(), axis=2)), index=target.index)        
        # create model
        mlp = Sequential()
        mlp.add(Dense(50, input_dim=df_scaled.shape[1], activation='sigmoid'))
        mlp.add(Dense(10, activation='sigmoid'))
        mlp.add(Dense(1))
        mlp.compile(loss='mae', optimizer='rmsprop', metrics=['mean_squared_logarithmic_error', 'mse', 'mae', 'mape','logcosh'])  
        # validation indexes
        traintestX, validationX, traintestY, validationY = train_test_split(df_scaled, target_scaled, test_size=0.2, random_state=0) 
        # traintest
        tscv_traintest = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in tscv_traintest.split(traintestY):
            print("TRAIN:", str(train_index[0]) + ' to ' + str(train_index[len(train_index)-1]), "TEST:", str(test_index[0]) + ' to ' + str(test_index[len(test_index)-1]))
            trainX, testX = traintestX.iloc[train_index], traintestX.iloc[test_index]
            trainY, testY = traintestY.iloc[train_index], traintestY.iloc[test_index]
            mlp.fit(trainX, trainY, epochs=100, batch_size=30, verbose=0)
            predictions_train = mlp.predict(trainX)
            predictions_test = mlp.predict(testX)
            # invert predictions
            predictions_train_trans = scaler.inverse_transform(predictions_train)
            predictions_test_trans = scaler.inverse_transform(predictions_test)
            trainY_trans = scaler.inverse_transform(trainY)
            testY_trans = scaler.inverse_transform(testY)
            # Final evaluation of the model
            trainScore = metrics.mean_absolute_error(trainY_trans, predictions_train_trans)
            print('Train Score: %.3f MAE' % (trainScore))
            testScore = metrics.mean_absolute_error(testY_trans, predictions_test_trans)
            print('Test Score: %.3f MAE' % (testScore))
            trainScore = metrics.mean_squared_error(trainY_trans, predictions_train_trans)
            print('Train Score: %.3f MSE' % (trainScore))
            testScore = metrics.mean_squared_error(testY_trans, predictions_test_trans)
            print('Test Score: %.3f MSE' % (testScore))
            trainScore = np.sqrt(((trainY_trans - predictions_train_trans) ** 2).mean())
            print('Train Score: %.3f RMSE' % (trainScore))
            testScore = np.sqrt(((testY_trans - predictions_test_trans) ** 2).mean())
            print('Test Score: %.3f RMSE' % (testScore))
        # validation
        mlp.fit(traintestX, traintestY, epochs=100, batch_size=30, verbose=0)
        res = mlp.predict(validationX)
        res_trans = scaler.inverse_transform(res)
        validationY_trans = scaler.inverse_transform(validationY) 
        # Final evaluation of the model
        validationScore = metrics.mean_absolute_error(validationY_trans, res_trans)
        print('Validation Score: %.3f MAE' % (validationScore))        
        validationScore = metrics.mean_squared_error(validationY_trans, res_trans)
        print('Validation Score: %.3f MSE' % (validationScore))        
        validationScore = np.sqrt(((validationY - res_trans) ** 2).mean())
        print('Train Score: %.3f RMSE' % (validationScore))
        return mlp, res_trans, validationY_trans
    
    
    def rf_model(self,
                 df,
                 target):    
        from sklearn.ensemble import RandomForestRegressor                
        # fix random seed for reproducibility
        np.random.seed(7) 
        # create model
        n_estimators = 30
        forest = RandomForestRegressor(n_estimators=n_estimators, random_state=1)
        # validation indexes
        traintestX, validationX, traintestY, validationY = train_test_split(df, target, test_size=0.2, random_state=0) 
        # traintest
        tscv_traintest = TimeSeriesSplit(n_splits=10)
        for train_index, test_index in tscv_traintest.split(traintestY):
            print("TRAIN:", str(train_index[0]) + ' to ' + str(train_index[len(train_index)-1]), "TEST:", str(test_index[0]) + ' to ' + str(test_index[len(test_index)-1]))
            trainX, testX = traintestX.iloc[train_index], traintestX.iloc[test_index]
            trainY, testY = traintestY.iloc[train_index], traintestY.iloc[test_index]
            forest.fit(trainX, trainY)
            predictions_train = forest.predict(trainX)
            predictions_test = forest.predict(testX)                        
            # Final evaluation of the model
            trainScore = metrics.mean_absolute_error(trainY, predictions_train)
            print('Train Score: %.3f MAE' % (trainScore))
            testScore = metrics.mean_absolute_error(testY, predictions_test)
            print('Test Score: %.3f MAE' % (testScore))
            trainScore = metrics.mean_squared_error(trainY, predictions_train)
            print('Train Score: %.3f MSE' % (trainScore))
            testScore = metrics.mean_squared_error(testY, predictions_test)
            print('Test Score: %.3f MSE' % (testScore))
            trainScore = np.sqrt(((trainY - predictions_train) ** 2).mean())
            print('Train Score: %.3f RMSE' % (trainScore))
            testScore = np.sqrt(((testY - predictions_test) ** 2).mean())
            print('Test Score: %.3f RMSE' % (testScore))  
        # validation
        forest.fit(traintestX, traintestY)
        res = forest.predict(validationX)
        # Final evaluation of the model        
        validationScore = metrics.mean_absolute_error(validationY, res)
        print('Validation Score: %.3f MAE' % (validationScore))        
        validationScore = metrics.mean_squared_error(validationY, res)
        print('Validation Score: %.3f MSE' % (validationScore))        
        validationScore = np.sqrt(((validationY - res) ** 2).mean())
        print('Train Score: %.3f RMSE' % (validationScore))       
        return forest, res, validationY
    
    
    def feature_ranking(self, 
                        X, 
                        y):
        '''
        Method to compute the feature importance based on their score when decreasing impurity in random forests model.
            Inputs:
                - X, y: set of features ([[X]]), target values ([y])
            Outputs: none
        '''
        ##################################
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cross_validation import train_test_split    
        from sklearn.model_selection import cross_val_score
        ##################################                       
        # set the number of estimators        
        n_estimators = 30
        # 10-fold CV
        forest = RandomForestRegressor(n_estimators=n_estimators, random_state=1)
#        kf = KFold(n_splits=10, random_state=1)
#        for train_index, test_index in kf.split(target_scaled):
#            print("TRAIN:", train_index, "TEST:", test_index)
#            trainX, testX = df_scaled.iloc[train_index], df_scaled.iloc[test_index]
#            trainY, testY = target_scaled.iloc[train_index], target_scaled.iloc[test_index]
        forest.fit(X, y)    
        score = forest.score(X, y)
        print("Score: " + str(score))
        # Build a forest and compute the feature importances    
        forest.fit(X, y)    
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        # normalize values between 0 and 1
        importances = self.scale_linear_bycolumn(importances)
        # Print the feature ranking
        print("Feature ranking:")
        for f in range(X.shape[1]):
            print("%d. %s (%f)" % (f+1, X.columns[indices[f]], importances[indices[f]]))
        # Plot the feature importances of the forest
        plt.figure(figsize=(30,8))
        plt.bar(np.arange(0,X.shape[1]),importances[indices],color="r", yerr=std[indices], align="center")
        #    plt.xticks(range(X.shape[1]), [str(i) for i in X.columns[indices]],size=20,rotation=90)
        plt.xticks(range(X.shape[1]), [str(i) for i in X.columns[indices]], rotation=45, ha="right", rotation_mode='anchor', size=24)
        plt.yticks(size=24)
        plt.ylabel('score', size=24)
        plt.xlim([-1, X.shape[1]])
        #plt.tight_layout()
        plt.grid()
        plt.show()
        return X.iloc[:,np.where(importances>0.0)[0]]
    
    
    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = np.concatenate(cols, axis=1)
        agg = DataFrame(agg, columns = names)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
    
    def scale_linear_bycolumn(self,
                              rawpoints, 
                              high=1.0,
                              low=0.0):
        '''
        Method that linearly scales a data set by columns.
            Inputs: 
                - rawpoints: a data set (matrix [m,n])
                - high: the maximum value
                - low: the minimum value 
            Output: scaled data set
        '''
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng)
    

    '''
    Mehtod that computes the AR model for a set of feats.
        - Inputs: df (DataFrame that contains feature values), 
                  frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''   
    def ar_model(self,
                 df,
                 frequency='W'):    
        # last date, given as prediction
        add_time = 0
        if frequency == 'W':
            add_time = 7
        elif frequency == 'D':
            add_time = 1
        anomalies = []
        for feat in df.columns:
            residuals = []
            threshold = np.inf
            dates = []
            limits = [0]
            tmp = 0
            # print(feat)
            groups = df[feat].groupby(pd.Grouper(freq=frequency))
            gp = groups.apply(lambda x: x.name)
            anomalies_tmp = [0]*(len(gp))
            n = len(groups.apply(lambda x: x.name))
            c = 0
            filtered_data = []
            for date, group in groups:
                # print(gp[c])
                dates.append(gp[c])
                data = group.values
                data = data[~np.isnan(data)]
                data = data[np.isfinite(data)]
                data = DataFrame(data)
                if not data.shape[0] > 1:
                    c += 1
                    residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                    anomalies_tmp[c-1] = 0
                    limits.append(tmp + len(filtered_data))
                    tmp += len(filtered_data)
                    continue                
                # train AR model after removing outliers
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)].values
                model = AR(filtered_data)
                model_fit = model.fit()
                #print(model_fit.summary())
                #print('Lag: %s' % model_fit.k_ar)
                #print('Coefficients: %s' % model_fit.params)
                if c == 0:
                    limits.append(len(filtered_data))
                    tmp += len(filtered_data)    
                # make predictions
                c += 1
                if c < n:
                    try:
                        data_sig = groups.get_group(gp[c])
                    except Exception as ex:
                        print('No data found for ' + str(gp[c]))
                        residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                        anomalies_tmp[c-1] = 0
                        limits.append(tmp + len(filtered_data))
                        tmp += len(filtered_data)
                        continue
                    #data_sig = data_sig[~np.isnan(data_sig)]
                    #data_sig = data_sig[np+.isfinite(data_sig)]
                    if model_fit.k_ar == len(model_fit.params):
                        print('Constant values used for training at ' + str(gp[c]))
                        residuals = np.concatenate((residuals, [0]*len(data_sig)), axis=None)
                        anomalies_tmp[c-1] = 0
                        limits.append(tmp + len(data_sig))
                        tmp += len(data_sig)
                        continue
                else:
                    data_sig = filtered_data.ravel()
                limits.append(tmp + len(data_sig))
                tmp += len(data_sig)
                ## ARIMA
                #model = ARIMA(filtered_data, order=(len(data_sig)-1,1,0))
                #model_fit = model.fit(disp=False)
                #arma_mod30 = sm.tsa.ARMA(filtered_data, (3,0)).fit(disp=False)
                #arma_mod30.predict(start=len(filtered_data), end=len(filtered_data)+len(data_sig)-1, dynamic=True)
                #print(arma_mod30.resid)
                try:
                    predictions = model_fit.predict(start=len(filtered_data), end=len(filtered_data)+len(data_sig)-1, dynamic=True)
                except Exception as ex:
                    residuals = np.concatenate((residuals, [0]*len(data_sig)), axis=None)
                    anomalies_tmp[c-1] = 0
                    continue
                #residuals = DataFrame(model_fit.resid)
                #error = mean_squared_error(data_sig, predictions)
                #print(gp[c])
                #print('Test MSE: %.3f \n' % error)
                residuals = np.concatenate((residuals, abs(predictions-np.nan_to_num(data_sig))), axis=None)
                # check anomalies, with severity
                anomalies_tmp[c-1] = np.round(len(np.where(abs(predictions-np.nan_to_num(data_sig))>threshold)[0])/len(predictions), decimals=3)
                threshold = np.mean(abs(predictions-np.nan_to_num(data_sig)))
            # add days to last prediction residuals
            if anomalies == []:
                anomalies = np.array(anomalies_tmp)
            else:
                anomalies = np.column_stack((anomalies, np.array(anomalies_tmp)))
            days = datetime.timedelta(days=add_time)
            dates.append(dates[-1]+days)
            # plot residuals
            residuals = self.normalize(residuals)
            plt.figure(figsize=(64,24))
            plt.title(feat, size=48)
            plt.plot(range(limits[1], limits[1]+len(residuals)), residuals, label='residuals', lw=3)
            plt.xlim(0, limits[1]+len(residuals))
            plt.legend(fontsize=48)
            plt.xticks(limits, [date.date() for date in dates], size=12, rotation=45, ha="right") 
            plt.yticks(size=48)
            plt.grid()
        # print anomalies
        res = DataFrame(data=anomalies, index=dates[1:], columns=df.columns)
        return res


    '''
    Mehtod that computes the VAR model for a set of feats.
        - Inputs: df (DataFrame that contains feature values), 
                  frequency (time period used to group data: 'W', 'D', ...)
        - Outputs: data (the same DataFrame, after removing points outside the IRQ boundaries)        
    '''
    def var_model(self,
                  df,
                  frequency='W'):
        # last date, given as prediction
        add_time = 0
        if frequency == 'W':
            add_time = 7
        elif frequency == 'D':
            add_time = 1
        dates = []
        limits = [0]
        tmp = 0 
        groups = df.groupby(pd.Grouper(freq=frequency))
        gp = groups.apply(lambda x: x.name)
        anomalies = []
        residuals = []
        n = len(groups.apply(lambda x: x.name))
        c = 0
        filtered_data = []
        for date, group in groups:  
            anomalies_tmp = []
            residuals_tmp = []
            # print(gp[c])
            dates.append(gp[c])
            data = group.dropna()
            data = data[~np.isnan(data)]
            data = data[np.isfinite(data)]
            if not data.shape[0] > 1:
                c += 1
                #residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                anomalies.append([0]*filtered_data.shape[1])
                limits.append(tmp + len(filtered_data))
                tmp += len(filtered_data)
                continue                
            # train VAR model after removing outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
            filtered_data.dropna(inplace=True)
            if filtered_data.shape[0] > 0:
                # avoid constants
                filtered_data = filtered_data.loc[:, (filtered_data != filtered_data.iloc[0]).any()]
            else:
                c += 1
                #residuals = np.concatenate((residuals, [0]*len(filtered_data)), axis=None)
                anomalies.append([0]*filtered_data.shape[1])
                limits.append(tmp + len(filtered_data))
                tmp += len(filtered_data)
                continue
            #print(model_fit.summary())
            model = VAR(filtered_data)
            model_fit = model.fit()
            #print('Lag: %s' % model_fit.k_ar)
            #print('Coefficients: %s' % model_fit.params)
            if c == 0:
                limits.append(len(filtered_data))
                tmp += len(filtered_data)
                threshold = [np.inf]*df.shape[1]
            # make predictions
            c += 1
            if c < n:
                try:
                    data_sig = groups.get_group(gp[c])
                    data_sig.dropna(inplace=True)
                except Exception as ex:
                    print('No data found for ' + str(gp[c]))
                    anomalies.append([0]*filtered_data.shape[1])
                    residuals.append(np.zeros((filtered_data.shape[0],filtered_data.shape[1])))
                    limits.append(tmp + len(filtered_data))
                    tmp += len(filtered_data)
                    continue
                #data_sig = data_sig[~np.isnan(data_sig)]
                #data_sig = data_sig[np+.isfinite(data_sig)]
                if model_fit.k_ar == len(model_fit.params):
                    print('Constant values used for training at ' + str(gp[c]))
                    anomalies.append([0]*filtered_data.shape[1])
                    residuals.append(np.zeros((filtered_data.shape[0],filtered_data.shape[1])))
                    limits.append(tmp + len(data_sig))
                    tmp += len(data_sig)
                    continue
            else:
                data_sig = DataFrame(filtered_data)
            data_sig = data_sig[~np.isnan(data_sig)]
            data_sig = data_sig[np.isfinite(data_sig)]
            data_sig = DataFrame(data_sig.as_matrix().astype(np.float), columns=df.columns)
            data_sig.dropna(inplace=True)
            limits.append(tmp + len(data_sig))
            tmp += len(data_sig)
            ## ARIMA
            #model = ARIMA(filtered_data, order=(len(data_sig)-1,1,0))
            #model_fit = model.fit(disp=False)
            #arma_mod30 = sm.tsa.ARMA(filtered_data, (3,0)).fit(disp=False)
            #arma_mod30.predict(start=len(filtered_data), end=len(filtered_data)+len(data_sig)-1, dynamic=True)
            #print(arma_mod30.resid)        
            predictions = model_fit.forecast(model_fit.y, steps=len(data_sig))    
            #converting predictions to dataframe
            pred = pd.DataFrame(index=range(0,len(predictions)),columns=[filtered_data.columns])
            for j in range(0,filtered_data.shape[1]):
                for i in range(0, len(predictions)):
                    pred.iloc[i][j] = predictions[i][j]            
            i = 0
            for feat in filtered_data.columns:
                #print('rmse value for', feat, 'is : ', np.sqrt(mean_squared_error(pred[feat], data_sig[feat])))
                anomalies_tmp.append(np.round(len(np.where(abs(pred[feat].values.ravel()-data_sig[feat].values)>threshold[i])[0])/data_sig.shape[0], decimals=3))
                residuals_tmp.append(abs(pred[feat].values.ravel()-data_sig[feat].values))
                threshold[i] = np.mean(abs(pred[feat].values.ravel()-data_sig[feat].values))
                print('mean value for', feat, 'is : ', threshold[i])
                i += 1
            #residuals = DataFrame(model_fit.resid)
            #error = mean_squared_error(data_sig, predictions)
            #print(gp[c])
            #print('Test MSE: %.3f \n' % error)
            anomalies.append(anomalies_tmp)
            residuals.append(residuals_tmp)
        # add days to last prediction residuals
        days = datetime.timedelta(days=add_time)
        dates.append(dates[-1]+days)
        anomalies = DataFrame(anomalies, index=dates[1:], columns=filtered_data.columns)
        residuals = DataFrame(residuals, index=dates[1:], columns=filtered_data.columns)
        # plot residuals
        for feat in anomalies.columns:
            res = []
            for n in range(residuals[feat].shape[0]):
                res = np.concatenate((res, residuals[feat].values[n]), axis=None)
            plt.figure(figsize=(64,24))
            plt.title(feat, size=48)
            plt.plot(range(limits[1], limits[1]+len(res)), res, label='residuals', lw=3)
            plt.xlim(0, limits[1]+len(residuals))
            plt.legend()
            plt.xticks(limits, [date.date() for date in dates], rotation=45, ha="right", size=32) 
            plt.yticks(size=48)
            plt.grid()
        # print anomalies
        res = DataFrame(data=anomalies, index=dates[1:], columns=df.columns)
        return res


    '''
        Normalize data.
            Inputs: 
                - x: set of data
            Output: (v/norm)
    '''
    def normalize(self,
                  x):
        return (x-min(x))/(max(x)-min(x))
    

    '''
    Mehtod that draws a lag plot (t vs t+1 values) for each feature.
        - Inputs: data (DataFrames that contains feature values)
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
            plt.figure(figsize=(14,6))
            plt.title(feat)
            plt.grid()  
            pd.plotting.lag_plot(data[feat])                
            a = np.min(data[feat][np.where(data[feat]!=-np.inf)[0]])
            b = np.max(data[feat][np.where(data[feat]!=np.inf)[0]])
            plt.plot([a, b], [a, b], c='r')

            
    '''
    Mehtod that draws the autocorrelation plot of each feature.
        - Inputs: data (DataFrames that contains feature values), lags
        - Outputs: 
    ''' 
    def autocorrelation(self,
                        data,
                        lags=30):        
        for feat in data.columns:
            fig = plt.figure(figsize=(12,8))
            ax1 = fig.add_subplot(211)  
            fig = sm.graphics.tsa.plot_acf(data[feat].values.squeeze(), lags=lags, ax=ax1)
            ax1.set_title("Autocorrelation " + str(feat))
            ax2 = fig.add_subplot(212)
            fig = sm.graphics.tsa.plot_pacf(data[feat], lags=lags, ax=ax2)
            ax2.set_title("Partial autocorrelation " + str(feat))                            
        