#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:48:01 2018

@author: alberto
"""

import pandas as pd
import numpy as np
import os



class data_management(object):

    
    '''
    Mehtod that loads data for a given asset.
        - Inputs: rootdir, assetId, measurePointId, parameter
        - Outputs: DataFrame, indexed by time
    '''
    def load_data(self,
                  rootdir='/media/alberto/9AE6A9E8E6A9C53B/Users/104166/Data/PETRONOR/data/12_to_19nov_2018/',
                  assetId='H4-FA-0002',
                  measurePointId='SH4',
                  parameter='vibrations'):
        data = []
        date = []
        for root, dirs, files in os.walk(rootdir+parameter):
            for filename in files:
                if filename.endswith((".json")):
                    fullpath = os.path.join(root, filename)            
                    # read the entire file into a python array
                    with open(fullpath, 'rb') as f:
                        file = f.read().decode("utf-8-sig").encode("utf-8")
                    res = pd.read_json(file, lines=True)
                    # parameters not specified
                    if assetId == '' and measurePointId == '':
                        # apply calibration to raw values
                        if parameter == 'vibrations':
                            data.append(np.array(res.Value.values[0])*float(res.Props.values[0][4]['Value']))
                        else:
                            data.append(np.array(res.Value.values[0]))
                        date.append(res.ServerTimeStamp.values[0])
                    else:
                        if measurePointId == '':
                            if res.AssetId.values[0] == assetId:
                                # apply calibration to raw values
                                if parameter == 'vibrations':
                                    data.append(np.array(res.Value.values[0])*float(res.Props.values[0][4]['Value']))
                                else:
                                    data.append(np.array(res.Value.values[0]))
                                date.append(res.ServerTimeStamp.values[0])
                        else:
                            if res.AssetId.values[0] == assetId and res.MeasurePointId.values[0] == measurePointId:
                                # apply calibration to raw values
                                if parameter == 'vibrations':
                                    data.append(np.array(res.Value.values[0])*float(res.Props.values[0][4]['Value']))
                                else:
                                    data.append(np.array(res.Value.values[0]))
                                date.append(res.ServerTimeStamp.values[0])
        df = pd.DataFrame(data=data, index=pd.to_datetime(date))
        df.sort_index(inplace=True)
        return df


    '''
    Mehtod that computes velocity (mm/s) from acceleration (g's peak).
        - Inputs: df, f (frequency)
        - Outputs: velocity
    '''
    def compute_velocity(self,
                         df, 
                         f=5120):
        velocity = 9.81*1000*np.cumsum(df.sub(df.mean(axis=1), axis=0))/f
        return velocity