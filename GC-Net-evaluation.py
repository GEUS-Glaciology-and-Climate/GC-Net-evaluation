# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics import mean_squared_error, r2_score

import gcnet_lib as gnl

#%% Comparison at Dye-2 with the U. Calg. AWS
# Loading gc-net data
path_gc = '../AWS_Processing/Input/GCnet/20190501_jaws/'
station = 'DYE-2'
df_gc = gnl.load_gcnet(path_gc, station)

df_gc['ta_tc1']=df_gc['ta_tc1']-273.15
df_gc['ta_tc2']=df_gc['ta_tc2']-273.15
df_gc['ta_cs1']=df_gc['ta_cs1']-273.15
df_gc['ta_cs2']=df_gc['ta_cs2']-273.15

# loading data from U. Calgary
path_ucalg = '../SAFIRE model/Input/Weather data/data_DYE-2_Samira_hour.txt'
df_samira = gnl.load_promice(path_ucalg)

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_samira['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_samira['time'].values[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_samira], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')
df_all['Albedo'] = df_all['ShortwaveRadiationUpWm2'] / df_all['ShortwaveRadiationDownWm2']
df_interpol = df_all.interpolate(method='time')
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()


#%% 
varname1 =  ['fsus', 'fsds', 'fsus_adjusted','fsds_adjusted','alb']
varname2 =  ['ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2',
             'ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2','Albedo']

gnl.plot_comp(df_all, df_interpol, varname1, varname2, 'dye-2_SWrad')

varname1 =  ['ta_tc1','ta_tc2','ta_cs1','ta_cs2']
varname2 =  ['AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, 'dye-2_temp')

varname1 =  ['rh1','rh2','ps']
varname2 =  ['RelativeHumidity', 'RelativeHumidity', 'AirPressurehPa']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, 'dye-2_temp')

varname1 =  ['wspd1','wspd2','wdir1','wdir2']
varname2 =  [ 'WindSpeedms', 'WindSpeedms','WindDirectiond','WindDirectiond']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, 'dye-2_temp')


#%% Comparison at EastGRIP
# Loading gc-net data
path_gc = '../AWS_Processing/Input/GCnet/20190501_jaws/'
station = 'Dye-2'
station_id = 8

filename = path_gc + str(station_id).zfill(2) + 'c.dat_Req1957.nc'
ds = xr.open_dataset(filename)
df_gc = ds.to_dataframe()
df_gc=df_gc.reset_index()

# loading data from U. Calgary
path_ucalg = '../SAFIRE model/Input/Weather data/data_DYE-2_Samira_hour.txt'
df_samira = pd.read_csv(path_ucalg,delim_whitespace=True)
df_samira['timestamp'] = df_samira.time

for i, y in enumerate(df_samira.Year.values):
    df_samira.time[i] = datetime.datetime(int(y), 1, 1)   + datetime.timedelta( days = df_samira.DayOfYear.values[i] - 1, hours = df_samira.HourOfDayUTC.values[i]-1) 

#set invalid values (-999) to nan 
df_samira[df_samira==-999.0]=np.nan

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_samira['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_samira['time'].values[-1]]