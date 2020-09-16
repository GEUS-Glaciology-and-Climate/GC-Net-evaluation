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


varname1 =  ['fsus', 'fsds', 'fsus_adjusted','fsds_adjusted','alb']
varname2 =  ['ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2',
             'ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2','Albedo']

gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_SWrad')

varname1 =  ['ta_tc1','ta_tc2','ta_cs1','ta_cs2']
varname2 =  ['AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_temp')

varname1 =  ['rh1','rh2','ps']
varname2 =  ['RelativeHumidity', 'RelativeHumidity', 'AirPressurehPa']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_rh_pres')

varname1 =  ['wspd1','wspd2','wdir1','wdir2']
varname2 =  [ 'WindSpeedms', 'WindSpeedms','WindDirectiond','WindDirectiond']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_wind')


#%% Comparison at EastGRIP
# Loading gc-net data
path_gc = 'Input/CR1000_EGRIP_GC-Net_Table046.dat'
station = 'EGP'
station_id = ''
df_gc = pd.read_csv(path_gc, skiprows=[0,2,3,4])

df_gc['time'] = [datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in df_gc.TIMESTAMP.values] 
# df_gc.AirPressurehPa = df_gc.pressure_Avg+300;
# df_gc.AirPressurehPa(df_gc.AirPressurehPa<690) = NaN;
    
# loading data from PROMICE
path_promice = '../AWS_Processing/Input/PROMICE/EGP_hour_v03.txt'
df_egp = pd.read_csv(path_promice,delim_whitespace=True)
df_egp['time'] = df_egp.Year * np.nan

for i, y in enumerate(df_egp.Year.values):
    df_egp.time[i] = datetime.datetime(int(y), df_egp['MonthOfYear'].values[i], df_egp['DayOfMonth'].values[i],df_egp['HourOfDay(UTC)'].values[i])

#set invalid values (-999) to nan 
df_egp[df_egp==-999.0]=np.nan

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_egp['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_egp['time'].values[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_egp], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')
# df_all['Albedo'] = df_all['ShortwaveRadiationUpWm2'] / df_all['ShortwaveRadiationDownWm2']
df_interpol = df_all.interpolate(method='time')
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()

# df_gc.columns
# Out[151]: 
# Index(['TIMESTAMP', 'RECORD', 'LoggerID', 'Year', 'Day_of_Year', 'Hour',
#        'sw_in_Avg', 'sw_ref_Avg', 'tc_air_Avg(1)', 'tc_air_Avg(2)',
#        't_air_Avg(1)', 't_air_Avg(2)', 'rh_Avg(1)', 'rh_Avg(2)', 'U_Avg(1)',
#        'U_Avg(2)', 'Dir_Avg(1)', 'Dir_Avg(2)', 'pressure_Avg', 'SD_1_Avg',
#        'SD_2_Avg', 'sw_in_Max', 'sw_in_Std', 'tc_air_Max(1)', 'tc_air_Max(2)',
#        'tc_air_Min(1)', 'tc_air_Min(2)', 'U_Max(1)', 'U_Max(2)', 'U_Std(1)',
#        'U_Std(2)', 'TRef_Avg', 'Battery', 'time'],
#       dtype='object')
# df_egp.columns
# Out[152]: 
# Index(['Year', 'MonthOfYear', 'DayOfMonth', 'HourOfDay(UTC)', 'DayOfYear',
#        'DayOfCentury', 'AirPressure(hPa)', 'AirTemperature(C)',
#        'AirTemperatureHygroClip(C)', 'RelativeHumidity(%)',
#        'SpecificHumidity(g/kg)', 'WindSpeed(m/s)', 'WindDirection(d)',
#        'SensibleHeatFlux(W/m2)', 'LatentHeatFlux(W/m2)',
#        'ShortwaveRadiationDown(W/m2)', 'ShortwaveRadiationDown_Cor(W/m2)',
#        'ShortwaveRadiationUp(W/m2)', 'ShortwaveRadiationUp_Cor(W/m2)',
#        'Albedo_theta<70d', 'LongwaveRadiationDown(W/m2)',
#        'LongwaveRadiationUp(W/m2)', 'CloudCover', 'SurfaceTemperature(C)',
#        'HeightSensorBoom(m)', 'HeightStakes(m)', 'DepthPressureTransducer(m)',
#        'DepthPressureTransducer_Cor(m)', 'IceTemperature1(C)',
#        'IceTemperature2(C)', 'IceTemperature3(C)', 'IceTemperature4(C)',
#        'IceTemperature5(C)', 'IceTemperature6(C)', 'IceTemperature7(C)',
#        'IceTemperature8(C)', 'TiltToEast(d)', 'TiltToNorth(d)',
#        'TimeGPS(hhmmssUTC)', 'LatitudeGPS(degN)', 'LongitudeGPS(degW)',
#        'ElevationGPS(m)', 'HorDilOfPrecGPS', 'LoggerTemperature(C)',
#        'FanCurrent(mA)', 'BatteryVoltage(V)', 'time'],
#       dtype='object')
#%% Plotting
varname1 =  ['tc_air_Avg(1)', 'tc_air_Avg(2)', 't_air_Avg(1)','t_air_Avg(2)']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)','AirTemperature(C)','AirTemperature(C)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_temp')

varname1 =  ['rh_Avg(1)', 'rh_Avg(2)','pressure_Avg']
varname2 =  ['RelativeHumidity(%)','RelativeHumidity(%)','AirPressure(hPa)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_rh_pres')


varname1 =  [ 'U_Avg(1)',  'U_Avg(2)', 'Dir_Avg(1)', 'Dir_Avg(2)']
varname2 =  ['WindSpeed(m/s)', 'WindSpeed(m/s)', 'WindDirection(d)', 'WindDirection(d)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2, station+'_rh_pres')

