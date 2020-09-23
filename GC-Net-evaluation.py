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
path_gc = '../../AWS_Processing/Input/GCnet/20190501_jaws/'
station = 'DYE-2'
df_gc = gnl.load_gcnet(path_gc, station)

df_gc['ta_tc1']=df_gc['ta_tc1']-273.15
df_gc['ta_tc2']=df_gc['ta_tc2']-273.15
df_gc['ta_cs1']=df_gc['ta_cs1']-273.15
df_gc['ta_cs2']=df_gc['ta_cs2']-273.15

# loading data from U. Calgary
path_ucalg = '../../SAFIRE model/Input/Weather data/data_DYE-2_Samira_hour.txt'
df_samira = gnl.load_promice(path_ucalg)

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_samira['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_samira['time'].values[-1]]
# df_gc.loc[df_gc.time>=datetime.datetime(2016, 10, 15, 9, 0)] = np.NaN

# df_samira = df_samira.loc[df_samira.time>=df_gc['time'].iloc[0]]
# df_samira = df_samira.loc[df_samira.time<=df_gc['time'].iloc[-1]]


# joining datasets
df_all = pd.concat([df_gc, df_samira], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')
df_all['Albedo'] = df_all['ShortwaveRadiationUpWm2'] / df_all['ShortwaveRadiationDownWm2']
df_interpol = df_all.interpolate(method='time')
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()


varname1 =  ['fsus', 'fsds', 'fsus_adjusted','fsds_adjusted','alb']
varname2 =  ['ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2',
             'ShortwaveRadiationUpWm2', 'ShortwaveRadiationDownWm2','Albedo']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,'U.Calg.', station+'_SWrad')

varname1 =  ['ta_tc1','ta_tc2','ta_cs1','ta_cs2']
varname2 =  ['AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC', 'AirTemperatureC']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,'U.Calg.', station+'_temp')

varname1 =  ['rh1','rh2','ps']
varname2 =  ['RelativeHumidity', 'RelativeHumidity', 'AirPressurehPa']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,'U.Calg.', station+'_rh_pres')

varname1 =  ['wspd1','wspd2','wdir1','wdir2']
varname2 =  [ 'WindSpeedms', 'WindSpeedms','WindDirectiond','WindDirectiond']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,'U.Calg.', station+'_wind')


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
path_promice = '../../AWS_Processing/Input/PROMICE/EGP_hour_v03.txt'
df_egp = pd.read_csv(path_promice,delim_whitespace=True)
df_egp['time'] = df_egp.Year * np.nan

for i, y in enumerate(df_egp.Year.values):
    df_egp.time[i] = datetime.datetime(int(y), df_egp['MonthOfYear'].values[i], df_egp['DayOfMonth'].values[i],df_egp['HourOfDay(UTC)'].values[i])

#set invalid values (-999) to nan 
df_egp[df_egp==-999.0]=np.nan

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_egp['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_egp['time'].values[-1]]

df_egp = df_egp.loc[df_egp.time>=df_gc['time'].iloc[0]]
df_egp = df_egp.loc[df_egp.time<=df_gc['time'].iloc[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_egp], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')
# df_all['Albedo'] = df_all['ShortwaveRadiationUpWm2'] / df_all['ShortwaveRadiationDownWm2']
df_all['RelativeHumidity_wrt'] = gnl.RH_ice2water(df_all['RelativeHumidity(%)'] ,
                                                       df_all['AirTemperature(C)'])
df_all['SpecHum'] = gnl.RH2SpecHum(df_all['RelativeHumidity(%)'] ,
                                                       df_all['AirTemperature(C)'] ,
                                                       df_all['AirPressure(hPa)'] )*1000
df_all.pressure_Avg = df_all.pressure_Avg+300
df_all.pressure_Avg.loc[df_all.pressure_Avg<690] = np.nan
df_all['AirPressure(hPa)'].loc[df_all['AirPressure(hPa)']>750] = np.nan
df_all['Dir_Avg(1)']=360-df_all['Dir_Avg(1)']
    
df_interpol = df_all.interpolate(method='time')
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()

# %% Plotting
varname1 =  ['tc_air_Avg(1)', 'tc_air_Avg(2)', 't_air_Avg(1)','t_air_Avg(2)']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)','AirTemperature(C)','AirTemperature(C)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,'PROMICE', station+'_temp')

varname1 =  ['rh_Avg(1)', 'rh_Avg(2)','SpecificHumidity(g/kg)', 'pressure_Avg']
varname2 =  ['RelativeHumidity_wrt','RelativeHumidity_wrt','SpecHum', 'AirPressure(hPa)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,'PROMICE', station+'_rh_pres')


varname1 =  [ 'U_Avg(1)',  'U_Avg(2)', 'Dir_Avg(1)', 'Dir_Avg(2)']
varname2 =  ['WindSpeed(m/s)', 'WindSpeed(m/s)', 'WindDirection(d)', 'WindDirection(d)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,'PROMICE', station+'_wind')

#%% making tables


