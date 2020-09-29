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
import nead.nead_io as nead
import gcnet_lib as gnl
import sunposition as sunpos
from windrose import WindroseAxes

np.seterr(invalid='ignore')

#%% Comparison at Dye-2 with the U. Calg. AWS
# Loading gc-net data
station = 'DYE-2'
path_gc = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/AWS/GC-Net/WSL/csv_output/'
filename ='8_dye2.csv'
df_gc = nead.read_nead(path_gc+filename)
df_gc[df_gc==-999.0]=np.nan

df_gc['time'] = pd.to_datetime(df_gc['timestamp'].values)- pd.Timedelta(hours=1) #.dt.tz_localize('UTC')
df_gc['TA1']=df_gc['TA1']-273.15
df_gc['TA2']=df_gc['TA2']-273.15
df_gc['ta_cs1']= np.nan #df_gc['ta_cs1']-273.15
df_gc['ta_cs2']= np.nan #df_gc['ta_cs2']-273.15
df_gc['fsds_adjusted']= np.nan 
df_gc['fsus_adjusted']= np.nan 
df_gc['alb']= df_gc['OSWR']/df_gc['ISWR'] 
df_gc.loc[df_gc['alb']>1,'alb']=np.nan
df_gc.loc[df_gc['alb']<0,'alb']=np.nan
df_gc.loc[df_gc['ISWR']<100, 'Albedo'] = np.nan
df_gc['RH1'] = df_gc['RH1']*100
df_gc['RH2'] = df_gc['RH2']*100
df_gc['RH1_w'] = gnl.RH_ice2water(df_gc['RH1'] ,df_gc['TA1'])
df_gc['RH2_w'] = gnl.RH_ice2water(df_gc['RH2'] ,df_gc['TA2'])
df_gc['P']=df_gc['P']/100
df_gc['SpecHum1'] = gnl.RH2SpecHum(df_gc['RH1'], df_gc['TA1'], df_gc['P'] )*1000
df_gc['SpecHum2'] = gnl.RH2SpecHum(df_gc['RH2'], df_gc['TA2'], df_gc['P'] )*1000

# plotting missing RH2
# fig = plt.figure(figsize = [10,5])
# plt.plot(df_gc['time'].values,df_gc['RH2'],linewidth=1)
# plt.ylabel('RH2 (%)')
# plt.xlabel('Year')
# fig.savefig('./Output/Dye-2_missing_RH2.png',bbox_inches='tight', dpi=200)

# loading data from U. Calgary
path_ucalg = '../../SAFIRE model/Input/Weather data/data_DYE-2_Samira_hour.txt'
df_samira = gnl.load_promice(path_ucalg)
df_samira['Albedo'] = df_samira['ShortwaveRadiationUpWm2'] / df_samira['ShortwaveRadiationDownWm2']
df_samira.loc[df_samira['ShortwaveRadiationDownWm2']<100, 'Albedo'] = np.nan
df_samira['RelativeHumidity_w'] = gnl.RH_ice2water(df_samira['RelativeHumidity'] ,
                                                   df_samira['AirTemperatureC'])
df_samira['SpecHum_ucalg'] = gnl.RH2SpecHum(df_samira['RelativeHumidity'] ,
                                                       df_samira['AirTemperatureC'] ,
                                                       df_samira['AirPressurehPa'] )*1000

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_samira['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_samira['time'].values[-1]]
df_gc=df_gc.loc[df_gc.time<='2016-10-15'] 

df_samira = df_samira.loc[df_samira.time>=df_gc['time'].iloc[0]]
df_samira = df_samira.loc[df_samira.time<=df_gc['time'].iloc[-1]]

# joining and interpolating datasets
df_all = pd.concat([df_gc, df_samira], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')
mask = df_all.copy()
for i in df_all.columns:
    df = pd.DataFrame( df_all[i] )
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask[i] = (df.groupby('new')['ones'].transform('count') < 5) | df_all[i].notnull()
df_interpol = df_all.interpolate(method='time').bfill()[mask]
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()
df_interpol['sza'] = np.nan
for k in range(len(df_interpol['sza'])-1):
    df_interpol['sza'][k] =   sunpos.observed_sunpos(  pd.Timestamp(
        df_interpol.index.values[k]).to_pydatetime(), 75.6, -36,2700)[1]
    
# plotting
varname1 =  [ 'ISWR','OSWR', #'fsds_adjusted','fsus_adjusted',
             'alb']
varname2 =  [ 'ShortwaveRadiationDownWm2',  'ShortwaveRadiationUpWm2', #'ShortwaveRadiationDownWm2', 'ShortwaveRadiationUpWm2',
             'Albedo']
varname3 = ['SWdown (W/m2)','SWdown (W/m2)',#'SWdown tilt corrected (W/m2)','SWdown tilt corrected (W/m2)',
            'Albedo (-)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3, 'U.Calg.', station+'_SWrad')

varname1 =  ['TA1','TA2']#,'ta_cs1','ta_cs2']
varname2 =  ['AirTemperatureC', 'AirTemperatureC']#, 'AirTemperatureC', 'AirTemperatureC']
varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)']#,'Air temperature cs1 (deg C)','Air temperature cs2 (deg C)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'U.Calg.', station+'_temp')

varname1 =  ['RH1_w','SpecHum1','P']
varname2 =  ['RelativeHumidity_w','SpecHum_ucalg','AirPressurehPa']
varname3 =  ['Relative Humidity 1 (%)', 
             'Specific humidity 1 (g/kg)','Air pressure (hPa)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'U.Calg.', station+'_rh_pres')

varname1 =  ['VW1','VW2','DW1','DW2']
varname2 =  [ 'WindSpeedms', 'WindSpeedms','WindDirectiond','WindDirectiond']
varname3 =  [ 'Wind speed 1 (m/s)', 'Wind speed 2 (m/s)','Wind direction 1 (d)','Wind direction 2 (d)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'U.Calg.', station+'_wind')

fig = plt.figure(figsize=(10,8))
ax = WindroseAxes.from_ax(fig=fig)
ws = np.abs(df_interpol['VW1']-df_interpol['WindSpeedms'])
ws[ws<np.nanmean(ws)] = np.nan
wd = df_interpol['WindDirectiond']
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend(title='Wind speed (m/s)')
ax.set_title(station)
fig.savefig('./Output/'+station+'_wind_bias_dir.png',bbox_inches='tight', dpi=200)

#% making day_night table

varname1 =  ['TA1', 'TA2']
varname2 =  ['AirTemperatureC', 'AirTemperatureC']

gnl.tab_comp(df_all, df_interpol, varname1, varname2, 'Output/stat_'+station)

#%% Comparison at EastGRIP
# Loading gc-net data
station = 'EGP'
path_gc = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/AWS/GC-Net/WSL/csv_output/'
filename ='24_east_grip.csv'
df_gc = nead.read_nead(path_gc+filename)
df_gc[df_gc==-999.0]=np.nan

df_gc['time'] = pd.to_datetime(df_gc['timestamp'].values)- pd.Timedelta(hours=1)
df_gc = df_gc.loc[df_gc['time']<'2018-05-01',:]
 #.dt.tz_localize('UTC')
df_gc['TA1']=df_gc['TA1']-273.15
df_gc['TA2']=df_gc['TA2']-273.15
df_gc['ta_cs1']= np.nan #df_gc['ta_cs1']-273.15
df_gc['ta_cs2']= np.nan #df_gc['ta_cs2']-273.15
df_gc['fsds_adjusted']= np.nan 
df_gc['fsus_adjusted']= np.nan 
df_gc['alb']= df_gc['OSWR']/df_gc['ISWR'] 
df_gc.loc[df_gc['alb']>1,'alb']=np.nan
df_gc.loc[df_gc['alb']<0,'alb']=np.nan
df_gc.loc[df_gc['ISWR']<100, 'Albedo'] = np.nan
df_gc['RH1'] = df_gc['RH1']*100
df_gc['RH2'] = df_gc['RH2']*100
df_gc['RH1_w'] = gnl.RH_ice2water(df_gc['RH1'] ,df_gc['TA1'])
df_gc['RH2_w'] = gnl.RH_ice2water(df_gc['RH2'] ,df_gc['TA2'])
df_gc['P']=df_gc['P']/100
df_gc['SpecHum1'] = gnl.RH2SpecHum(df_gc['RH1'], df_gc['TA1'], df_gc['P'] )*1000
df_gc['SpecHum2'] = gnl.RH2SpecHum(df_gc['RH2'], df_gc['TA2'], df_gc['P'] )*1000
df_gc['DW1']=360-df_gc['DW1']
 
# loading data from PROMICE
import pytz
path_promice = '../../AWS_Processing/Input/PROMICE/EGP_hour_v03.txt'
df_egp = pd.read_csv(path_promice,delim_whitespace=True)
df_egp['time'] = df_egp.Year * np.nan

for i, y in enumerate(df_egp.Year.values):
    tmp = datetime.datetime(int(y), df_egp['MonthOfYear'].values[i], df_egp['DayOfMonth'].values[i],df_egp['HourOfDay(UTC)'].values[i])
    df_egp.time[i] = tmp.replace(tzinfo=pytz.UTC)

#set invalid values (-999) to nan 
df_egp[df_egp==-999.0]=np.nan
df_egp['RelativeHumidity_w'] = gnl.RH_ice2water(df_egp['RelativeHumidity(%)'] ,
                                                   df_egp['AirTemperature(C)'])

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_egp['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_egp['time'].values[-1]]

df_egp = df_egp.loc[df_egp.time>=df_gc['time'].iloc[0]]
df_egp = df_egp.loc[df_egp.time<=df_gc['time'].iloc[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_egp], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')

mask = df_all.copy()
for i in df_all.columns:
    df = pd.DataFrame( df_all[i] )
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask[i] = (df.groupby('new')['ones'].transform('count') < 5) | df_all[i].notnull()
df_interpol = df_all.interpolate(method='time').bfill()[mask]
df_interpol =  df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()
df_interpol['ta_cs1'].loc[df_interpol['ta_cs1']<=-39.5] = np.nan
df_interpol['ta_cs2'].loc[df_interpol['ta_cs2']<=-39.5] = np.nan

df_interpol['sza'] = np.nan
for k in range(len(df_interpol['sza'])-1):
    df_interpol['sza'][k] =   sunpos.observed_sunpos(  pd.Timestamp(
        df_interpol.index.values[k]).to_pydatetime(), 75.6, -36,2700)[1]
    
# % Plotting
varname1 =  ['TA1','TA2']#,'ta_cs1','ta_cs2']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']#, 'AirTemperatureC', 'AirTemperatureC']
varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)']#,'Air temperature cs1 (deg C)','Air temperature cs2 (deg C)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'PROMICE', station+'_temp')


varname1 =  ['RH1_w', 'RH2_w','SpecHum1','SpecHum2','P']
varname2 =  ['RelativeHumidity_w','RelativeHumidity_w',
             'SpecificHumidity(g/kg)','SpecificHumidity(g/kg)','AirPressure(hPa)']
varname3 =  ['Relative humidity 1 (%)','Relative humidity 2 (%)',
             'Specific humidity 1 (g/kg)','Specific humidity 2 (g/kg)','Air pressure (hPa)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'PROMICE', station+'_rh_pres')
# gnl.day_night_plot(df_all, df_interpol, varname1, varname2, station+'_rh_pres_violin')

varname1 =  [ 'VW1',  'VW2', 'DW1', 'DW2']
varname2 =  ['WindSpeed(m/s)', 'WindSpeed(m/s)', 'WindDirection(d)', 'WindDirection(d)']
varname3 =  ['Wind Speed (m/s)', 'Wind Speed (m/s)', 'Wind Direction (d)', 'Wind Direction (d)']

gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'PROMICE', station+'_wind')
# gnl.day_night_plot(df_all, df_interpol, varname1, varname2, station+'_wind_violin')

fig = plt.figure(figsize=(10,8))
ax = WindroseAxes.from_ax(fig=fig)
ws = np.abs(df_interpol['VW1']-df_interpol['WindSpeed(m/s)'])
ws[ws<np.nanmean(ws)] = np.nan
wd = df_interpol['WindDirection(d)']
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend(title='Wind speed (m/s)')
ax.set_title(station)
fig.savefig('./Output/'+station+'_wind_bias_dir.png',bbox_inches='tight', dpi=200)

#% making day_night table

varname1 =  ['TA1', 'TA2']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']

gnl.tab_comp(df_all, df_interpol, varname1, varname2, 'Output/stat_'+station)

#%% GITS-Camp Century
# Loading gc-net data
station = 'CEN'
path_gc = 'C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Data/AWS/GC-Net/WSL/csv_output/'
filename ='4_gits.csv'
df_gc = nead.read_nead(path_gc+filename)
df_gc[df_gc==-999.0]=np.nan

df_gc['time'] = pd.to_datetime(df_gc['timestamp'].values)- pd.Timedelta(hours=1)
df_gc = df_gc.loc[df_gc['time']<'2019-01-01',:]
df_gc['TA1']=df_gc['TA1']-273.15
df_gc['TA2']=df_gc['TA2']-273.15
df_gc['ta_cs1']= np.nan #df_gc['ta_cs1']-273.15
df_gc['ta_cs2']= np.nan #df_gc['ta_cs2']-273.15
df_gc['fsds_adjusted']= np.nan 
df_gc['fsus_adjusted']= np.nan 
# df_gc['OSWR']= df_gc['OSWR']/2
# df_gc['ISWR']= df_gc['ISWR']/2
df_gc['alb']= df_gc['OSWR']/df_gc['ISWR'] 
df_gc.loc[df_gc['alb']>1,'alb']=np.nan
df_gc.loc[df_gc['alb']<0,'alb']=np.nan
df_gc.loc[df_gc['ISWR']<100, 'Albedo'] = np.nan
df_gc['RH1'] = df_gc['RH1']*100
df_gc['RH2'] = df_gc['RH2']*100
df_gc['RH1_w'] = gnl.RH_ice2water(df_gc['RH1'] ,df_gc['TA1'])
df_gc['RH2_w'] = gnl.RH_ice2water(df_gc['RH2'] ,df_gc['TA2'])
df_gc['P']=df_gc['P']/100
df_gc.loc[df_gc['P']<500, 'P'] = np.nan
df_gc['SpecHum1'] = gnl.RH2SpecHum(df_gc['RH1'], df_gc['TA1'], df_gc['P'] )*1000
df_gc['SpecHum2'] = gnl.RH2SpecHum(df_gc['RH2'], df_gc['TA2'], df_gc['P'] )*1000

# loading data from CEN
path_promice = '../../AWS_Processing/Input/PROMICE/CEN_hour_v03.txt'
df_cen = pd.read_csv(path_promice,delim_whitespace=True)
df_cen['time'] = df_cen.Year * np.nan

for i, y in enumerate(df_cen.Year.values):
    tmp = datetime.datetime(int(y), df_cen['MonthOfYear'].values[i], df_cen['DayOfMonth'].values[i], df_cen['HourOfDay(UTC)'].values[i])
    df_cen.time[i] = tmp.replace(tzinfo=pytz.UTC)

df_cen[df_cen==-999.0]=np.nan
df_cen['Albedo'] = df_cen['ShortwaveRadiationUp(W/m2)'] / df_cen['ShortwaveRadiationDown(W/m2)']
df_cen.loc[df_cen['Albedo']>1,'Albedo']=np.nan
df_cen.loc[df_cen['Albedo']<0,'Albedo']=np.nan
df_cen['AirPressure(hPa)'].loc[df_cen['AirPressure(hPa)']>900] = np.nan
df_cen['RelativeHumidity_w'] = gnl.RH_ice2water(df_cen['RelativeHumidity(%)'] ,df_cen['AirTemperature(C)'])

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_cen['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_cen['time'].values[-1]]

df_cen = df_cen.loc[df_cen.time>=df_gc['time'].iloc[0]]
df_cen = df_cen.loc[df_cen.time<=df_gc['time'].iloc[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_cen], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')

# df_all.P = df_all.P+300
# df_all.P.loc[df_all.P<690] = np.nan
# df_all['AirPressure(hPa)'].loc[df_all['AirPressure(hPa)']>750] = np.nan
# df_all['DW1']=360-df_all['DW1']
mask = df_all.copy()
for i in df_all.columns:
    df = pd.DataFrame( df_all[i] )
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask[i] = (df.groupby('new')['ones'].transform('count') < 5) | df_all[i].notnull()
df_interpol = df_all.interpolate(method='time').bfill()[mask]
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()
df_interpol['sza'] = np.nan
for k in range(len(df_interpol['sza'])-1):
    df_interpol['sza'][k] =   sunpos.observed_sunpos(  pd.Timestamp(
        df_interpol.index.values[k]).to_pydatetime(), 75.6, -36,2700)[1]
# df_interpol=df_interpol.set_index('time')
# df_interpol['ta_cs1'].loc[df_interpol['ta_cs1']<=-40] = np.nan
# df_interpol['ta_cs2'].loc[df_interpol['ta_cs2']<=-40] = np.nan

varname1 =  [ 'ISWR','OSWR', #'fsds_adjusted','fsus_adjusted',
             'alb']
varname2 =  [ 'ShortwaveRadiationDown_Cor(W/m2)',  'ShortwaveRadiationUp_Cor(W/m2)', #'ShortwaveRadiationDownWm2', 'ShortwaveRadiationUpWm2',
             'Albedo']
varname3 = ['SWdown (W/m2)','SWdown (W/m2)',#'SWdown tilt corrected (W/m2)','SWdown tilt corrected (W/m2)',
            'Albedo (-)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3, 'CEN', station+'_SWrad')

varname1 =  ['TA1','TA2']#,'ta_cs1','ta_cs2']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']#, 'AirTemperatureC', 'AirTemperatureC']
varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)']#,'Air temperature cs1 (deg C)','Air temperature cs2 (deg C)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,'CEN', station+'_temp')

varname1 =  ['RH1_w','RH2_w','SpecHum1','SpecHum2','P']
varname2 =  ['RelativeHumidity_w','RelativeHumidity_w',
              'SpecificHumidity(g/kg)','SpecificHumidity(g/kg)','AirPressure(hPa)']
varname3 =  ['RelativeHumidity(%)','RelativeHumidity(%)',
              'SpecificHumidity(g/kg)','SpecificHumidity(g/kg)','AirPressure(hPa)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, varname3,'CEN', station+'_rh_pres')

varname1 =  ['VW1','VW2','DW1','DW2']
varname2 =  [ 'WindSpeed(m/s)', 'WindSpeed(m/s)','WindDirection(d)','WindDirection(d)']
varname3 =  [ 'WindSpeed(m/s)', 'WindSpeed(m/s)','WindDirection(d)','WindDirection(d)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, varname3,'CEN', station+'_wind')


fig = plt.figure(figsize=(10,8))
ax = WindroseAxes.from_ax(fig=fig)
ws = np.abs(df_interpol['VW1']-df_interpol['WindSpeed(m/s)'])
ws[ws<np.nanmean(ws)] = np.nan
wd = df_interpol['WindDirection(d)']
ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
ax.set_legend(title='Wind speed (m/s)')
ax.set_title(station)
fig.savefig('./Output/'+station+'_wind_bias_dir.png',bbox_inches='tight', dpi=200)

#% making day_night table

varname1 =  ['TA1', 'TA2']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']

gnl.tab_comp(df_all, df_interpol, varname1, varname2, 'Output/stat_'+station)