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
import matplotlib.pyplot as plt
import gcnet_lib as gnl
import sunposition as sunpos
from windrose import WindroseAxes
import nead
np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import xarray as xr

jaws_alias = {'RH1':'rh1','RH2':'rh2','TA1':'ta_tc2','TA2':'ta_tc2','P':'ps',
              'SZA':'zenith_angle', 'ISWR':'fsds', 'OSWR':'fsus',
              'TA3':'ta_cs1','TA4':'ta_cs2','VW1':'wspd1','VW2':'wspd2',
              'DW1':'wdir1','DW2':'wdir1','HS1':'snh1', 'HS2':'snh2'}
import tocgen

# %% Comparing different file versions
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)[1:2]
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

f = open("out/L1_vs_historical_files/report.md", "w")

for site, ID in zip(site_list.Name,site_list.ID):
    print(site)
    plt.close('all')
    f.write('# '+str(ID)+ ' ' + site)
    df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
    df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
    df_L1 = df_L1.set_index('timestamp')
    df_L1[df_L1==-999] = np.nan
    
    path_to_hist_data = '../../../Data/AWS/GC-Net/20190501_jaws/'
    try:
        df_hist_jaws = xr.open_dataset(path_to_hist_data+'%0.2ic.dat_Req1957.nc'%ID).sel(nbnd=1).squeeze().to_dataframe()
    except:
        f.write('no historical file to compare')
        continue
    df_hist_jaws.index = df_hist_jaws.index + pd.Timedelta(minutes=30)
    df_hist_jaws.index = pd.to_datetime(df_hist_jaws.index, utc=True)
    df_hist_jaws[[ 'ta_tc1', 'ta_tc2', 'ta_cs1', 'ta_cs2']] = df_hist_jaws[[ 'ta_tc1', 'ta_tc2', 'ta_cs1', 'ta_cs2']] -273.15
    df_hist_jaws.ps = df_hist_jaws.ps/100
    
    fig, ax = plt.subplots(3,1, figsize=(10,10),sharex=True)
    plt.subplots_adjust(top=0.95)
    # for i, var in enumerate(['RH1','RH2','TA1','TA2','P','ISWR','OSWR', 'SZA']):
    for i, var in enumerate(['ISWR','OSWR', 'SZA']):
        df_L1[var].plot(ax=ax[i], label = 'L1')
        df_hist_jaws[jaws_alias[var]].plot(ax=ax[i], label = 'hist', alpha=0.7)
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])
    plt.legend()
    plt.suptitle(site)
    fig.savefig('out/L1_vs_historical_files/'+site.replace(' ','')+'_1.png')
    f.write('![](out/L1_vs_historical_files/'+site+'_1.png)')
    
    fig, ax = plt.subplots(8,1, figsize=(10,10))
    plt.subplots_adjust(top=0.95)
    for i, var in enumerate(['TA3','TA4','VW1','VW2','DW1','DW2','HS1', 'HS2']):
        df_L1[var].plot(ax=ax[i], label = 'L1')
        df_hist_jaws[jaws_alias[var]].plot(ax=ax[i], label = 'hist', alpha=0.7)
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])
    plt.legend()
    plt.suptitle(site)
    fig.savefig('out/L1_vs_historical_files/'+site+'_2.png')
    f.write('![](out/L1_vs_historical_files/'+site+'_2.png)')

f.close()

tocgen.processFile('out/L1_vs_historical_files/report.md','out/L1_vs_historical_files/report.md')
#%%
fig, ax = plt.subplots(1,1)
df_L1.P.plot(ax=ax)
(df_hist_jaws.ps/100).plot(ax=ax)
plt.legend()
#%%
fig, ax = plt.subplots(2,1)
df_L1.TA1.plot(ax=ax[0], alpha=0.7)
(df_hist_jaws.ta_tc1-273.15).plot(ax=ax[0], alpha=0.7)
df_L1.TA2.plot(ax=ax[1], alpha=0.7)
(df_hist_jaws.ta_tc2-273.15).plot(ax=ax[1], alpha=0.7)

df_L1.TA3.plot(ax=ax[0], alpha=0.7)
(df_hist_jaws.ta_cs1-273.15).plot(ax=ax[0], alpha=0.7)
df_L1.TA4.plot(ax=ax[1], alpha=0.7)
(df_hist_jaws.ta_cs2-273.15).plot(ax=ax[1], alpha=0.7)

plt.legend()
#%% Comparison at Dye-2 with the U. Calg. AWS
# Loading gc-net data
station = 'DYE-2'
df_gc = gnl.load_gcnet('8_dye2.csv')

# loading data from U. Calgary
path_ucalg = 'Input/data_DYE-2_Samira_hour.txt'
df_samira = gnl.load_ucalg(path_ucalg)

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
        df_interpol.index.values[k]).to_pydatetime(), 66.48001, -46.27889,2100)[1]


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
df_gc = gnl.load_gcnet('24_east_grip.csv')

df_gc = df_gc.loc[df_gc['time']<'2018-05-01',:]
df_gc['DW1']=360-df_gc['DW1']
 
# loading data from PROMICE
path_promice = 'Input/PROMICE/EGP_hour_v03.txt'
df_egp = gnl.load_promice(path_promice)

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
        df_interpol.index.values[k]).to_pydatetime(), 75.6, -35.9,2700)[1]
    
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
df_gc = gnl.load_gcnet('gits.csv')
df_gc = df_gc.loc[df_gc['time']<'2019-01-01',:]
# df_gc['OSWR']= df_gc['OSWR']/2
# df_gc['ISWR']= df_gc['ISWR']/2
df_gc.loc[df_gc['P']<500, 'P'] = np.nan
# df_all.P = df_all.P+300
# df_all.P.loc[df_all.P<690] = np.nan
# df_all['AirPressure(hPa)'].loc[df_all['AirPressure(hPa)']>750] = np.nan
# df_all['DW1']=360-df_all['DW1']

# loading data from CEN
path_promice = 'Input/PROMICE/CEN_hour_v03.txt'
df_cen = gnl.load_promice(path_promice)
df_cen['AirPressure(hPa)'].loc[df_cen['AirPressure(hPa)']>900] = np.nan

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_cen['time'][0]]
df_gc = df_gc.loc[df_gc.time<=df_cen['time'].values[-1]]

df_cen = df_cen.loc[df_cen.time>=df_gc['time'].iloc[0]]
df_cen = df_cen.loc[df_cen.time<=df_gc['time'].iloc[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_cen], axis = 0).sort_values(by='time')
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
        df_interpol.index.values[k]).to_pydatetime(), 77.13781, -61.04113,2000)[1]

varname1 =  [ 'ISWR','OSWR', #'fsds_adjusted','fsus_adjusted',
             'alb']
varname2 =  [ 'ShortwaveRadiationDown_Cor(W/m2)',  'ShortwaveRadiationUp_Cor(W/m2)', #'ShortwaveRadiationDownWm2', 'ShortwaveRadiationUpWm2',
             'Albedo']
varname3 = ['SWdown (W/m2)','SWdown (W/m2)',#'SWdown tilt corrected (W/m2)','SWdown tilt corrected (W/m2)',
            'Albedo (-)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3, 'CEN', station+'_SWrad')

# df_gc=df_gc.set_index('timestamp')
# df_cen=df_cen.set_index('time')
# df_gc.index=pd.to_datetime(df_gc.index)
# df_cen.index=pd.to_datetime(df_cen.index)
# # %% linear regression
# msk = df_gc.index.intersection(df_cen.index)
# x = df_gc.loc[msk, 'ISWR'].values
# y = df_cen.loc[msk, 'ShortwaveRadiationDown(W/m2)'].values
# msk = (~np.isnan(x+y) & (y>200)& (x>30))
# p = np.linalg.lstsq(x[msk][..., None], y[msk][..., None])[0]

# plt.figure()
# # plt.plot(x)
# plt.plot(y)
# plt.plot(x*0.385)

# plt.figure()
# plt.scatter(x,y)
# plt.scatter(x[msk], y[msk])

# plt.figure()
# plt.scatter(x/1.82,y)
# # %% 
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

#%% Summit
# Loading gc-net data
station = 'Summit'
df_gc = gnl.load_gcnet('summit.csv')
df_gc = df_gc.loc[df_gc['time']<'2019-01-01',:]
df_gc.loc[df_gc['P']<500, 'P'] = np.nan
df_gc.loc[df_gc['RH1']>100, 'RH1'] = np.nan
df_gc.loc[df_gc['RH2']>100, 'RH2'] = np.nan

# data source : https://www.esrl.noaa.gov/gmd/dv/data/?site=SUM
from os import listdir
from os.path import isfile, join
path_dir = 'Input/Summit/'
file_list = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
file_list = [s for s in file_list if "met" in s]
df_sum = pd.DataFrame()
for i in range(len(file_list)):
    df = pd.read_csv(path_dir+file_list[i], header=None,  delim_whitespace=True)
    df.columns =['site', 'year', 'month', 'day', 'hour', 'WindDirection(d)', 
                    'WindSpeed(m/s)', 'wsf','AirPressure(hPa)', 'AirTemperature(C)', 'ta_10', 'ta_top', 'RelativeHumidity_w', 'rf']
    df_sum = df_sum.append(df)
df_sum[df_sum==-999.0]=np.nan
df_sum[df_sum==-999.90]=np.nan
df_sum[df_sum==-999.99]=np.nan
df_sum[df_sum==-99.9]=np.nan
df_sum[df_sum==-99]=np.nan

df_sum['time'] =pd.to_datetime(df_sum[['year','month','day','hour']], utc = True)
# time_dt = [ datetime.datetime(y, 
#                                       df_sum['month'].values[d],
#                                       df_sum['day'].values[d],
#                                       df_sum['hour'].values[d]) 
#                   for d,y in enumerate(df_sum['year'].values)]
df_sum.loc[df_sum['RelativeHumidity_w']>100, 'RelativeHumidity_w'] = np.nan

df_sum['RelativeHumidity(%)'] = gnl.RH_ice2water(df_sum['RelativeHumidity_w'] ,
                                                   df_sum['AirTemperature(C)'])
df_sum['SpecificHumidity(g/kg)'] = gnl.RH2SpecHum(df_sum['RelativeHumidity(%)'] ,
                                                       df_sum['AirTemperature(C)'] ,
                                                       df_sum['AirPressure(hPa)'] )*1000

# selecting overlapping data
df_gc = df_gc.loc[df_gc.time>=df_sum['time'].iloc[0]]
df_gc = df_gc.loc[df_gc.time<=df_sum['time'].iloc[-1]]

df_sum = df_sum.loc[df_sum.time>=df_gc['time'].iloc[0]]
df_sum = df_sum.loc[df_sum.time<=df_gc['time'].iloc[-1]]

# joining datasets
df_all = pd.concat([df_gc, df_sum], axis = 0).sort_values(by='time')
df_all = df_all.set_index('time')

mask = df_all.copy()
for i in df_all.columns:
    df = pd.DataFrame( df_all[i] )
    df['new'] = ((df.notnull() != df.shift().notnull()).cumsum())
    df['ones'] = 1
    mask[i] = (df.groupby('new')['ones'].transform('count') < 5) | df_all[i].notnull()
df_interpol = df_all.interpolate(method='time').bfill()[mask]
df_interpol = df_interpol[~df_interpol.index.duplicated(keep='first')].resample('h').asfreq()
# df_interpol['sza'] = np.nan
# for k in range(len(df_interpol['sza'])-1):
#     df_interpol['sza'][k] =   sunpos.observed_sunpos(  pd.Timestamp(
#         df_interpol.index.values[k]).to_pydatetime(), 72.57972, -38.50454,3300)[1]

# varname1 =  [ 'ISWR','OSWR', #'fsds_adjusted','fsus_adjusted',
#               'alb']
# varname2 =  [ 'ShortwaveRadiationDown_Cor(W/m2)',  'ShortwaveRadiationUp_Cor(W/m2)', #'ShortwaveRadiationDownWm2', 'ShortwaveRadiationUpWm2',
#               'Albedo']
# varname3 = ['SWdown (W/m2)','SWdown (W/m2)',#'SWdown tilt corrected (W/m2)','SWdown tilt corrected (W/m2)',
#             'Albedo (-)']
# gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3, 'CEN', station+'_SWrad')

varname1 =  ['TA1','TA2']#,'ta_cs1','ta_cs2']
varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']#, 'AirTemperatureC', 'AirTemperatureC']
varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)']#,'Air temperature cs1 (deg C)','Air temperature cs2 (deg C)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2,varname3,station, station+'_temp')

varname1 =  ['RH1_w','RH2_w','SpecHum1','SpecHum2','P']
varname2 =  ['RelativeHumidity_w','RelativeHumidity_w',
              'SpecificHumidity(g/kg)','SpecificHumidity(g/kg)','AirPressure(hPa)']
varname3 =  ['RelativeHumidity(%)','RelativeHumidity(%)',
              'SpecificHumidity(g/kg)','SpecificHumidity(g/kg)','AirPressure(hPa)']
gnl.plot_comp(df_all, df_interpol, varname1, varname2, varname3,station, station+'_rh_pres')

# varname1 =  ['VW1','VW2','DW1','DW2']
# varname2 =  [ 'WindSpeed(m/s)', 'WindSpeed(m/s)','WindDirection(d)','WindDirection(d)']
# varname3 =  [ 'WindSpeed(m/s)', 'WindSpeed(m/s)','WindDirection(d)','WindDirection(d)']
# gnl.plot_comp(df_all, df_interpol, varname1, varname2, varname3,station, station+'_wind')
plt.figure()
df_all.loc['2017-05-01':].P.plot(label = 'GC-Net')
df_interpol.loc['2017-05-01':]['AirPressure(hPa)'].plot(label='NOAA')
plt.title('mean difference: '+str((df_all.loc['2017-05-01':].P-df_interpol.loc['2017-05-01':]['AirPressure(hPa)']).mean()))
plt.legend()
# fig = plt.figure(figsize=(10,8))
# ax = WindroseAxes.from_ax(fig=fig)
# ws = np.abs(df_interpol['VW1']-df_interpol['WindSpeed(m/s)'])
# ws[ws<np.nanmean(ws)] = np.nan
# wd = df_interpol['WindDirection(d)']
# ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
# ax.set_legend(title='Wind speed (m/s)')
# ax.set_title(station)
# fig.savefig('./Output/'+station+'_wind_bias_dir.png',bbox_inches='tight', dpi=200)

#% making day_night table

# varname1 =  ['TA1', 'TA2']
# varname2 =  ['AirTemperature(C)', 'AirTemperature(C)']

# gnl.tab_comp(df_all, df_interpol, varname1, varname2, 'Output/stat_'+station)