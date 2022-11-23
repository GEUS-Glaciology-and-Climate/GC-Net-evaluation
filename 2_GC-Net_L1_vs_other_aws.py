# -*- coding: utf-8 -*-
"""
Created on 2022-11-22
@author: bav@geus.dk

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
import os

jaws_alias = {'RH1':'rh1','RH2':'rh2','TA1':'ta_tc2','TA2':'ta_tc2','P':'ps',
              'SZA':'zenith_angle', 'ISWR':'fsds', 'OSWR':'fsus',
              'TA3':'ta_cs1','TA4':'ta_cs2','VW1':'wspd1','VW2':'wspd2',
              'DW1':'wdir1','DW2':'wdir1','HS1':'snh1', 'HS2':'snh2'}
sec_stations = {'SwissCamp': ['SWC'], 
                'NASA-U': ['NAU'],
                'NASA-E': ['NAE'],
                'GITS': ['CEN', 'CEN2'],
                'Tunu-N': ['TUN'],
                'NEEM': ['NEM'],
                'E-GRIP': ['EGP'],
                'Saddle': ['SDL'],
                'SouthDome': ['SDM'],
                'NASA-SE': ['NSE'],
                'DYE2': ['U. Calg.', 'DY2'],
                'Summit': ['NOAA']}
import tocgen

# %% Comparing to GEUS, U.Calg. and NOAA AWS
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0,skipinitialspace=True)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'
plt.close('all')
f = open("out/L1_vs_other_AWS/report.md", "w")
# 
station_list = ['Swiss Camp'] #, 'NASA-U', 'NASA-E','GITS','NEEM','E-GRIP','Saddle', 'DYE2', 'Summit',]
for site in station_list:
    ID = site_list.loc[site_list.Name==site, 'ID'].iloc[0]
    print(site)
    f.write('\n# '+str(ID)+ ' ' + site)
    site = site.replace(' ','')
    df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site)).to_dataframe()
    df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
    df_L1 = df_L1.set_index('timestamp')
    df_L1[df_L1==-999] = np.nan
    
    for name_sec in sec_stations[site]:
        df_gc = df_L1.copy()
        if os.path.exists('Input/GEUS stations/'+name_sec+'_hour_v01.csv'):
            print('found two-boom AWS data')
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour_v01.csv')
        if os.path.exists('Input/GEUS stations/'+name_sec+'_hour.csv'):
            print('found one-boom AWS data')
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour.csv')
        if os.path.exists('./Input/GEUS stations/'+name_sec+'_hour_v04.csv'):
            print('found one-boom AWS data')
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour_v04.csv')
        if os.path.exists('Input/data_'+site+'_Samira_hour.txt'):
            print('reading data from U. Calg. AWS')
            df_sec = gnl.load_ucalg('Input/data_'+site+'_Samira_hour.txt')
        if 'time' in df_sec.columns:
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')

        # selecting overlapping data
        try: 
            df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
            df_sec = df_sec.loc[slice(df_gc.index[0], df_gc.index[-1]),:]
        except:
            pass

        # plotting
        varname1 =  [ 'ISWR','OSWR', 'Alb']
        varname2 =  [ 'dsr_cor',  'usr_cor','albedo']
        varname3 = ['SWdown (W/m2)', 'SWup (W/m2)', 'Albedo (-)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2, varname3, name_sec, site+'_1')
        
        varname1 =  ['TA1','TA2', 'P']
        varname2 =  ['t_l', 't_u', 'p_u']
        varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)', 'Air pressure (hPa)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, site+'_2')
        
        varname1 =  ['RH1','RH2','SH1','SH2']
        varname2 =  ['rh_u','rh_l','qh_l', 'qh_u']
        varname3 =  ['Relative Humidity 1 (%)', 'Relative Humidity 2 (%)', 
                     'Specific humidity 1 (g/kg)','Specific humidity 2 (g/kg)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, site+'_3')
        
        varname1 =  ['VW1','VW2','DW1','DW2']
        varname2 =  [ 'wspd_l', 'wspd_u','wdir_l','wdir_u']
        varname3 =  [ 'Wind speed 1 (m/s)', 'Wind speed 2 (m/s)','Wind direction 1 (d)','Wind direction 2 (d)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, site+'_4')
        
        varname1 =  ['LHF', 'SHF']
        varname2 =  [ 'dlhf_u', 'dshf_u']
        varname3 =  [ 'Latent heat flux (W m-2)', 'Sensible heat flux (W m-2)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, site+'_5')

        # fig = plt.figure(figsize=(10,8))
        # ax = WindroseAxes.from_ax(fig=fig)
        # ws = np.abs(df_sec['VW1']-df_sec['WindSpeedms'])
        # ws[ws<np.nanmean(ws)] = np.nan
        # wd = df_sec['WindDirectiond']
        # ax.bar(wd, ws, normed=True, opening=0.8, edgecolor='white')
        # ax.set_legend(title='Wind speed (m/s)')
        # ax.set_title(site)
        # fig.savefig('./out/L1_vs_other_AWS/'+site+'_wind_bias_dir.png',bbox_inches='tight', dpi=200)
        
        #% making day_night table
        
        # varname1 =  ['TA1', 'TA2']
        # varname2 =  ['AirTemperatureC', 'AirTemperatureC']
        
        # gnl.tab_comp(df_gc, df_sec, varname1, varname2, 'out/stat_'+site+'_'+name_sec)
    for i in range(1,6):
        f.write('\n![](out/L1_vs_other_AWS/'+site+'_'+str(i)+'.png)')

f.close()

tocgen.processFile('out/L1_vs_historical_files/report.md','out/L1_vs_historical_files/report_toc.md')
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

# %% 
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
from sklearn.metrics import r2_score


def RMSD(x, y):
    msk = ~np.isnan(x+y)
    return np.sqrt(np.mean(x.values[msk]**2-y.values[msk]**2))

                   
def MD(x, y):
    return np.mean(x.values - y.values)

                 
def plot_comp(var_list=['RH','TA','P','VW','DW', 'SHF','LHF']):
    fig, ax = plt.subplots(len(var_list),1, figsize=(10,10))
    plt.subplots_adjust(top=0.95)
    for i, var in enumerate(var_list):
        if var in ['RH','TA','VW','DW']:
            df_L1[var+'1'].plot(ax=ax[i], alpha=0.7, label = 'L1_1')
            df_L1[var+'2'].plot(ax=ax[i], alpha=0.7, label = 'L1_2')
        else:
            df_L1[var].plot(ax=ax[i], alpha=0.7, label = 'L1')
    
        df_sec[var].plot(ax=ax[i],color='magenta', alpha=0.5, label = source_name)
    
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])
            ax[i].set_xlabel('')
    plt.legend()
    plt.suptitle(site)
    fig.savefig('out/L1_vs_other_AWS/'+site.replace(' ','')+'_'+source_name.replace(' ','_').replace('.','')+'.png')
    
    fig, ax = plt.subplots(3,3, figsize=(12,7))
    plt.subplots_adjust(top=0.95, hspace=0.3, wspace=0.3, left=0.1,right=0.98)
    ax = ax.flatten()
    # plt.subplots_adjust(top=0.95)
    for i, var in enumerate(var_list):
        ax[i].plot(df_L1[var], df_sec[var],marker='+', linestyle='None')
        min_val = min(df_L1[var].min(), df_sec[var].min())
        max_val = max(df_L1[var].max(), df_sec[var].max())
        ax[i].plot([min_val, max_val], [min_val, max_val], color='k')
        ax[i].set_ylabel(var+' ('+source_name+')')
        ax[i].set_xlabel(var+' (GC-Net)')
        ax[i].grid()
        msk = (df_L1[var].notnull() & df_sec[var].notnull())
        if len(df_L1.loc[msk,var].values)==0:
            stat_text='no values'
        else:
            stat_text = 'R$^2$ = %0.2f\nRMSD = %0.2f\nMD = %0.2f'% \
                (r2_score(df_L1.loc[msk,var].values, df_sec.loc[msk,var].values),
                 RMSD(df_L1.loc[msk,var], df_sec.loc[msk,var]),
                 MD(df_L1.loc[msk,var], df_sec.loc[msk,var]))
             
        ax[i].text(0.02, 0.67, stat_text, transform=ax[i].transAxes,
                   bbox={'facecolor':'white', 'alpha':0.5})
            
        # if i%2==1:
        #     ax[i].yaxis.set_ticks_position('right')
        #     ax[i].yaxis.set_label_position("right")
    
    plt.suptitle(site)
    for k in range(i+1,len(ax)):
        ax[k].set_axis_off()
    fig.savefig('out/L1_vs_other_AWS/'+site.replace(' ','_')+'_'+source_name.replace(' ','').replace('.','')+'_2.png')
    
    if 'TA' not in df_sec.columns:
        df_sec['TA'] = df_sec.TA1
        df_sec['RH'] = df_sec.RH1
        df_sec['RH_i'] = df_sec.RH1_i
    fig, ax = plt.subplots(2,1, figsize=(9,9), sharex=True)
    plt.subplots_adjust(top=0.95, hspace=0.02)
    ax[0].plot(df_sec.TA,df_sec.RH,marker='+',linestyle='None',alpha=0.5, label=source_name)
    ax[0].plot(df_L1.TA1,df_L1.RH1,marker='+',linestyle='None',alpha=0.5, label='GC-Net')
    ax[0].text(0.3, 0.1, 'all values relative to water', transform=ax[0].transAxes)

    ax[1].plot(df_sec.TA,df_sec['RH_i'],marker='+',linestyle='None',alpha=0.5, label=source_name)
    ax[1].plot(df_L1.TA1, 
               gnl.RH_water2ice(df_L1.RH1.values, df_L1.TA1.values),
               marker='+',linestyle='None',alpha=0.5, label='GC-Net')
    ax[1].text(0.3, 0.1, 'subfreezing values relative to ice', transform=ax[1].transAxes)

    plt.suptitle(site)
    ax[1].set_xlabel('Air temperature ($^o$C)')
    ax[0].set_ylabel('Relative humidity (%)')
    ax[1].set_ylabel('Relative humidity (%)')
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim(-60, 5)
    ax[0].set_ylim(40, 100)
    ax[1].set_xlim(-60, 5)
    ax[1].set_ylim(50, 120)
    plt.legend()
    fig.savefig('out/L1_vs_other_AWS/'+site.replace(' ','_')+'_'+source_name.replace(' ','').replace('.','')+'_3.png')
    
# %% Dye-2 vs Samimi's station
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

f = open("out/L1_vs_other_AWS/report.md", "w")
site = 'DYE2'
ID = 8

print(site)
plt.close('all')

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

#% Comparison at Dye-2 with the U. Calg. AWS
# Loading gc-net data

# loading data from U. Calgary
source_name = 'U. Calgary'
path_ucalg = 'Input/AWS_Samimi_Marshall_Dye2_Summer2016.xlsx'
df_sec = pd.read_excel(path_ucalg,skiprows=5)
df_sec.columns = ['timestamp','doy','rec','Batt','loggerT',	'TA',
                  'RH', 'P','ISWR','OSWR','CNR1TK','LWin','LWout',
                  'VW','DW','DW_std']
df_sec = df_sec.set_index('timestamp')
df_sec.index = df_sec.index.tz_localize('utc')
df_sec.index=df_sec.index+pd.Timedelta(minutes=-30)

df_sec = df_sec.resample('H').mean()
df_sec['RH_i'] = gnl.RH_water2ice(df_sec.RH.values, df_sec.TA.values)
# selecting overlapping data
df_L1.index=df_L1.index+pd.Timedelta(hours=-2)
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)

plot_comp(['RH','TA','P','ISWR','OSWR','VW','DW'])

# %% EastGrip vs PROMICE station
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'E-GRIP'
ID = 24

print(site)
df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

# loading secondary data
source_name = 'PROMICE'
path_to_PROMICE = '../PROMICE-AWS-toolbox/out/v03_L3/'
df_sec = gnl.load_promice(path_to_PROMICE+'EGP_hour_v03_L3.txt')

# selecting overlapping data
df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
# df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
# df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)

df_sec = df_sec.rename(columns={'AirPressure(hPa)':'P',
                                'AirTemperature(C)':'TA',
                                'RelativeHumidity_w':'RH',
                                'RelativeHumidity(%)':'RH_i',
                                'WindSpeed(m/s)':'VW',
                                'WindDirection(d)':'DW',
                                'SensibleHeatFlux(W/m2)':'SHF',
                                'LatentHeatFlux(W/m2)':'LHF',
                                'ShortwaveRadiationDown_Cor(W/m2)':'ISWR',
                                'ShortwaveRadiationUp_Cor(W/m2)':'OSWR'})   
plot_comp()

# %% GITS vs PROMICE station
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'GITS'
ID = 4

print(site)
plt.close('all')

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

# loading secondary data
source_name = 'PROMICE'
path_to_PROMICE = '../PROMICE-AWS-toolbox/out/v03_L3/'
df_sec = gnl.load_promice(path_to_PROMICE+'CEN_hour_v03_L3.txt')

# selecting overlapping data
df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
# df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
# df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)

df_sec = df_sec.rename(columns={'AirPressure(hPa)':'P',
                                'AirTemperature(C)':'TA',
                                'RelativeHumidity_w':'RH',
                                'RelativeHumidity(%)':'RH_i',
                                'WindSpeed(m/s)':'VW',
                                'WindDirection(d)':'DW',
                                'SensibleHeatFlux(W/m2)':'SHF',
                                'LatentHeatFlux(W/m2)':'LHF',
                                'ShortwaveRadiationDown_Cor(W/m2)':'ISWR',
                                'ShortwaveRadiationUp_Cor(W/m2)':'OSWR'})
plot_comp(['RH','TA','P','VW','DW','ISWR','OSWR', 'SHF','LHF'])

#%% Summit NOAA
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'Summit'
ID = 6

plt.close('all')
source_name = 'NOAA'
print(site, source_name)

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

# loading secondary data
# data source : https://www.esrl.noaa.gov/gmd/dv/data/?site=SUM
from os import listdir
from os.path import isfile, join
path_dir = 'Input/Summit/'
file_list = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
file_list = [s for s in file_list if "met" in s]
df_sec = pd.DataFrame()
for i in range(len(file_list)):
    df = pd.read_csv(path_dir+file_list[i], header=None,  delim_whitespace=True)
    df.columns =['site', 'year', 'month', 'day', 'hour', 'DW', 
                    'VW', 'wsf','P', 'TA', 'ta_10', 'ta_top', 'RH', 'rf']
    df_sec = df_sec.append(df)
df_sec[df_sec==-999.0]=np.nan
df_sec[df_sec==-999.90]=np.nan
df_sec[df_sec==-999.99]=np.nan
df_sec[df_sec==-99.9]=np.nan
df_sec[df_sec==-99]=np.nan
df_sec['time'] =pd.to_datetime(df_sec[['year','month','day','hour']], utc = True)
df_sec = df_sec.set_index('time')
df_sec.loc['2014-09-27':'2017-07-24','RH'] = np.nan
# selecting overlapping data
df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1=df_L1.resample('H').mean()
df_sec=df_sec.resample('H').mean()
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)
df_sec.loc[df_sec['RH']>120, 'RH'] = np.nan
df_sec['RH_i'] = gnl.RH_water2ice(df_sec.RH.values, df_sec.TA.values)

plot_comp(['RH','TA','P','VW','DW'])

#%% Summit DMI
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'Summit'
ID = 6

plt.close('all')
source_name = 'DMI'
print(site, source_name)

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan
stat_id = '441600'
path_to_sec = '../../../Data/AWS/DMI/data/'
DMI_aliases = {'101': 'TA', '201': 'RH_i', '301':'VW', '365':'DW', '401': 'P', '550': 'Rin'}
df_sec = pd.read_csv(path_to_sec+str(stat_id)+'.csv',sep=';').rename(columns={'Hour(utc)':'Hour'})
df_sec['timestamp'] = pd.to_datetime(df_sec[['Year','Month','Day','Hour']],utc=True)
df_sec = df_sec.set_index('timestamp')
df_sec = df_sec[DMI_aliases.keys()].rename(columns=DMI_aliases)

# selecting overlapping data
# df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1=df_L1.resample('H').mean()
# df_sec=df_sec.resample('H').mean()
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
df_L1 = df_L1.resample('3H').mean()
df_sec = df_sec.resample('3H').mean()

# df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['TA'] = df_L1['TA1'].values
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
# df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
# df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)
df_sec['RH'] = gnl.RH_ice2water(df_sec.RH_i.values, df_sec.TA.values)

plot_comp(['RH','TA','P','VW','DW'])

#%% Summit DMI-NOAA
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'Summit'
ID = 6

plt.close('all')
source_name = 'DMI-NOAA'
print(site, source_name)

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan


stat_id = '441900'
path_to_sec = '../../../Data/AWS/DMI/data/'
DMI_aliases = {'101': 'TA', '201': 'RH', '301':'VW', '365':'DW', '401': 'P', '550': 'Rin'}
df_sec = pd.read_csv(path_to_sec+str(stat_id)+'.csv',sep=';').rename(columns={'Hour(utc)':'Hour'})
df_sec['timestamp'] = pd.to_datetime(df_sec[['Year','Month','Day','Hour']],utc=True)
df_sec = df_sec.set_index('timestamp')
df_sec = df_sec[DMI_aliases.keys()].rename(columns=DMI_aliases)

# selecting overlapping data
# df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1=df_L1.resample('H').mean()
df_sec=df_sec.resample('H').mean()
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
# df_L1 = df_L1.resample('3H').mean()
msk = df_sec.DW.diff()==0
df_sec.loc[msk, 'DW'] = np.nan
df_sec.loc[msk.shift(-1).fillna(False), 'DW'] = np.nan


# df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['TA'] = df_L1['TA1'].values
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
# df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
# df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)
# df_sec['RH'] = gnl.RH_ice2water(df_sec.RH_i.values, df_sec.TA.values)
df_sec['RH_i'] = gnl.RH_water2ice(df_sec.RH.values, df_sec.TA.values)

plot_comp(['RH','TA','P','VW','DW'])

# %% NASA-U the GC-Netv2
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'NASA-U'
ID = site_list.loc[site_list.Name==site,'ID'].values[0]

plt.close('all')
source_name = 'GC-Net_v2'
print(site, source_name)

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

GCNet_v2_aliases = {'Pressure_L':'P', 'Asp_temp_L':'TA1', 'Asp_temp_U':'TA2',
                    'Humidity_L':'RH1', 'Humidity_U':'RH2', 'WindSpeed_L':'VW1',
                    'WindDirection_L':'DW1', 'WindSpeed_U':'VW2',
                    'WindDirection_U':'DW2', 'SWUpper':'ISWR', 'SWLower':'OSWR',
                    'LWUpper':'ILWR', 'LWLower':'OLWR',
                    'SR1':'HW1', 'SR2':'HW2'}
stat_id = 'NSU'
path_to_sec = '../../../../GitHub/GC-Net_v2/out/'
df_sec = pd.read_csv(path_to_sec+str(stat_id)+'.csv')
df_sec.time = pd.to_datetime(df_sec.time,utc=True)
df_sec = df_sec.set_index('time')
df_sec = df_sec.rename(columns=GCNet_v2_aliases)

# selecting overlapping data
# df_L1.index=df_L1.index+pd.Timedelta(hours=-1)
df_L1=df_L1.resample('H').mean()
df_sec=df_sec.resample('H').mean()
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]
msk = df_L1.DW1.diff()==0
df_L1.loc[msk, 'DW1'] = np.nan
df_L1.loc[msk.shift(-1).fillna(False), 'DW1'] = np.nan

# df_sec['RH'] = gnl.RH_ice2water(df_sec.RH_i.values, df_sec.TA.values)
df_sec['RH1_i'] = gnl.RH_water2ice(df_sec.RH1.values, df_sec.TA1.values)
df_sec['RH2_i'] = gnl.RH_water2ice(df_sec.RH2.values, df_sec.TA2.values)

plot_comp(['RH1','RH2','TA1','TA2','P','VW1','VW2','HW1','HW2'])

#%%  SHF ad LhF calc
from jaws_tools import gradient_fluxes
df_L1['HW1'] = 2.2
df_L1['HW2'] = 3.4
shf, lhf = gradient_fluxes(df_L1)
shf_sec, lhf_sec = gradient_fluxes(df_sec)
 
plt.figure()
plt.plot(shf_sec)
plt.plot(shf)
# %% SWC the GC-Netv2
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

site = 'Swiss Camp'
ID = site_list.loc[site_list.Name==site,'ID'].values[0]

plt.close('all')
source_name = 'GC-Net_v2'
print(site, source_name)

df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site.replace(' ',''))).to_dataframe()
df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
df_L1 = df_L1.set_index('timestamp')
df_L1[df_L1==-999] = np.nan

GCNet_v2_aliases = {'Pressure':'P', 'Asp_temp':'TA',
                    'Humidity':'RH', 'WindSpeed':'VW',
                    'WindDirection':'DW','SWUpper':'ISWR', 'SWLower':'OSWR',
                    'LWUpper':'ILWR', 'LWLower':'OLWR',
                    'SR1':'HW1', 'SR2':'HW2'}
stat_id = 'SWC'
path_to_sec = '../../../../GitHub/GC-Net_v2/out/'
df_sec = pd.read_csv(path_to_sec+str(stat_id)+'.csv')
df_sec.time = pd.to_datetime(df_sec.time,utc=True)
df_sec = df_sec.set_index('time')
df_sec = df_sec.rename(columns=GCNet_v2_aliases)

# selecting overlapping data
df_L1.index=df_L1.index+pd.Timedelta(hours=2)
df_L1=df_L1.resample('H').mean()
df_sec=df_sec.resample('H').mean()
df_L1 = df_L1.loc[df_sec.index[0]:df_sec.index[-1]]
df_sec = df_sec.loc[df_L1.index[0]:df_L1.index[-1]]

# df_L1['TA'] = df_L1[['TA1','TA2','TA3','TA4']].mean(axis=1)
df_L1['TA'] = df_L1['TA1'].values
df_L1['RH'] = df_L1[['RH1','RH2']].mean(axis=1)
# df_L1['DW'] = df_L1[['DW1','DW2']].mean(axis=1)
# df_L1['VW'] = df_L1[['VW1','VW2']].mean(axis=1)
df_L1['DW'] = df_L1[['DW1']].mean(axis=1)
df_L1['VW'] = df_L1[['VW1']].mean(axis=1)
df_sec['RH_i'] = gnl.RH_water2ice(df_sec.RH.values, df_sec.TA.values)

plot_comp(['RH','TA','P','VW','DW','ISWR','OSWR','HW1','HW2'])

