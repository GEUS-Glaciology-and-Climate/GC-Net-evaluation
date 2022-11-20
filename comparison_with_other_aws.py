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

