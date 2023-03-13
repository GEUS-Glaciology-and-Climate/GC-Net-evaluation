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
import nead
import warnings
import os
import tocgen
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore")
plt.close('all')
comp_matrix = np.array([['Swiss Camp', 'SWC'], 
                ['NASA-U', 'NAU'],
                ['NASA-E', 'NAE'],
                ['GITS', 'CEN'],
                ['GITS', 'CEN2'],
                ['NEEM', 'NEM'],
                ['E-GRIP', 'EGP'],
                ['Saddle', 'SDL'],
                ['DYE2', 'U. Calg.'],
                ['DYE2', 'DY2'],
                ['Summit', 'DMI'],
                ['Summit', 'NOAA']])

# Comparing to GEUS, U.Calg. and NOAA AWS
site_list = pd.read_csv('Input/GC-Net_location.csv', header=0, skipinitialspace=True)
path_to_L1 = '../GC-Net-Level-1-data-processing/L1/'
plt.close('all')
f = open("out/L1_vs_other_AWS/report.md", "w")
# 'Swiss Camp', 'NASA-U','GITS','NEEM','E-GRIP','Saddle', 'Summit','DYE2',

var_list1 = [ 'ISWR','OSWR', 'Alb','TA1','TA2', 'P', 'RH1','RH2','SH1','SH2',  'VW1','VW2','DW1','DW2', 'LHF', 'SHF']
var_list2 = [ 'dsr_cor',  'usr_cor','albedo','t_l', 't_u', 'p_u']+['rh_u','rh_l','qh_l', 'qh_u']+ [ 'wspd_l', 'wspd_u','wdir_l','wdir_u']+ [ 'dlhf_u', 'dshf_u']
ylabels = ['Downward shortwave radiation, $Wm^{-2}$', 
           'Upward shortwave radiation, $Wm^{-2}$', 
           'Albedo, -',
           'Air temperature 1, $^oC$', 
           'Air temperature 2, $^oC$', 
           'Air pressure, hPa',
           'Relative Humidity 1, %', 
           'Relative Humidity 2, %',  
           'Specific humidity 1, $gkg^{-1}$',
           'Specific humidity 2, $gkg^{-1}$',
           'Wind speed 1, $ms^{-1}$', 
           'Wind speed 2, $ms^{-1}$',
           'Wind direction 1, $^o$',
           'Wind direction 2, $^o$',
           'Latent heat flux, $Wm^{-2}$', 
           'Sensible heat flux, $Wm^{-2}$']
ABC = 'ABCDEFGHIJKL'

print('Variable AWS1 AWS2 ME RMSE')
for var1, var2, lab in zip(var_list1, var_list2, ylabels):
    fig, ax = plt.subplots(9,1, figsize=(8,11))
    fig.subplots_adjust(hspace=0.8, top=0.95, bottom=0.02)
    count = -1
    for i in range(comp_matrix.shape[0]):
        site = comp_matrix[i,0]
        name_sec = comp_matrix[i,1]
        ID = site_list.loc[site_list.Name==site, 'ID'].iloc[0]
        site = site.replace(' ','')
        df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site)).to_dataframe()
        df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
        df_L1 = df_L1.set_index('timestamp')
        df_L1[df_L1==-999] = np.nan
        
        df_gc = df_L1.copy()
        if os.path.exists('Input/GEUS stations/'+name_sec+'_hour_v01.csv'):
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour_v01.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        elif os.path.exists('Input/data_'+site+'_Samira_hour.txt'):
            df_sec = gnl.load_ucalg('Input/data_'+site+'_Samira_hour.txt')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('Input/GEUS stations/'+name_sec+'_hour.csv'):
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('./Input/GEUS stations/'+name_sec+'_hour_v04.csv'):
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour_v04.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if name_sec == 'DMI':
            df_sec = gnl.load_dmi()
        if name_sec == 'NOAA':
            df_sec = gnl.load_noaa()        

        if name_sec == 'SWC':
            df_sec['wdir_u'] = (df_sec['wdir_u'] -160) % 360
            df_gc.index = df_gc.index +pd.Timedelta('2H')

        # selecting overlapping data
        df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
        df_sec = df_sec.loc[slice(df_gc.index[0], df_gc.index[-1]),:]
        df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
        df_sec['net_rad'] = df_sec.dsr_cor-df_sec.usr_cor+df_sec.dlr-df_sec.ulr
    
        if var2 not in df_sec.columns:
            continue
        if df_sec[var2].isnull().all():
            continue
        if var1 not in df_gc.columns:
            continue
        if df_gc[var1].isnull().all():
            continue
        if var1 in ['VW1','VW2','DW1','DW2']:
            if site == 'GITS':
                continue
        count = count+1
        ax[count].plot(df_gc[var1], label = 'GC-Net AWS', alpha=0.6, zorder=10)
        ax[count].plot(df_sec[var2], label = 'secondary AWS', alpha=0.8)
        
        ax[count].set_ylabel(lab.split(', ')[1])
        xlow = np.maximum(df_L1[var1].first_valid_index(), df_sec[var2].first_valid_index())
        xhigh = np.minimum(df_L1[var1].last_valid_index(), df_sec[var2].last_valid_index())
        ax[count].set_xlim(xlow, xhigh)
        if count==0:
            ax[count].legend(loc='upper right', bbox_to_anchor=(1, 2))
        ax[count].set_title(ABC[count]+') '+site+ ' AWS vs. '+name_sec+' AWS',
             loc='left')
        print(count, var1, site,name_sec,
              np.round((df_gc[var1] - df_sec[var2]).mean(),2), 
              np.round(np.sqrt(((df_gc[var1] - df_sec[var2])**2).mean()),2))
        ax[count].set_xlabel('')
        # Define the date format
        # date_form = DateFormatter("%Y-%m")
        # ax[count].xaxis.set_major_formatter(date_form)
        if (xhigh-xlow)>pd.Timedelta(days=365):
            ax[count].xaxis.set_major_locator(mdates.YearLocator())
            ax[count].xaxis.set_minor_locator(mdates.MonthLocator())
            ax[count].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        else:
            ax[count].xaxis.set_major_locator(mdates.MonthLocator())
            ax[count].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

        del df_L1
        del df_sec
    plt.suptitle(lab.split(', ')[0], y=1.02)
    ax[count].set_xlabel('Time')

    for k in range(count+1,len(ax)):
        ax[k].axis('off')
    fig.savefig('out/evaluation_'+lab.split(', ')[0].replace(' ','_'),bbox_inches='tight')
