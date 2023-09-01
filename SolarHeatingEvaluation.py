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
comp_matrix = np.array([
                ["GITS", "CEN"],
                ["GITS", "CEN2"],
                ["Swiss Camp", "SWC"], 
                ["NASA-U", "NAU"],
                ["NASA-E", "NAE"],
                ["NEEM", "NEM"],
                ["EastGRIP", "EGP"],
                ["Saddle", "SDL"],
                ["DYE-2", "U. Calg."],
                ["DYE-2", "DY2"],
                ["Summit", "DMI"],
                ])

# Comparing to GEUS, U.Calg. and NOAA AWS
site_list = pd.read_csv("Input/GC-Net_location.csv", header=0, skipinitialspace=True)
path_to_L1 = "../GC-Net-Level-1-data-processing/L1/hourly/"
plt.close('all')

var_list1 = [ 'TA1','TA2',]
var_list2 = [ 't_l', 't_u']
ylabels = [' ',  ' ' ]
ABC = 'ABCDEFGHIJKL'

fig, ax = plt.subplots(4,3, figsize=(14,8), sharex=True, sharey=True)
ax=ax.flatten(order='F')
fig.subplots_adjust(hspace=0.2, top=0.95, bottom=0.1)
count = -1
for var1, var2, lab in zip(var_list1, var_list2, ylabels):
    for i in range(comp_matrix.shape[0]):
        site = comp_matrix[i,0]
        name_sec = comp_matrix[i,1]
        ID = site_list.loc[site_list.Name==site, 'ID'].iloc[0]
        site = site.replace(' ','')
        df_L1 = nead.read(path_to_L1 +site +".csv").to_dataframe()
        df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
        df_L1 = df_L1.set_index('timestamp')
        df_L1[df_L1==-999] = np.nan
        
        df_gc = df_L1.copy()
        if os.path.exists('Data/GEUS stations/'+name_sec+'_hour_v01.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour_v01.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        elif os.path.exists('Data/GEUS stations/'+name_sec+'_hour.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        elif os.path.exists('Data/data_'+site+'_Samira_hour.txt'):
            df_sec = gnl.load_ucalg('Input/data_'+site+'_Samira_hour.txt')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('Data/GEUS stations/'+name_sec+'_hour.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('./Data/GEUS stations/'+name_sec+'_hour_v04.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour_v04.csv')
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
            print('no',var2,'at',name_sec)
            continue
        if df_sec[var2].isnull().all():
            print('no overlapping',var2,'at',name_sec)
            continue
        if var1 not in df_gc.columns:
            print('no',var1,'at',site)
            continue
        if df_gc[var1].isnull().all():
            print('no overlapping',var1,'at',site)
            continue

        count = count+1
        
        x = df_sec.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200) , 'dsr_cor'].values
        y = df_gc.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200),var1] - df_sec.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200), var2]
        y = y.values
        ind = ~np.isnan(x+y)
        x=x[ind].reshape(-1, 1)
        y=y[ind].reshape(-1, 1)
        if len(x) == 0:
            print('no overlapping data at',site,name_sec)
            continue
            
        ax[count].plot(x,y, 'k',
                       marker='.',linestyle='None')
        
        from sklearn import datasets, linear_model
        from sklearn.metrics import mean_squared_error, r2_score
        regr = linear_model.LinearRegression()
        regr.fit(x, y)
        y_pred = regr.predict(x)
        import statsmodels.api as sm
        X2 = sm.add_constant(x)
        est = sm.OLS(y, X2)
        est2 = est.fit()
                
        ax[count].plot([min(x), max(x)], regr.predict([min(x), max(x)]), 
                       color="red", linewidth=3)
        ax[count].set_ylabel('')
        ax[count].set_xlabel('')
        if count>4:
            ax[count].yaxis.set_label_position("right")
            ax[count].yaxis.tick_right()

        ax[count].set_title(ABC[count]+') '+site+": R$^2$=%.2f p=%.2f" % (r2_score(y, y_pred),est2.pvalues[-1]), loc='left')
        ax[count].grid()
        del df_L1
        del df_sec
    plt.suptitle('Air temperature difference as a function of incoming radiation', y=1.02)
    fig.text(0.5, 0.02, "Incoming shortwave radiation (W m$^{-2}$)", ha="center", va="center", fontsize=14)
    fig.text(0.02, 0.5,
        "Unventilated minus ventilated hourly air temperature (W m$^{-2}$)",
        fontsize=14, ha="center", va="center", rotation="vertical",
    )

    fig.savefig('out/solar_heating_evaluation_'+lab.split(', ')[0].replace(' ','_'),bbox_inches='tight')
# %% 

fig, ax = plt.subplots(6,2, figsize=(14,8))
ax=ax.flatten(order='F')
fig.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1)
count = -1
for var1, var2, lab in zip(var_list1, var_list2, ylabels):
    for i in range(comp_matrix.shape[0]):
        site = comp_matrix[i,0]
        name_sec = comp_matrix[i,1]
        ID = site_list.loc[site_list.Name==site, 'ID'].iloc[0]
        site = site.replace(' ','')
        df_L1 = nead.read(path_to_L1 +site +".csv").to_dataframe()
        df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
        df_L1 = df_L1.set_index('timestamp')
        df_L1[df_L1==-999] = np.nan
        
        df_gc = df_L1.copy()
        if os.path.exists('Data/GEUS stations/'+name_sec+'_hour_v01.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour_v01.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        elif os.path.exists('Data/GEUS stations/'+name_sec+'_hour.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        elif os.path.exists('Data/data_'+site+'_Samira_hour.txt'):
            df_sec = gnl.load_ucalg('Input/data_'+site+'_Samira_hour.txt')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('Data/GEUS stations/'+name_sec+'_hour.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
        if os.path.exists('./Data/GEUS stations/'+name_sec+'_hour_v04.csv'):
            df_sec = pd.read_csv('Data/GEUS stations/'+name_sec+'_hour_v04.csv')
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
            print('no',var2,'at',name_sec)
            continue
        if df_sec[var2].isnull().all():
            print('no overlapping',var2,'at',name_sec)
            continue
        if var1 not in df_gc.columns:
            print('no',var1,'at',site)
            continue
        if df_gc[var1].isnull().all():
            print('no overlapping',var1,'at',site)
            continue

        count = count+1
        
        df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
        df_sec = df_sec.loc[slice(df_gc.index[0], df_gc.index[-1]),:]
        df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
        
        x = df_sec.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200) , 'dsr_cor'].values
        y = df_gc.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200),var1] - df_sec.loc[(df_sec.wspd_u<2) & (df_sec.dsr_cor>200), var2]
        y = y.values
        ind = ~np.isnan(x+y)
        x=x[ind].reshape(-1, 1)
        y=y[ind].reshape(-1, 1)
        if len(x) == 0:
            print('no overlapping data at',site,name_sec)
            continue
            
        low_wind = df_gc.VW1.resample('D').mean() 
        low_wind = low_wind.loc[low_wind<2]
        for t in (low_wind.index):
            ax[count].axvspan(t, t + pd.Timedelta(days=1), color='orange', alpha=0.1)
            
        ax[count].plot(df_sec.index,df_sec.t_u,  marker='x',alpha=0.5, label='TA1_sec')
        if 't_l' in df_sec.columns:
            ax[count].plot(df_sec.index,df_sec.t_l,  marker='x',alpha=0.5, label='TA2_sec')
            

        ax[count].plot(df_gc.index,df_gc.TA1, marker='+',alpha=0.5, label='TA1_gcn')
        ax[count].plot(df_gc.index,df_gc.TA2, marker='+',alpha=0.5, label='TA2_gcn')
        if count == 0:
            ax[count].legend(loc='upper left', bbox_to_anchor=(-0.3, 1.05))
        ax[count].set_title(site)

        ax[count].grid()
        del df_L1
        del df_sec
    # plt.suptitle('Air temperature difference as a function of incoming radiation', y=1.02)
    # fig.text(0.5, 0.02, "Incoming shortwave radiation (W m$^{-2}$)", ha="center", va="center", fontsize=14)
    fig.text(0.02, 0.5,
        "Hourly air temperature (W m$^{-2}$)",
        fontsize=14, ha="center", va="center", rotation="vertical",
    )

    fig.savefig('out/solar_heating_evaluation_'+lab.split(', ')[0].replace(' ','_'),bbox_inches='tight')

