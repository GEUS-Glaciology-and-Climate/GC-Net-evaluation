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

#%%
def load_gcnet(path_gc, station):
    df_gc_m = pd.read_csv('Input/Gc-net_documentation_Nov_10_2000.csv',sep=';')
    station_id = df_gc_m.loc[df_gc_m['Station Name']==station]['StationID'].values[0]
    filename = path_gc + str(station_id).zfill(2) + 'c.dat_Req1957.nc'
    ds = xr.open_dataset(filename)
    df_gc = ds.to_dataframe()
    df_gc=df_gc.reset_index()
    return df_gc

#%%
def load_promice(path_promice):
    df_promice = pd.read_csv(path_promice, delim_whitespace=True)
    df_promice['timestamp'] = df_promice.time

    for i, y in enumerate(df_promice.Year.values):
        df_promice.time[i] = datetime.datetime(int(y), 1, 1)   + datetime.timedelta( days = df_promice.DayOfYear.values[i], hours = df_promice.HourOfDayUTC.values[i]-1) 

    #set invalid values (-999) to nan 
    df_promice[df_promice==-999.0]=np.nan
    return df_promice

#%% 
def plot_comp(df_all, df_interpol, varname1, varname2, figure_name):
    fig, ax = plt.subplots(np.size(varname1),2,
                           figsize=(30, 7*np.size(varname1)),
                           gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(np.size(varname1)):
        j = 0
        ax[i, j].plot(df_all.index, df_all[varname1[i]],
                      'bo--',label=varname1[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname1[i]],
                      'bx:',label= varname1[i] + ' after interp')
        ax[i, j].plot(df_all.index, df_all[varname2[i]],
                      'ro--',label=varname2[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname2[i]],
                      'rx:',label=varname2[i] + ' after interp')
        ax[i, j].legend()
        
        x=df_interpol[varname1[i]].values
        y =df_interpol[varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        j = 1
        ax[i, j].scatter(df_interpol[varname1[i]],df_interpol[varname2[i]])
        ax[i, j].set_title('R2=%.3f RMSE=%.2f N=%.0f' % (r2_score(x2,y2),
                                                      mean_squared_error(x2,y2),
                                                      len(x2)),
                           fontsize=18, fontweight='bold')
        ax[i, j].set_xlabel(varname1[i])
        ax[i, j].set_ylabel(varname2[i])
        ax[i, j].set_aspect('equal')
    fig.savefig('./Output/'+figure_name+'.png',bbox_inches='tight')
