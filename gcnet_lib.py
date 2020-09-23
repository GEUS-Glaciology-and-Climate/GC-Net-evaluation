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
def plot_comp(df_all, df_interpol, varname1, varname2,txt2, figure_name):
    fig, ax = plt.subplots(np.size(varname1),2,
                           figsize=(15, 3*np.size(varname1)),
                           gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(np.size(varname1)):
        j = 0
        # ax[i, j].plot(df_all.index, df_all[varname1[i]],
        #               'bo--',label=varname1[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname1[i]],
                      'b',label= varname1[i] + ' GC-Net')
        # ax[i, j].plot(df_all.index, df_all[varname2[i]],
        #               'ro--',label=varname2[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname2[i]],
                      'r',label=varname2[i] + ' PROMICE')
        ax[i, j].legend()
        
        x=df_interpol[varname1[i]].values
        y =df_interpol[varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        j = 1
        ax[i, j].scatter(df_interpol[varname1[i]],df_interpol[varname2[i]])
        min_val = np.nanmin(np.minimum(df_interpol[varname1[i]],df_interpol[varname2[i]]).values)
        max_val = np.nanmax(np.maximum(df_interpol[varname1[i]],df_interpol[varname2[i]]).values)
        ax[i, j].plot([min_val, max_val], [min_val, max_val], 'k--',linewidth=4)
        ax[i, j].annotate('R2=%.3f \nRMSE=%.2f \nN=%.0f' % (r2_score(x2,y2),
                                                      mean_squared_error(x2,y2),
                                                      len(x2)),
                        xy=(1.1, 0.5), xycoords='axes fraction',
                        xytext=(0, 0), textcoords='offset pixels',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=18, fontweight='bold')
        ax[i, j].set_xlabel(varname1[i]+ ' GC-Net')
        ax[i, j].set_ylabel(varname2[i] + txt2)
        ax[i, j].set_aspect('equal')
    fig.savefig('./Output/'+figure_name+'.png',bbox_inches='tight', dpi=300)
    
#%% 
# def tab_comp(df_all, df_interpol, varname1, varname2, figure_name):
#     df = pd.DataFrame(columns=varname1)
#     df['metric'] = ['RMSE', 'ME', 'R2', 'N', 'RMSE', 'ME', 'R2', 'N', 'RMSE', 'ME', 'R2', 'N']
#     df['time'] = ['all', 'all', 'all', 'all', 'night', 'night', 'night', 'night', 'day',  'day',  'day',  'day']
#     df.set_index(['metric','time'],inplace=True)
    
#     for i in range(np.size(varname1)):
        
#         x=df_interpol[varname1[i]].values
#         y =df_interpol[varname2[i]].values
#         x2=x[~np.isnan(y)&~np.isnan(x)]
#         y2=y[~np.isnan(y)&~np.isnan(x)]
        
#         df.loc[('R2','all'),varname1[i]] = r2_score(x2,y2)
#         df.loc[('ME','all'),varname1[i]] = np.mean(x2,y2)
#         df.loc[('RMSE','all'),varname1[i]] = mean_squared_error(x2,y2)
#         df.loc[('R2','all'),varname1[i]] = len(x2)),
#                            fontsize=18, fontweight='bold')
#         ax[i, j].set_xlabel(varname1[i])
#         ax[i, j].set_ylabel(varname2[i])
#         ax[i, j].set_aspect('equal')
#     fig.savefig('./Output/'+figure_name+'.png',bbox_inches='tight')
    
#%% Relative humidity tools
def RH_water2ice(RH, T):
    # switch ONLY SUBFREEZING timesteps to with-regards-to-ice

    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH[ind] = RH[ind] * Es_Water[ind]/Es_Ice[ind] 
    return RH

def RH_ice2water(RH, T):
    # switch ALL timestep to with-regards-to-water
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 
    return RH

def RH2SpecHum(RH, T, pres):
    # Note: RH[T<0] needs to be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water
    es_all[T < 0] = Es_Ice[T < 0] 
    
    # specific humidity at saturation
    q_sat = es * es_all/(pres-(1-es)*es_all)

    # specific humidity
    q = RH * q_sat /100
    return q

def SpecHum2RH(q, T, pres):
    # Note: RH[T<0] will be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water
    es_all[T < 0] = Es_Ice
    
    # specific humidity at saturation
    q_sat = es * es_all/(pres-(1-es)*es_all)

    # relative humidity
    RH = q / q_sat *100
    return RH
    


