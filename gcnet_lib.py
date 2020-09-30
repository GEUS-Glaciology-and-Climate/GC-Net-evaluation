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
import sys
sys.path.insert(0,'../jaws/jaws')
from pytablewriter import MarkdownTableWriter
import math      
from matplotlib.patches import Patch
import pytz
import sunposition as sunpos
import nead.nead_io as nead

#%%
def load_gcnet(filename):
    df_gc = nead.read_nead('Input/GC-Net/'+filename)
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
    df_gc['RH1_w'] = RH_ice2water(df_gc['RH1'] ,df_gc['TA1'])
    df_gc['RH2_w'] = RH_ice2water(df_gc['RH2'] ,df_gc['TA2'])
    df_gc['P']=df_gc['P']/100
    df_gc['SpecHum1'] = RH2SpecHum(df_gc['RH1'], df_gc['TA1'], df_gc['P'] )*1000
    df_gc['SpecHum2'] = RH2SpecHum(df_gc['RH2'], df_gc['TA2'], df_gc['P'] )*1000
    return df_gc

#%%
def load_ucalg(path_promice):
    df_samira = pd.read_csv(path_promice, delim_whitespace=True)
    df_samira['time'] = df_samira.Year * np.nan
    df_samira['timestamp'] = df_samira.time

    for i, y in enumerate(df_samira.Year.values):
        tmp = datetime.datetime(int(y), 1, 1)   + datetime.timedelta( days = df_samira.DayOfYear.values[i], hours = df_samira.HourOfDayUTC.values[i]-1) 
        df_samira.time[i] = tmp.replace(tzinfo=pytz.UTC)
    #set invalid values (-999) to nan 
    df_samira[df_samira==-999.0]=np.nan
    df_samira['Albedo'] = df_samira['ShortwaveRadiationUpWm2'] / df_samira['ShortwaveRadiationDownWm2']
    df_samira.loc[df_samira['ShortwaveRadiationDownWm2']<100, 'Albedo'] = np.nan
    df_samira['RelativeHumidity_w'] = RH_ice2water(df_samira['RelativeHumidity'] ,
                                                       df_samira['AirTemperatureC'])
    df_samira['SpecHum_ucalg'] = RH2SpecHum(df_samira['RelativeHumidity'] ,
                                                           df_samira['AirTemperatureC'] ,
                                                           df_samira['AirPressurehPa'] )*1000
    return df_samira

def load_promice(path_promice):
    df_pro = pd.read_csv(path_promice,delim_whitespace=True)
    df_pro['time'] = df_pro.Year * np.nan
    
    for i, y in enumerate(df_pro.Year.values):
        tmp = datetime.datetime(int(y), 
                                df_pro['MonthOfYear'].values[i],
                                df_pro['DayOfMonth'].values[i],
                                df_pro['HourOfDay(UTC)'].values[i])
        df_pro.time[i] = tmp.replace(tzinfo=pytz.UTC)
    
    #set invalid values (-999) to nan 
    df_pro[df_pro==-999.0]=np.nan
    df_pro['Albedo'] = df_pro['ShortwaveRadiationUp(W/m2)'] / df_pro['ShortwaveRadiationDown(W/m2)']
    df_pro.loc[df_pro['Albedo']>1,'Albedo']=np.nan
    df_pro.loc[df_pro['Albedo']<0,'Albedo']=np.nan

    df_pro['RelativeHumidity_w'] = RH_ice2water(df_pro['RelativeHumidity(%)'] ,
                                                       df_pro['AirTemperature(C)'])
    return df_pro
#%% 
def plot_comp(df_all, df_interpol, varname1, varname2, varname3,txt2, figure_name):
    fig, ax = plt.subplots(np.size(varname1),2,
                           figsize=(13, 3*np.size(varname1)),
                           gridspec_kw={'width_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.6, wspace=0.02)
    for i in range(np.size(varname1)):
        j = 0
        # ax[i, j].plot(df_all.index, df_all[varname1[i]],
        #               'bo--',label=varname1[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname1[i]],
                      'b',label= 'GC-Net')
        # ax[i, j].plot(df_all.index, df_all[varname2[i]],
        #               'ro--',label=varname2[i] + ' before')
        ax[i, j].plot(df_interpol.index, df_interpol[varname2[i]],
                      'r',label=txt2,alpha=0.7)
        ax[i, j].legend()
        ax[i, j].set_ylabel(varname3[i])
        
        x=df_interpol[varname1[i]].values
        y =df_interpol[varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        j = 1
        ax[i, j].scatter(df_interpol[varname1[i]],df_interpol[varname2[i]])
        min_val = np.nanmin(np.minimum(df_interpol[varname1[i]],df_interpol[varname2[i]]).values)
        max_val = np.nanmax(np.maximum(df_interpol[varname1[i]],df_interpol[varname2[i]]).values)
        ax[i, j].plot([min_val, max_val], [min_val, max_val], 'k--',linewidth=4)
        ax[i, j].set_title(varname3[i])
        if len(x2)>0:
            ax[i, j].annotate('R2=%.3f \nRMSE=%.2f \nME=%.2f \nN=%.0f' % (r2_score(x2,y2),
                                                          mean_squared_error(x2,y2),
                                                          np.nanmean(x2-y2),
                                                          len(x2)),
                            xy=(1.1, 0.5), xycoords='axes fraction',
                            xytext=(0, 0), textcoords='offset pixels',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=14)
        ax[i, j].set_xlabel(' GC-Net')
        ax[i, j].set_ylabel( txt2)
        ax[i, j].set_aspect('equal')
    fig.savefig('./Output/'+figure_name+'.png',bbox_inches='tight', dpi=200)
    
#%% 
def tab_comp(df_all, df_interpol, varname1, varname2, filename):
    df = pd.DataFrame(columns=varname1)
    df['metric'] = ['RMSE', 'bias', 'R2', 'N', 'RMSE', 'bias', 'R2', 'N', 'RMSE', 'bias', 'R2', 'N']
    df['time'] = ['all', 'all', 'all', 'all', 'night', 'night', 'night', 'night', 'day',  'day',  'day',  'day']
    df.set_index(['metric','time'],inplace=True)
    
    sza=df_interpol['sza']

    day = sza<70
    night = sza > 110
        
    for i in range(np.size(varname1)):
        
        x=df_interpol[varname1[i]].values
        y =df_interpol[varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        df.loc[('R2','all'),varname1[i]] = r2_score(x2,y2)
        df.loc[('bias','all'),varname1[i]] = np.mean(x2-y2)
        df.loc[('RMSE','all'),varname1[i]] = mean_squared_error(x2,y2)
        df.loc[('N','all'),varname1[i]] = len(x2)
             

        x=df_interpol.loc[night,varname1[i]].values
        y =df_interpol.loc[night,varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        df.loc[('R2','night'),varname1[i]] = r2_score(x2,y2)
        df.loc[('bias','night'),varname1[i]] = np.mean(x2-y2)
        df.loc[('RMSE','night'),varname1[i]] = mean_squared_error(x2,y2)
        df.loc[('N','night'),varname1[i]] = len(x2)
        
        x = df_interpol.loc[day,varname1[i]].values
        y = df_interpol.loc[day,varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        df.loc[('R2','day'),varname1[i]] = r2_score(x2,y2)
        df.loc[('bias','day'),varname1[i]] = np.mean(x2-y2)
        df.loc[('RMSE','day'),varname1[i]] = mean_squared_error(x2,y2)
        df.loc[('N','day'),varname1[i]] = len(x2)
    trunc = lambda x: math.trunc(100 * x) / 100;
    df = df.applymap(trunc)
    df=df.reset_index()
    df.to_csv(filename+'.csv')
    
    writer = MarkdownTableWriter()
    writer.from_dataframe(df)
    writer.write_table()
        # change the output stream to a file
    with open(filename+'.md', "w") as f:
        writer.stream = f
        writer.write_table()
    
#%% 
def day_night_plot(df_all, df_interpol, varname1, varname2, figure_name):
             
    day = df_interpol.sza<70
    night = df_interpol.sza > 110
        
    diff_night = [df_interpol.loc[night,varname1[i]].values \
                  - df_interpol.loc[night,varname2[i]].values for i in range(len(varname1))]
    diff_day = [df_interpol.loc[day,varname1[i]].values \
                  - df_interpol.loc[day,varname2[i]].values for i in range(len(varname1))]
    for i in range(len(varname1)):
        diff_day[i] = diff_day[i][~np.isnan(diff_day[i])]
        diff_night[i] = diff_night[i][~np.isnan(diff_night[i])]
    
    diff_all = [sub[item] for item in range(len(diff_day)) 
                  for sub in [diff_day, diff_night]] 
    for i in range(len(diff_all)):
        if len(diff_all[i])==0:
            diff_all[i] = [0]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    violin_parts = ax.violinplot(diff_all, 
                                 np.arange(1,len(diff_all)+1)/2+0.25,
                                 showmedians=True,
                                 showextrema=False) 
    cmap = mpl.cm.get_cmap('tab20')


    for i, pc in enumerate(violin_parts['bodies']):
        rgba = cmap(i)
        pc.set_facecolor(rgba)
        pc.set_edgecolor(rgba)

    ax.set_xticks(np.arange(1, len(varname1) + 1))
    ax.set_xticklabels(varname1)
    ax.set_ylabel('GC-Net - PROMICE')
    ax.axhline(0, color='black', lw=2, alpha=0.5)
    
    legend_elements = [Patch(facecolor='0.65', edgecolor='k', label='Day'),
                       Patch(facecolor='0.85', edgecolor='k', label='Night')]
    
    ax.legend(handles=legend_elements)

    fig.savefig('./Output/'+figure_name+'.png',bbox_inches='tight', dpi=200)

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
    RH_out = RH.copy()
    RH_out[ind] = RH[ind] * Es_Water[ind]/Es_Ice[ind] 
    return RH_out

def RH_ice2water(RH, T):
    # switch ALL timestep to with-regards-to-water
    RH = np.array(RH)
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    
    # T_100 = 373.15
    # T_0 = 273.15
    # T = T +T_0
    # # GOFF-GRATCH 1945 equation
    #    # saturation vapour pressure above 0 C (hPa)
    # Es_Water = 10**(  -7.90298*(T_100/T - 1) + 5.02808 * np.log(T_100/T) 
    #     - 1.3816E-7 * (10**(11.344*(1-T/T_100))-1) 
    #     + 8.1328E-3*(10**(-3.49149*(T_100/T-1)) -1.) + np.log(1013.246) )
    # # saturation vapour pressure below 0 C (hPa)
    # Es_Ice = 10**(  -9.09718 * (T_0 / T - 1.) - 3.56654 * np.log(T_0 / T) + 
    #              0.876793 * (1. - T / T_0) + np.log(6.1071)  )   
    
    RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

    return RH_out

def RH_ice2water2(RH, T):
    # switch ALL timestep to with-regards-to-water
    RH = np.array(RH)
    # Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    # Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    # Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    ind = T < 0
    # TCoeff = 1/273.15 - 1/(T+273.15)
    # Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    # Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    RH_out = RH.copy()
    
    T_100 = 373.15
    T_0 = 273.15
    T = T +T_0
    # GOFF-GRATCH 1945 equation
        # saturation vapour pressure above 0 C (hPa)
    Es_Water = 10**(  -7.90298*(T_100/T - 1) + 5.02808 * np.log10(T_100/T) 
        - 1.3816E-7 * (10**(11.344*(1-T/T_100))-1) 
        + 8.1328E-3*(10**(-3.49149*(T_100/T-1)) -1.) + np.log10(1013.246) )
    # saturation vapour pressure below 0 C (hPa)
    Es_Ice = 10**(  -9.09718 * (T_0 / T - 1.) - 3.56654 * np.log10(T_0 / T) + 
                  0.876793 * (1. - T / T_0) + np.log10(6.1071)  )   
    
    RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

    return RH_out

# def RH_ice2water3(RH, T):
#     # switch ALL timestep to with-regards-to-water
#     RH = np.array(RH)
#     # Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
#     # Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
#     # Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
#     ind = T < 0
#     # TCoeff = 1/273.15 - 1/(T+273.15)
#     # Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
#     # Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
#     RH_out = RH.copy()
    
#     T_100 = 373.15
#     T_0 = 273.15
#     T = T +T_0
#    # saturation vapour pressure above 0 C (hPa)
#     Es_Water = 10**(  10.79574*(1 - T_100/T) + 5.028 * np.log10(T / T_100)
#                     + 1.50475E-4 * (1 - 10**(-8.2969 * (T/T_100 - 1)))
#                     + 0.42873E-3*(10**(4.76955*(1 - T_100/T)) -1.) +  0.78614 + 2.0 )

#     Es_Ice = 10**( -9.09685 * (T_0 / T - 1.) - 3.56654 * np.log10(T_0 / T) +
#                   0.87682 * (1. - T / T_0) + 0.78614   )
#     RH_out[ind] = RH[ind] / Es_Water[ind]*Es_Ice[ind] 

#     return RH_out

def RH2SpecHum(RH, T, pres):
    # Note: RH[T<0] needs to be with regards to ice
    
    Lv = 2.5001e6  # H2O Vaporization Latent Heat (J/kg)
    Ls = 2.8337e6  # H2O Sublimation Latent Heat (J/kg)
    Rv = 461.5     # H2O Vapor Gaz constant (J/kg/K)
    es = 0.622
    
    TCoeff = 1/273.15 - 1/(T+273.15)
    Es_Water = 6.112*np.exp(Lv/Rv*TCoeff)
    Es_Ice = 6.112*np.exp(Ls/Rv*TCoeff)
    
    es_all = Es_Water.copy()
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
    


