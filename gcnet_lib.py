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
import datetime
from sklearn.metrics import mean_squared_error, r2_score
from pytablewriter import MarkdownTableWriter
import math      
from matplotlib.patches import Patch
import pytz
import nead
from os import listdir
from os.path import isfile, join

def load_ucalg(path_promice='Input/data_DYE2_Samira_hour.txt'):
    df_samira = pd.read_csv(path_promice, delim_whitespace=True)
    df_samira['time'] = df_samira.Year * np.nan
    df_samira['timestamp'] = df_samira.time

    for i, y in enumerate(df_samira.Year.values):
        tmp = datetime.datetime(int(y), 1, 1)   + datetime.timedelta( days = df_samira.DayOfYear.values[i], hours = df_samira.HourOfDayUTC.values[i]-1) 
        df_samira.time[i] = tmp.replace(tzinfo=pytz.UTC)
    #set invalid values (-999) to nan 
    df_samira[df_samira==-999.0]=np.nan
    df_samira = df_samira.set_index('time')
    df_samira['Albedo'] = df_samira['ShortwaveRadiationUpWm2'] / df_samira['ShortwaveRadiationDownWm2']
    df_samira.loc[df_samira['ShortwaveRadiationDownWm2']<100, 'Albedo'] = np.nan
    df_samira['RelativeHumidity_w'] = RH_ice2water(df_samira['RelativeHumidity'] ,
                                                       df_samira['AirTemperatureC'])
    df_samira['SpecHum_ucalg'] = RH2SpecHum(df_samira['RelativeHumidity'] ,
                                                           df_samira['AirTemperatureC'] ,
                                                           df_samira['AirPressurehPa'] )*1000
    df_samira = df_samira.rename(columns={'AirPressurehPa':'p_u',
                                          'AirTemperatureC':'t_u',
                                          'RelativeHumidity_wrtWater':'rh_u',
                                          'WindSpeedms':'wspd_u',
                                          'WindDirectiond':'wdir_u',
                                          'ShortwaveRadiationDownWm2':'dsr_cor',
                                          'ShortwaveRadiationUpWm2':'usr_cor',
                                          'LongwaveRadiationDownWm2':'dlr', 
                                          'LongwaveRadiationUpWm2':'ulr',
                                          'Albedo':'albedo',
                                          'RelativeHumidity_w':'rh_l', 
                                          'SpecHum_ucalg':'qh_u'})
    return df_samira


def load_dmi():
    stat_id = '441900'
    path_to_sec = '../../../Data/AWS/DMI/data/'
    DMI_aliases = {'101': 't_u', 
                   '201': 'rh_u', 
                   '301':'wspd_u',
                   '365':'wdir_u', 
                   '401': 'p_u',
                   '550': 'dsr_cor'}
    df_sec = pd.read_csv(path_to_sec+str(stat_id)+'.csv',sep=';').rename(columns={'Hour(utc)':'Hour'})
    df_sec['timestamp'] = pd.to_datetime(df_sec[['Year','Month','Day','Hour']],utc=True)
    df_sec = df_sec.set_index('timestamp')
    df_sec = df_sec[DMI_aliases.keys()].rename(columns=DMI_aliases)
    df_sec['usr_cor'] = np.nan
    df_sec['ulr'] = np.nan
    df_sec['dlr'] = np.nan
    if 'time' in df_sec.columns:
        df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
        df_sec = df_sec.set_index('time')
    return df_sec.resample('H').mean()


def load_noaa():
    # data source : https://www.esrl.noaa.gov/gmd/dv/data/?site=SUM
    path_dir = 'Input/Summit/'
    file_list = [f for f in listdir(path_dir) if isfile(join(path_dir, f))]
    file_list = [s for s in file_list if "met" in s]
    df_sec = pd.DataFrame()
    for i in range(len(file_list)):
        df = pd.read_csv(path_dir+file_list[i], header=None,  delim_whitespace=True)
        df.columns =['site', 'year', 'month', 'day', 'hour', 'wdir_u', 
                        'wspd_u', 'wsf','p_u', 't_l', 't_u', 'ta_top', 'rh_u', 'rf']
        df_sec = df_sec.append(df)
    df_sec[df_sec==-999.0]=np.nan
    df_sec[df_sec==-999.90]=np.nan
    df_sec[df_sec==-999.99]=np.nan
    df_sec[df_sec==-99.9]=np.nan
    df_sec[df_sec==-99]=np.nan
    df_sec['time'] =pd.to_datetime(df_sec[['year','month','day','hour']], utc = True)
    df_sec = df_sec.set_index('time')
    df_sec[['dsr_cor','usr_cor','ulr','dlr']] = np.nan
    df_sec.loc['2014-09-27':'2017-07-24','rh_u'] = np.nan
    df_sec.loc[df_sec['rh_u']>120, 'rh_u'] = np.nan
    # df_sec['RH_i'] = gnl.RH_water2ice(df_sec.RH.values, df_sec.TA.values)
    return df_sec.resample('H').mean()


def plot_comp(df_gc, df_sec, varname1, varname2, varname3,txt2, figure_name):
    # going through the var names and checking that they are in both datasets
    i_remove = []
    for i in range(len(varname1)):
        if varname1[i] not in df_gc.columns:
            print(varname1[i], 'not in L1 dataset')
            i_remove.append(i)
        if varname2[i] not in df_sec.columns:
            print(varname2[i], 'not in secondary dataset')
            if '_l' in varname2[i]:
                varname2[i] = varname2[i].replace('_l','_u')

            if varname2[i] in df_sec.columns:
                print('only one level available for', txt2)
            else:
                i_remove.append(i)
        
    varname1 = [var for i, var in enumerate(varname1) if i not in i_remove]
    varname2 = [var for i, var in enumerate(varname2) if i not in i_remove]
    varname3 = [var for i, var in enumerate(varname3) if i not in i_remove]            
    
    if len(varname1)>0:     
        fig, ax = plt.subplots(np.size(varname1),2,
                               figsize=(13, 3*np.size(varname1)),
                               gridspec_kw={'width_ratios': [2.5, 1]})
        fig.subplots_adjust(left=0.1, hspace=0.6, wspace=0.05)
        for i in range(np.size(varname1)):
            df1= df_gc.copy()
            df2= df_sec.copy()
            # if varname1[i] in ['Alb','NR']:
            #     df1=df1.resample('D').mean()
            #     df2=df2.resample('D').mean()
            j = 0
            
            ax[i, j].plot(df1.index, df1[varname1[i]], 'b',label= 'GC-Net')
            ax[i, j].plot(df2.index, df2[varname2[i]], 'r', label=txt2, alpha=0.7)
            ax[i, j].legend()
            ax[i, j].set_ylabel(varname3[i])
            ax[i, j].grid()
            
            x=df1[varname1[i]].values
            y =df2[varname2[i]].values
            x2=x[~np.isnan(y)&~np.isnan(x)]
            y2=y[~np.isnan(y)&~np.isnan(x)]
            
            j = 1
            ax[i, j].scatter(df1[varname1[i]],df2[varname2[i]])
            min_val = np.nanmin(np.minimum(df1[varname1[i]],df2[varname2[i]]).values)
            max_val = np.nanmax(np.maximum(df1[varname1[i]],df2[varname2[i]]).values)
            ax[i, j].plot([min_val, max_val], [min_val, max_val], 'k--',linewidth=4)
            ax[i, j].set_title(varname3[i])
            if len(x2)>0:
                ax[i, j].annotate('R2=%.3f \nRMSE=%.2f \nME=%.2f \nN=%.0f' % (r2_score(x2,y2),
                                                              np.sqrt(mean_squared_error(x2,y2)),
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
            ax[i, j].grid()
    
        fig.savefig('./out/L1_vs_other_AWS/'+figure_name+'.png',bbox_inches='tight', dpi=200)
    

 
def tab_comp(df_gc, df_sec, varname1, varname2, filename):
    df = pd.DataFrame(columns=varname1)
    df['metric'] = ['RMSE', 'bias', 'R2', 'N', 'RMSE', 'bias', 'R2', 'N', 'RMSE', 'bias', 'R2', 'N']
    df['time'] = ['all', 'all', 'all', 'all', 'night', 'night', 'night', 'night', 'day',  'day',  'day',  'day']
    df.set_index(['metric','time'],inplace=True)
    
    sza=df_sec['sza']

    day = sza<70
    night = sza > 110
        
    for i in range(np.size(varname1)):
        
        x=df_sec[varname1[i]].values
        y =df_sec[varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        df.loc[('R2','all'),varname1[i]] = r2_score(x2,y2)
        df.loc[('bias','all'),varname1[i]] = np.mean(x2-y2)
        df.loc[('RMSE','all'),varname1[i]] = mean_squared_error(x2,y2)
        df.loc[('N','all'),varname1[i]] = len(x2)
             

        x=df_sec.loc[night,varname1[i]].values
        y =df_sec.loc[night,varname2[i]].values
        x2=x[~np.isnan(y)&~np.isnan(x)]
        y2=y[~np.isnan(y)&~np.isnan(x)]
        
        df.loc[('R2','night'),varname1[i]] = r2_score(x2,y2)
        df.loc[('bias','night'),varname1[i]] = np.mean(x2-y2)
        df.loc[('RMSE','night'),varname1[i]] = mean_squared_error(x2,y2)
        df.loc[('N','night'),varname1[i]] = len(x2)
        
        x = df_sec.loc[day,varname1[i]].values
        y = df_sec.loc[day,varname2[i]].values
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
    


