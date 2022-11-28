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
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore")

sec_stations = {'SwissCamp': ['SWC'], 
                'NASA-U': ['NAU'],
                'NASA-E': ['NAE'], # not used because NAE has only intermittent data
                'GITS': ['CEN', 'CEN2'],
                'Tunu-N': ['TUN'],
                'NEEM': ['NEM'],
                'E-GRIP': ['EGP'],
                'Saddle': ['SDL'],
                'SouthDome': ['SDM'],
                'NASA-SE': ['NSE'],
                'DYE2': ['U. Calg.', 'DY2'],
                'Summit': ['DMI', 'NOAA']}

# Comparing to GEUS, U.Calg. and NOAA AWS
site_list = pd.read_csv('Input/GC-Net_location.csv', header=0, skipinitialspace=True)
path_to_L1 = '../GC-Net-Level-1-data-processing/L1/'
plt.close('all')
f = open("out/L1_vs_other_AWS/report.md", "w")
# 'Swiss Camp', 'NASA-U','GITS','NEEM','E-GRIP','Saddle', 'Summit','DYE2',

for site in ['E-GRIP']: #'Swiss Camp', 'NASA-U',  'GITS', 'NEEM','E-GRIP','Saddle', 'Summit', 'DYE2']:
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
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
            
        elif os.path.exists('Input/data_'+site+'_Samira_hour.txt'):
            print('reading data from U. Calg. AWS')
            df_sec = gnl.load_ucalg('Input/data_'+site+'_Samira_hour.txt')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
            
        if os.path.exists('Input/GEUS stations/'+name_sec+'_hour.csv'):
            print('found one-boom AWS data')
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')
            
        if os.path.exists('./Input/GEUS stations/'+name_sec+'_hour_v04.csv'):
            print('found one-boom AWS data')
            df_sec = pd.read_csv('Input/GEUS stations/'+name_sec+'_hour_v04.csv')
            df_sec['time'] = pd.to_datetime(df_sec.time, utc=True)
            df_sec = df_sec.set_index('time')
            df_sec.index = df_sec.index + pd.Timedelta('1H')

        if name_sec == 'DMI':
            print('loading DMI station')
            df_sec = gnl.load_dmi()
        if name_sec == 'NOAA':
            print('loading NOAA station')
            df_sec = gnl.load_noaa()        

        if name_sec == 'SWC':
            df_sec['wdir_u'] = (df_sec['wdir_u'] -160) % 360
            df_gc.index = df_gc.index +pd.Timedelta('3H')
        df_gc['DW1'] = df_gc['DW1']%360
        df_gc['DW2'] = df_gc['DW2']%360
        # selecting overlapping data
        try: 
            df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
            df_sec = df_sec.loc[slice(df_gc.index[0], df_gc.index[-1]),:]
            df_gc = df_gc.loc[slice(df_sec.index[0], df_sec.index[-1]),:]
        except:
            pass

        df_sec['net_rad'] = df_sec.dsr_cor-df_sec.usr_cor+df_sec.dlr-df_sec.ulr
        # plotting
        varname1 =  [ 'ISWR','OSWR', 'Alb','NR']
        varname2 =  [ 'dsr_cor',  'usr_cor','albedo','net_rad']
        varname3 = ['SWdown (W/m2)', 'SWup (W/m2)', 'Albedo (-)','NetRad (W/m2)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2, varname3, name_sec, name_sec+'_1')
        
        varname1 =  ['TA1','TA2', 'P']
        varname2 =  ['t_l', 't_u', 'p_u']
        varname3 =  ['Air temperature 1 (deg C)', 'Air temperature 2 (deg C)', 'Air pressure (hPa)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, name_sec+'_2')
        
        varname1 =  ['RH1','RH2','SH1','SH2']
        varname2 =  ['rh_u','rh_l','qh_l', 'qh_u']
        varname3 =  ['Relative Humidity 1 (%)', 'Relative Humidity 2 (%)', 
                     'Specific humidity 1 (g/kg)','Specific humidity 2 (g/kg)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, name_sec+'_3')
        
        varname1 =  ['VW1','VW2','DW1','DW2']
        varname2 =  [ 'wspd_l', 'wspd_u','wdir_l','wdir_u']
        varname3 =  [ 'Wind speed 1 (m/s)', 'Wind speed 2 (m/s)','Wind direction 1 (d)','Wind direction 2 (d)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, name_sec+'_4')
        
        varname1 =  ['LHF', 'SHF']
        varname2 =  [ 'dlhf_u', 'dshf_u']
        varname3 =  [ 'Latent heat flux (W m-2)', 'Sensible heat flux (W m-2)']
        gnl.plot_comp(df_gc, df_sec, varname1, varname2,varname3, name_sec, name_sec+'_5')

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
        del df_sec

    for i in range(1,6):
        f.write('\n![](out/L1_vs_other_AWS/'+name_sec+'_'+str(i)+'.png)')

f.close()

tocgen.processFile('out/L1_vs_other_AWS/report.md','out/L1_vs_other_AWS/report_toc.md')
