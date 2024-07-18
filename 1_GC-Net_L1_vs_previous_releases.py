# -*- coding: utf-8 -*-
"""
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
# from windrose import WindroseAxes
import nead
np.seterr(invalid='ignore')
import warnings
warnings.filterwarnings("ignore")
import xarray as xr
import tocgen

jaws_alias = {'RH1':'rh1','RH2':'rh2','TA1':'ta_tc2','TA2':'ta_tc2','P':'ps',
              'SZA':'zenith_angle', 'ISWR':'fsds', 'OSWR':'fsus',
              'TA3':'ta_cs1','TA4':'ta_cs2','VW1':'wspd1','VW2':'wspd2',
              'DW1':'wdir1','DW2':'wdir1','HS1':'snh1', 'HS2':'snh2'}

# %% Comparing different file versions
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0)
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'

f = open("out/L1_vs_historical_files/report.md", "w")

for site, ID in zip(site_list.Name,site_list.ID):
    print(site)
    plt.close('all')
    site = site.replace(' ','')
    f.write('\n# '+str(ID)+ ' ' + site)

    
    path_to_hist_data = './Data/20190501_jaws/'
    try:
        df_hist_jaws = xr.open_dataset(path_to_hist_data+'%0.2ic.dat_Req1957.nc'%ID).sel(nbnd=1).squeeze().to_dataframe()
    except:
        f.write('\nno historical file to compare')
        continue

    df_L1 = nead.read(path_to_L1 + "/hourly/" + site + ".csv").to_dataframe()
    df_L1.timestamp = pd.to_datetime(df_L1.timestamp)
    df_L1 = df_L1.set_index('timestamp')
    df_L1[df_L1==-999] = np.nan
    
    df_hist_jaws.index = df_hist_jaws.index + pd.Timedelta(minutes=30)
    df_hist_jaws.index = pd.to_datetime(df_hist_jaws.index, utc=True)
    df_hist_jaws[[ 'ta_tc1', 'ta_tc2', 'ta_cs1', 'ta_cs2']] = df_hist_jaws[[ 'ta_tc1', 'ta_tc2', 'ta_cs1', 'ta_cs2']] -273.15
    df_hist_jaws.ps = df_hist_jaws.ps/100
    
    fig, ax = plt.subplots(3,1, figsize=(10,10),sharex=True)
    plt.subplots_adjust(top=0.9)
    for i, var in enumerate(['TA1','TA2','P']):
    # for i, var in enumerate(['ISWR','OSWR', 'SZA']):
        df_L1[var].plot(ax=ax[i], label = 'reprocessed by GEUS')
        df_hist_jaws[jaws_alias[var]].plot(ax=ax[i], label = 'last provided by K. Steffen', alpha=0.7)
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.7, 1.02))
    plt.suptitle(site)
    fig.savefig('out/L1_vs_historical_files/'+site.replace(' ','')+'_1.png')
    f.write('\n![]('+site+'_1.png)')
    
    # fig, ax = plt.subplots(8,1, figsize=(10,10))
    # plt.subplots_adjust(top=0.95)
    # for i, var in enumerate(['TA3','TA4','VW1','VW2','DW1','DW2','HS1', 'HS2']):
    #     df_L1[var].plot(ax=ax[i], label = 'L1')
    #     df_hist_jaws[jaws_alias[var]].plot(ax=ax[i], label = 'hist', alpha=0.7)
    #     ax[i].set_ylabel(var)
    #     if i<len(ax)-1:
    #         ax[i].xaxis.set_ticklabels([])
    # plt.legend()
    # plt.suptitle(site)
    # fig.savefig('out/L1_vs_historical_files/'+site+'_2.png')
    # f.write('\n![](out/L1_vs_historical_files/'+site+'_2.png)')

f.close()

tocgen.processFile('out/L1_vs_historical_files/report.md','out/L1_vs_historical_files/report_toc.md')

# %% Comparing different file versions
site_list = pd.read_csv('Input/GC-Net_location.csv',header=0,skipinitialspace=True)[-3:]
path_to_L1 =  '../GC-Net-Level-1-data-processing/L1/'
meta = pd.read_csv('Input/GC-Net LAR stations/C-Level Format.csv',skipinitialspace=True)
aliases = {'LAR'+str(i): 'Larsen'+str(i) for i in range(1,4)}
plt.close('all')

for site, ID in zip(site_list.Name,site_list.ID):
    print(site)

    df_L1 = nead.read(path_to_L1 + '%0.2i-%s.csv'%(ID, site)).to_dataframe()
    df_L1.timestamp = pd.to_datetime(df_L1.timestamp,utc=True)
    df_L1 = df_L1.set_index('timestamp')
    df_L1[df_L1==-999] = np.nan
    
    path_to_hist_data = 'Input/GC-Net LAR stations/'
    df_lar = pd.read_csv(path_to_hist_data+aliases[site]+'.csv', header=None)
    df_lar.columns = meta.Parameter
    df_lar['hour'] =np.round(( df_lar.doy - np.trunc(df_lar.doy))*24)
    df_lar.doy = np.trunc(df_lar.doy)
    df_lar['timestamp'] = pd.to_datetime(df_lar.Year*100000+df_lar.doy*100+df_lar.hour, format='%Y%j%H',utc=True)
    df_lar = df_lar.set_index('timestamp')

    fig, ax = plt.subplots(8,1, figsize=(10,20),sharex=True)
    plt.subplots_adjust(top=0.95)
    for i, var in enumerate(['RH1_cor','RH2_cor','TA1','TA2','P','ISWR','OSWR', 'Alb']):
    # for i, var in enumerate(['ISWR','OSWR', 'SZA']):
        df_L1[var].plot(ax=ax[i], label = 'L1')
        df_lar[var].plot(ax=ax[i], label = 'hist', alpha=0.7)
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])
        if var in  ['ISWR', 'OSWR']:
            print(site, var, (df_lar[var]/df_L1[var]).median())
    plt.legend()
    plt.suptitle(site)
    fig.savefig('out/L1_vs_historical_files/'+site.replace(' ','')+'_1.png')
    
    fig, ax = plt.subplots(8,1, figsize=(10,20))
    plt.subplots_adjust(top=0.95)
    for i, var in enumerate(['TA3','TA4','VW1','VW2','DW1','DW2','HS1', 'HS2']):
        df_L1[var].plot(ax=ax[i], label = 'L1')
        df_lar[var].plot(ax=ax[i], label = 'hist', alpha=0.7)
        ax[i].set_ylabel(var)
        if i<len(ax)-1:
            ax[i].xaxis.set_ticklabels([])
    plt.legend()
    plt.suptitle(site)
    fig.savefig('out/L1_vs_historical_files/'+site+'_2.png')
    
