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
from pandas.tseries.offsets import DateOffset
from os import path
import nead
import xarray as xr


def xcorr(x, y, normed=True, maxlags=30):
    x[np.isnan(x)]=np.nanmean(x)
    y[np.isnan(y)]=np.nanmean(y)
    Nx = len(x)
    if Nx != len(y): raise ValueError('x and y must be equal length')
    c = np.correlate(x, y, mode=2)
    if normed: c /= np.sqrt(np.dot(x, x) * np.dot(y, y))
    if maxlags is None: maxlags = Nx - 1
    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]

    return lags, c


def max_xcorr(x, y, maxlag=30):
    if (np.isnan(x).all())|(np.isnan(y).all()):
        return np.nan
    try:
        lagsOut, corrCoef = xcorr(x, y)
    except:
        print('ERROR: ', x, y)
        return np.nan
    if np.isnan(corrCoef).all():
        return np.nan
    return lagsOut[np.argmax(corrCoef)]


def shift_data(df, t0, t1, shift, clean_up_type = 'all data before target'):
    var_list_rm = df.columns
    t0 = pd.to_datetime(t0)
    t1 = pd.to_datetime(t1)
    df_out = df.copy()
    df_out.loc[t0+pd.Timedelta(hours=shift): t1+pd.Timedelta(hours=shift), :] = df_out.loc[t0:t1, :].values
    if clean_up_type == 'all data before target':
        if shift>0:
            df_out.loc[t0:t0+pd.Timedelta(hours=shift), var_list_rm] = np.nan
        else:
            df_out.loc[t1-pd.Timedelta(hours=shift):t1, var_list_rm] = np.nan
    if clean_up_type == 'data within source period':
        df_out.loc[t0:t1, var_list_rm] = np.nan

    return df_out


# plt.close('all')

# loading sites metadata
site_list = pd.read_csv('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/GC-Net-Level-1-data-processing/metadata/GC-Net_location.csv',header=0)
# uncomment for use at specific sites
# All station names: 'Swiss Camp 10m', 'Swiss Camp', 'Crawford Point 1', 'NASA-U',
       # 'GITS', 'Humboldt', 'Summit', 'Tunu-N', 'DYE2', 'JAR1', 'Saddle',
       # 'South Dome', 'NASA-E', 'CP2', 'NGRIP', 'NASA-SE', 'KAR', 'JAR 2',
       # 'KULU', 'Petermann ELA', 'NEEM', 'E-GRIP'
site_list = site_list.loc[site_list.Name.values == 'GITS',:]


name_aliases = dict(zip(site_list.Name, site_list.Name))
name_aliases.update({'Swiss Camp': 'SwissCamp','Swiss Camp 10m': 'SwissCamp', 'Crawford Point 1': 'CrawfordPt.',
                'Tunu-N':'TUNU-N', 'DYE2': 'DYE-2', 'JAR1':'JAR',
                'South Dome': 'SouthDome', 'JAR 2': 'JAR2'})

# choosing variable to evaluate
var1 = 'TA1'
var2 = 'tas2m'
# var1 = 'P'
# var2 = 'ps'
# var1 = 'RH1'
# var2 = 'relhum2m'
# var1 = 'ISWR'
# var2 = 'dswrad'

# Loading RACMO data
ds_racmo = xr.open_dataset('../../../Data/RCM/RACMO/Data_AWS_RACMO2.3p2_FGRN055_1957-2017/RACMO_3h_AWS_sites.nc')
ds_racmo['station_name'] = ('Station',
                            pd.read_csv('../../../Data/RCM/RACMO/Data_AWS_RACMO2.3p2_FGRN055_1957-2017/AWS.csv',sep=';')['Station name'].values)

for site, ID in zip(site_list.Name,site_list.ID):
    print('# '+str(ID)+ ' ' + site)
    print(var1)

    filename = '../GC-Net-Level-1-data-processing/L1/'+str(ID).zfill(2)+'-'+site+'.csv'
    if not path.exists(filename):
        print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if name_aliases[site] not in ds_racmo.station_name:
        continue
    
    # loading L1 AWS file
    ds = nead.read(filename)
    df = ds.to_dataframe()
    df=df.reset_index(drop=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df = df.set_index('timestamp').replace(-999,np.nan)
    
    # selecting data in RACMO dataset and interpolating to hourly values
    station_num = ds_racmo.Station[ds_racmo.station_name == name_aliases[site]]
    df_racmo = ds_racmo.sel(Station = station_num).squeeze().to_dataframe()
    df_racmo.index = pd.to_datetime(df_racmo.index, utc=True)
    df_racmo = df_racmo.loc[(df.index[0]-pd.Timedelta('1D')):,:]
    df = df.loc[:df_racmo.index[-1],:]
    df_racmo_interp = df_racmo.resample('H').interpolate(method='pchip')
    df_racmo_interp = df_racmo_interp.loc[df.index[0]:,:]
    df[var1+'_racmo'] = df_racmo_interp[var2]

    # calculating the monthly best lag between the AWS data and the RACMO data
    # df_lag = df[[var1, var1+'_racmo']].resample('M').apply(lambda x: max_xcorr(x[var1].values, x[var1+'_racmo'].values))
    
    # plotting before adjustments
    fig,ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(15,8))
    ax=[ax]
    df[var1].plot(ax=ax[0], label='GC-Net')
    # df_racmo[var2].plot(ax=ax, marker='o',label='RACMO')
    df_racmo_interp[var2].plot(ax=ax[0], label='RACMO')
    # ax1=ax[0].twinx()
    # df_lag.plot(ax=ax1, drawstyle="steps", color='k')
    plt.title(site) #+' before adjustment')
    # ax1.set_ylabel('Lag maximizing the correlation \nbetween GC-Net and RACMO (hours)')
    ax[0].set_ylabel(var1)
    ax[0].legend()

    # print('Months with non-zero best lags:',df_lag.loc[(df_lag!=0)&(df_lag.notnull())])
    
    # adjusting
    # if site == 'JAR1':
    #     df = shift_data(df, '2003-04-24', '2005-05-07', 24)
        
    # if site == 'South Dome':
    #     df = shift_data(df, '1998-04-20T03:00:00', '1999-04-23T03:00:00', -3)
               
    # if site == 'Humboldt':
    #     df = shift_data(df, '2004-08-08', '2005-01-01', -24)
    #     df = shift_data(df, '2005-01-02', '2006-05-04', -48)
    #     df = shift_data(df, '2016-12-01T00', '2017-10-03T00', 2943)
    #     df = shift_data(df, '2015-01-01T00', '2015-02-18T06', 2198)
    #     df = shift_data(df, '2017-12-10T00', '2018-02-02T15', 2691)
    #     df = shift_data(df, '2018-12-06T00', '2019-01-18T13', 2954)

    # if site == 'GITS':
    #     df = shift_data(df, '2019-03-14T00', '2019-04-29T01', 520)
    # if site == 'NASA-U':
    #     df = shift_data(df, '2010-03-20', '2010-10-11', 48)

    # if site == 'Crawford Point 1':
    #     df = shift_data(df, '1990-01-01 16:00:00','1990-09-26 14:00:00', 180552, clean_up_type = 'data within source period')
    #     df = shift_data(df, '1999-08-09', '2000-06-04T06', 24)
    #     df = shift_data(df, '2003-04-19T14', '2004-06-09', 24)
    #     df = shift_data(df, '2008-06-12T00', '2009-04-27', 24)
    
    # df[var1+'_racmo'] = df_racmo_interp[var2]
    # df_lag = df[[var1, var1+'_racmo']].resample('M').apply(lambda x: max_xcorr(x[var1].values, x[var1+'_racmo'].values))
    
    # plotting adjuted values
    # df[var1].plot(ax=ax[1], label='GC-Net')
    # df_racmo_interp[var2].plot(ax=ax[1], label='RACMO')
    # ax[1].set_ylabel(var1)
    # ax[1].legend()
    # ax3=ax[1].twinx()
    # df_lag.plot(ax=ax3, drawstyle="steps", color='k')
    # plt.title(site+' before adjustment')
    # ax1.set_ylabel('Lag maximizing the correlation \nbetween GC-Net and RACMO (hours)')
