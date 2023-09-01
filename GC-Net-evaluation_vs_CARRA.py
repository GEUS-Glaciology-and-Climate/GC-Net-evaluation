# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
from rasterio.crs import CRS
import xarray as xr

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gcnet_lib as gnl
from os import path
import nead
import xarray as xr

# loading sites metadata
site_list = pd.read_csv('C:/Users/bav/OneDrive - Geological survey of Denmark and Greenland/Code/PROMICE/GC-Net-Level-1-data-processing/metadata/GC-Net_location.csv', skipinitialspace=True) 
     
df_metadata = pd.read_csv('C:/data_save/CARRA/AWS_station_locations.csv')

name_aliases = {
                'si10': ['VW1','VW2'],
                'r2': ['RH1_cor','RH2_cor'],
                't2m': ['TA1','TA2','TA3','TA4'],
                'sp':'P',
                'ssrd':'ISWR',
                'dsr_cor':'ssrd',
                'usr_cor':'ssru',
                'dlr':'strd',
                'net_sr':'net_ssrd',
                }
plt.close('all')

var_carra = 'r2'
site = 'Humboldt'
year=2011

# var_carra = 'si10'
print(var_carra, site, year)
var_gcn = name_aliases[var_carra]

# if var_promice == 'net_sr':
#     CARRA_nearest_point = xr.open_dataset('CARRA_ssrd_nearest_PROMICE.nc')
#     al = xr.open_dataset('CARRA_al_nearest_PROMICE.nc')
#     CARRA_nearest_point['net_ssrd']= CARRA_nearest_point.ssrd - CARRA_nearest_point.ssrd*al.al/100
# elif var_promice == 'usr_cor':
#     CARRA_nearest_point = xr.open_dataset('CARRA_ssrd_nearest_PROMICE.nc')
#     al = xr.open_dataset('CARRA_al_nearest_PROMICE.nc')
#     CARRA_nearest_point['ssru']=  CARRA_nearest_point.ssrd*al.al/100
# else:
CARRA_nearest_point = xr.open_dataset('C:/data_save/CARRA/CARRA_'+var_carra+'_nearest_PROMICE.nc')

# units
if var_carra == 't2m': CARRA_nearest_point['t2m'] = CARRA_nearest_point['t2m']-273.15
if var_carra == 'sp': CARRA_nearest_point['sp'] = CARRA_nearest_point['sp']/100
# if var_carra == 't2m': CARRA_nearest_point['r2'] = gnl.RH_ice2water(RH, CARdRA_nearest_point['t2m']) 
# if var_promice == 't_surf': CARRA_nearest_point['skt'] = CARRA_nearest_point['skt']-273.15
# if var_promice == 'albedo': CARRA_nearest_point['al'] = CARRA_nearest_point['al']/100
if var_carra in ['ssrd', 'usr_cor','dlr','ulr', 'net_sr']:
    CARRA_nearest_point[var_carra] = CARRA_nearest_point[var_carra] / (3 * 3600)
   
# Loading carra data
ds_carra = CARRA_nearest_point.sel(station=df_metadata.loc[df_metadata.stid==site].index[0])

ID = site_list.loc[site_list.Name == site, 'ID'].values[0]

filename = '../GC-Net-Level-1-data-processing/L1/'+str(ID).zfill(2)+'-'+site.replace(' ','')+'.csv'
if not path.exists(filename):
    print('Warning: No file for station '+str(ID)+' '+site)

# loading L1 AWS file
ds = nead.read(filename)
df = ds.to_dataframe()
df=df.reset_index(drop=True)
df.timestamp = pd.to_datetime(df.timestamp)
df = df.set_index('timestamp').replace(-999,np.nan)

# selecting data in carra dataset and interpolating to hourly values
df_carra = ds_carra.to_dataframe()

# plotting before adjustments
fig,ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(15,8))
ax=[ax]

df_carra.loc[str(year), var_carra].plot(ax=ax[0], label='carra', alpha = 0.8)
df.loc[str(year), var_gcn].plot(ax=ax[0], label='GC-Net', alpha = 0.8)

plt.title(site+' '+var_carra+' '+str(year)) #+' before adjustment')
ax[0].set_ylabel(var_carra)
ax[0].legend()


