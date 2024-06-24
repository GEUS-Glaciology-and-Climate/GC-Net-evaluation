# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nead
from os import path
from jaws_tools import extrapolate_temp
path_to_L1 = '../GC-Net-level-1-data-processing/'

# %% L1 temperature climatology
plt.close("all")
site_list = pd.read_csv(path_to_L1 + "L1/GC-Net_location.csv", header=0,skipinitialspace=(True)).iloc[1:25].sort_values('ID')

fig, ax = plt.subplots(4, 4, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.9, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1
ABC = 'ABCDEFGHIJKLMNOPQR'
for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/daily/" + site.replace(" ", "") + "_daily.csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR','JAR2','JAR3',
                'NGRIP','Petermann Glacier', 'LAR1', 'LAR2', 'LAR3']:
        # too short for climatology
        continue
    i = i + 1
    ds = nead.read(filename)
    df = ds.to_dataframe()
    df = df.reset_index(drop=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df = df.set_index("timestamp").replace(-999, np.nan)
    df = df[["TA1", "TA2", "TA3", "TA4", "HW1", "HW2"]]
    df.loc[df.TA1.isnull(), "TA1"] = df.loc[df.TA1.isnull(), "TA3"]
    df.loc[df.TA2.isnull(), "TA2"] = df.loc[df.TA2.isnull(), "TA4"]

    df["T2m"] = extrapolate_temp(df).values
    df.loc[df.T2m.isnull(), "T2m"] = df.loc[df.T2m.isnull(), "TA1"]
    df.loc[df.T2m.isnull(), "T2m"] = df.loc[df.T2m.isnull(), "TA2"]
    climatology = df.groupby(df.index.dayofyear).mean()["T2m"]
    climatology_count = df.groupby(df.index.dayofyear).count()["T2m"]
    climatology.loc[climatology_count<5] = np.nan
    # seasonal averages
    msk = (climatology.index < 60) | (climatology.index > 334)
    JFD = climatology.loc[msk].mean()
    msk = (climatology.index >= 60) & (climatology.index < 152)
    MAM = climatology.loc[msk].mean()
    msk = (climatology.index >= 152) & (climatology.index < 244)
    JJA = climatology.loc[msk].mean()
    msk = (climatology.index >= 244) & (climatology.index < 335)
    SON = climatology.loc[msk].mean()
    print(
        "%s, %0.1f, %0.1f, %0.1f, %0.1f, %0.1f"
        % (site, JFD, MAM, JJA, SON, climatology.mean())
    )

    for year in df.index.year.unique():
        # try:
        tmp = df.loc[str(year), :]
        doy = tmp.index.dayofyear.values
        ax[i].plot(doy, tmp["T2m"].values, color="gray", label='_nolegend_', alpha=0.2)
        # except:
        #     pass
    climatology.plot(ax=ax[i], color="k", linewidth=2, label="_nolegend_")
    # plt.legend()
    ax[i].set_title("  "+ABC[i]+". " + site, loc="left", y=1.0, pad=-14)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(-65, 20)
    ax[i].set_xlabel("")
    if i < 12:
        ax[i].set_xticklabels("")
    if i not in [0, 4, 8, 12]:
        ax[i].set_yticklabels("")
    ax[i].grid("on")
fig.text(0.5, 0.02, "Day of year", ha="center", va="center", fontsize=14)
fig.text(
    0.02,
    0.5,
    "Near-surface air temperature ($^o$C)",
    fontsize=14,
    ha="center",
    va="center",
    rotation="vertical",
)
ax[0].plot(np.nan,np.nan, color="gray", label='individual years', alpha=0.2)
ax[0].plot(np.nan,np.nan, color="k", linewidth=2, label="mean")
ax[0].legend(loc='upper center', bbox_to_anchor=(2,1.4), ncol=2)
plt.savefig("out/L1_climatologies/climatology_temperature", bbox_inches="tight")
plt.savefig("out/L1_climatologies/climatology_temperature.pdf", bbox_inches="tight")

# %% L1 wind speed climatology
plt.close("all")

fig, ax = plt.subplots(4, 4, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.9, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1

for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/daily/" + site.replace(" ", "") + "_daily.csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR','JAR2','JAR3',
                'NGRIP','Petermann Glacier', 'LAR1', 'LAR2', 'LAR3']:
        # too short for climatology
        continue
    i = i + 1
    ds = nead.read(filename)
    df = ds.to_dataframe()
    df = df.reset_index(drop=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df = df.set_index("timestamp").replace(-999, np.nan)
    df = df[["VW1", "VW2", "HW1", "HW2"]]

    from jaws_tools import extrapolate_temp

    df["VW10m"] = extrapolate_temp(df, var=["VW1", "VW2"], target_height=10).values
    df.loc[df.VW10m.isnull(), "VW10m"] = df.loc[df.VW10m.isnull(), "VW1"]
    df.loc[df.VW10m.isnull(), "VW10m"] = df.loc[df.VW10m.isnull(), "VW2"]
    climatology = df.groupby(df.index.dayofyear).mean()["VW10m"]
    climatology_count = df.groupby(df.index.dayofyear).count()["VW10m"]
    climatology.loc[climatology_count<5] = np.nan
    
    # seasonal averages
    msk = (climatology.index < 60) | (climatology.index > 334)
    JFD = climatology.loc[msk].mean()
    msk = (climatology.index >= 60) & (climatology.index < 152)
    MAM = climatology.loc[msk].mean()
    msk = (climatology.index >= 152) & (climatology.index < 244)
    JJA = climatology.loc[msk].mean()
    msk = (climatology.index >= 244) & (climatology.index < 335)
    SON = climatology.loc[msk].mean()
    print(
        "%s, %0.1f, %0.1f, %0.1f, %0.1f, %0.1f"
        % (site, JFD, MAM, JJA, SON, climatology.mean())
    )

    for year in df.index.year.unique():
        try:
            tmp = df.loc[str(year), :]
            doy = tmp.index.dayofyear.values
            ax[i].plot(doy, tmp["VW10m"].values, color="gray", label="_nolegend_", alpha=0.2)
        except:
            pass
    climatology.plot(ax=ax[i], color="k", linewidth=2, label="_nolegend_")
    # plt.legend()

    ax[i].set_title("  "+ABC[i]+". " + site, loc="left", y=1.0, pad=-14)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(0, 30)
    ax[i].set_xlabel("")
    if i < 12:
        ax[i].set_xticklabels("")
    if i not in [0, 4, 8, 12]:
        ax[i].set_yticklabels("")
    ax[i].grid("on")
fig.text(0.5, 0.02, "Day of year", ha="center", va="center", fontsize=14)
fig.text(0.02, 0.5,
    "Near-surface wind speed (m s$^{-1}$)",
    fontsize=14, ha="center", va="center",
    rotation="vertical",
)
ax[0].plot(np.nan,np.nan, color="gray", label='individual years', alpha=0.2)
ax[0].plot(np.nan,np.nan, color="k", linewidth=2, label="mean")
ax[0].legend(loc='upper center', bbox_to_anchor=(2,1.4), ncol=2)
plt.savefig("out/L1_climatologies/climatology_windspeed", bbox_inches="tight")
plt.savefig("out/L1_climatologies/climatology_windspeed.pdf", bbox_inches="tight")
