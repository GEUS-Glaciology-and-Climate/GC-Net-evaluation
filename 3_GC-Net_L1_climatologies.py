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
path_to_L1 = '../GC-Net-Level-1-data-processing/'

# %% L1 temperature climatology
plt.close("all")
site_list = pd.read_csv(path_to_L1 + "metadata/GC-Net_location.csv", header=0,skipinitialspace=(True)).iloc[1:]

fig, ax = plt.subplots(4, 5, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.95, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1

for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/" + str(ID).zfill(2) + "-" + site.replace(" ", "") + ".csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR', 'LAR1', 'LAR2', 'LAR3']:
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
    df.loc[df.TA1.isnull(), "TA2"] = df.loc[df.TA1.isnull(), "TA4"]

    

    df["T2m"] = extrapolate_temp(df).values
    df["T"] = df["T2m"]
    df.loc[df.T2m.isnull(), "T"] = df.loc[df.T2m.isnull(), "TA1"]
    df.loc[df.T2m.isnull(), "T"] = df.loc[df.T2m.isnull(), "TA2"]
    climatology = df.groupby(df.index.dayofyear).mean()["T"]

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
            tmp = df.loc[str(year), :].resample("D").mean()
            doy = tmp.index.dayofyear.values
            ax[i].plot(doy, tmp["T"].values, color="gray", label=str(year), alpha=0.2)
        except:
            pass
    climatology.plot(ax=ax[i], color="k", linewidth=3, label="average")
    # plt.legend()
    ax[i].set_title(" " + site, loc="left", y=1.0, pad=-14)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(-65, 15)
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
plt.savefig("out/L1_climatologies/climatology_temperature", bbox_inches="tight")

# %% L1 humidity climatology
plt.close("all")

fig, ax = plt.subplots(4, 5, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.95, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1

for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/" + str(ID).zfill(2) + "-" + site.replace(" ", "") + ".csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR', 'LAR1', 'LAR2', 'LAR3']:
        # too short for climatology
        continue
    i = i + 1
    ds = nead.read(filename)
    df = ds.to_dataframe()
    df = df.reset_index(drop=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df = df.set_index("timestamp").replace(-999, np.nan)
    df = df[["RH1", "RH2", "HW1", "HW2"]]
    df["RH2m"] = extrapolate_temp(df, var=["RH1", "RH2"]).values
    df["RH"] = df["RH2m"]
    df.loc[df.RH2m.isnull(), "RH"] = df.loc[df.RH2m.isnull(), "RH1"]
    df.loc[df.RH2m.isnull(), "RH"] = df.loc[df.RH2m.isnull(), "RH2"]
    climatology = df.groupby(df.index.dayofyear).mean()["RH"]

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
            tmp = df.loc[str(year), :].resample("D").mean()
            doy = tmp.index.dayofyear.values
            ax[i].plot(doy, tmp["RH"].values, color="gray", label=str(year), alpha=0.2)
        except:
            pass
    climatology.plot(ax=ax[i], color="k", linewidth=3, label="average")
    # plt.legend()
    ax[i].set_title(" " + site, loc="left", y=0.0, pad=5)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(30, 100)
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
    "Near-surface relative humidity (%)",
    fontsize=14,
    ha="center",
    va="center",
    rotation="vertical",
)
plt.savefig("out/L1_climatologies/climatology_humidity", bbox_inches="tight")

# %% L1 pressure climatology
plt.close("all")

fig, ax = plt.subplots(4, 5, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.95, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1

for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/" + str(ID).zfill(2) + "-" + site.replace(" ", "") + ".csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR', 'LAR1', 'LAR2', 'LAR3']:
        # too short for climatology
        continue
    i = i + 1
    ds = nead.read(filename)
    df = ds.to_dataframe()
    df = df.reset_index(drop=True)
    df.timestamp = pd.to_datetime(df.timestamp)
    df = df.set_index("timestamp").replace(-999, np.nan)
    df = df[["P"]]

    climatology = df.groupby(df.index.dayofyear).mean()["P"]

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
            tmp = df.loc[str(year), :].resample("D").mean()
            doy = tmp.index.dayofyear.values
            ax[i].plot(doy, tmp["P"].values, color="gray", label=str(year), alpha=0.2)
        except:
            pass
    climatology.plot(ax=ax[i], color="k", linewidth=3, label="average")
    # plt.legend()
    if site == "Summit":
        ax[i].set_title(" " + site, loc="left", y=1.0, pad=-14)
    else:
        ax[i].set_title(" " + site, loc="left", y=0.0, pad=5)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(600, 950)
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
    "Near-surface air pressure (hPa)",
    fontsize=14,
    ha="center",
    va="center",
    rotation="vertical",
)
plt.savefig("out/L1_climatologies/climatology_pressure", bbox_inches="tight")

# %% L1 wind speed climatology
plt.close("all")

fig, ax = plt.subplots(4, 5, figsize=(8, 6))
plt.subplots_adjust(
    left=0.08, right=0.99, bottom=0.09, top=0.95, hspace=0.15, wspace=0.03
)
ax = ax.flatten()
i = -1

for site, ID in zip(site_list.Name, site_list.ID):
    filename = path_to_L1+"L1/" + str(ID).zfill(2) + "-" + site.replace(" ", "") + ".csv"
    if not path.exists(filename):
        # print('Warning: No file for station '+str(ID)+' '+site)
        continue
    if site in ['CP2','Aurora','KULU','KAR', 'LAR1', 'LAR2', 'LAR3']:
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

    df["T2m"] = extrapolate_temp(df, var=["VW1", "VW2"], target_height=10).values
    df["T"] = df["T2m"]
    df.loc[df.T2m.isnull(), "T"] = df.loc[df.T2m.isnull(), "VW1"]
    df.loc[df.T2m.isnull(), "T"] = df.loc[df.T2m.isnull(), "VW2"]
    climatology = df.groupby(df.index.dayofyear).mean()["T"]

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
            tmp = df.loc[str(year), :].resample("D").mean()
            doy = tmp.index.dayofyear.values
            ax[i].plot(doy, tmp["T"].values, color="gray", label=str(year), alpha=0.2)
        except:
            pass
    climatology.plot(ax=ax[i], color="k", linewidth=3, label="average")
    # plt.legend()

    ax[i].set_title(" " + site, loc="left", y=1.0, pad=-14)
    ax[i].set_xlim(0, 365)
    ax[i].set_ylim(0, 30)
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
    "Near-surface wind speed (m s$^{-1}$)",
    fontsize=14,
    ha="center",
    va="center",
    rotation="vertical",
)
plt.savefig("out/L1_climatologies/climatology_windspeed", bbox_inches="tight")
