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
import xarray as xr
from sklearn.linear_model import LinearRegression
import sys

path_to_L1 = '../GC-Net-Level-1-data-processing/'

# Comparison with JoG dataset
gcnet_path = "C:/Data_save/Data JoG 2020/Corrected/data out/"
site_list = [
    "CP1",
    "DYE_2",
    "NASA_SE",
    "NASA_E",
    "NASA_U",
    "Saddle",
    "TUNU_N",
    "Summit",
    "SouthDome",
]
site_list = sorted(site_list)

fig, ax = plt.subplots(3, 3)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.3)
for i, folder in enumerate(site_list):

    site = site_list[i]
    sys.stdout.flush()
    ds = xr.open_dataset(gcnet_path + "/" + site + "_surface.nc")
    df = ds.to_dataframe().resample("D").mean()
    i, j = np.unravel_index(i, np.shape(ax))

    trend = np.array([])
    for year in range(1998, 2018):
        df2 = df.loc[df.index.year == year, :]

        X = df2.index.dayofyear.values
        Y = df2.H_surf_mod.values - np.nanmean(df2.H_surf_mod.values)
        mask = ~np.isnan(X) & ~np.isnan(Y)
        lm = LinearRegression()
        if ~np.any(mask):
            continue
        lm.fit(X[mask].reshape(-1, 1), Y[mask].reshape(-1, 1))
        if lm.coef_[0][0] < 0:
            continue
        Y_pred = lm.predict(X[mask].reshape(-1, 1))

        ax[i, j].plot(X, Y)
        ax[i, j].plot(X[mask].reshape(-1, 1), Y_pred, linestyle="--")
        ax[i, j].set_xlabel("")
        ax[i, j].set_title(site)
        trend = np.append(trend, lm.coef_[0][0] * 365)
    trend_avg = np.nanmean(trend[trend > 0])
    trend_std = np.nanstd(trend[trend > 0])
    ax[i, j].text(
        100,
        ax[i, j].get_ylim()[1] * 0.8,
        " "
        + str(round(trend_avg, 2))
        + " $\pm$ "
        + str(round(trend_std, 2))
        + " $m yr^{-1}$",
    )
    print(
        site
        + " "
        + str(round(trend_avg, 2))
        + " +/- "
        + str(round(trend_std, 2))
        + " m yr-1"
    )
    # print(lm.intercept_)

ax[1, 0].set_ylabel("Surface height (m)")
ax[2, 1].set_xlabel("Day of year")