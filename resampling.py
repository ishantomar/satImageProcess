# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:15:55 2023

@author: HP
"""
import rasterio
from rasterio.enums import Resampling

xres, yres = 20,20

with rasterio.open(r'D:\10007\research\crop_classification\2017-01-01\S2A_MSIL1C_20170101T082332_N0204_R121_T34JEP_20170101T084543.SAFE\GRANULE\L1C_T34JEP_A007983_20170101T084543\IMG_DATA\red.jp2') as dataset:
    scale_factor_x = dataset.res[0]/xres
    scale_factor_y = dataset.res[1]/yres

    profile = dataset.profile.copy()
    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * scale_factor_y),
            int(dataset.width * scale_factor_x)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (1 / scale_factor_x),
        (1 / scale_factor_y)
    )
    profile.update({"height": data.shape[-2],
                    "width": data.shape[-1],
                   "transform": transform})

with rasterio.open("red.tif", "w", **profile) as dataset:
    dataset.write(data)
   