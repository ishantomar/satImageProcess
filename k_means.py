# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 15:45:04 2023

@author: HP
"""

#%%
import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import rasterio
import os
#%%K-means with one image
raster_path = input(r"enter raster image path: ")
output_path = input(r"enter output path: ")

with rasterio.open(raster_path) as src:    
    img = src.read()
    img_shape = img.shape
    img_dtype = img.dtype
    
    
    X = img.reshape(img_shape[0], -1).T
    
    n_clusters = 10
    k_means = cluster.KMeans(n_clusters)
    k_means.fit(X)
    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img_shape[1:])

with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=img_shape[1],
    width=img_shape[2],
    count=1,
    dtype=img_dtype,
    crs=src.crs,
    transform=src.transform
) as dst:
    
    dst.write(X_cluster, 1)

#%%