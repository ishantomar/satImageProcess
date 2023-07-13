# -*- coding: utf-8 -*-


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import os
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pathlib

class imageprocess:
    def __init__(self):
        self.vectorPath = "/Users/ishantomar/Azista/Test/vector_data/polypoint.shp"
        self.rasterPath = "/Users/ishantomar/Azista/Test/raster_data"
    
    def getNDVI(self, save=True, folderPath=None):
        folderPath = self.rasterPath
        for root, folder, files in os.walk(folderPath):
            # print("Root Folder")
            # print(root)
            # print("Folder")
            # print(folder)
            print(files)

if __name__=="__main__":
    x = imageprocess()  
    test = pathlib.Path(x.rasterPath)
    testList = list(test.iterdir())
    fpath = []
    img_path = []
    g = "GRANULE"
    for i in testList:
        j = pathlib.Path(i)
        try :
            k = list(j.iterdir())
            #print(k)
        except:
            testList.remove(i)
    # print("Value of fpath")
    # print(fpath)
    for l in fpath:
        print(list(l)[0])
        ipath = list(l)[0]
        gpath = pathlib.Path(f"{ipath}/{g}")
        print(gpath)
        # print(gpath)
        # hpath = gpath.iterdir()[0]
        # newPath = f"{hpath}/IMG_DATA"
        # img_path.append(newPath)