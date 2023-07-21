import rasterio
import matplotlib.pyplot as plt
import pathlib
import geopandas as gpd
import fiona
from pathlib import Path
import numpy as np
import os
import glob
class ImageStack:
    def __init__(self):
        pass

    def image_stack(self):
        """
        To stack rgb and nir image
        """
        rgbfpath = input(r"Path to rgb image")
        nirfpath = input(r"Path to nir file")
        outputfolder = input(r"Path to output folder")

        rgbimg = rasterio.open(rgbfpath)
        nirimg = rasterio.open(nirfpath)
        #check if the file names are same and extract the file name
        rgbfname = Path(rgbfpath).name
        nirfname = Path(nirfpath).name
        try: rgbfname == nirfname
        except NameError:
            print("Filenames do not match")
        outfname = rgbfname
        outfpath = Path(outputfolder, outfname)
        rgbarray = rgbimg.read()
        nirarray = nirimg.read()
        stackedarray = np.dstack((rgbarray, nirarray))
        meta = rgbimg.meta
        with rasterio.open(outfpath, 'w', **meta) as dst:
            dst.write(stackedarray)

    def rgbnirFolderstack(self):
        rgbfolder = input(r"rgb image folder: ")
        nirfolder = input(r"nir image folder: ")
        outputfolder = input(r"output folder: ")
        rgbfiles = os.listdir(rgbfolder)
        filesAbsent = []
        for i in rgbfiles:
            if os.path.isfile(Path(nirfolder,i))==False:
                print(i)
                filesAbsent.append(i)
        try:
            len(filesAbsent) == 0
        except OSError:
            print("Following files are absent in nir folder")
            print(filesAbsent)
        for i in rgbfiles:
            rgbfPath = Path(rgbfolder, i)
            nirfPath = Path(nirfolder, i) 
            outfPath = Path(outputfolder, i)
            rgbimg = rasterio.open(rgbfPath)
            nirimg = rasterio.open(nirfPath)
            b = rgbimg.read(1)
            # print(b.shape)
            g = rgbimg.read(2)
            # print(g.shape)
            r = rgbimg.read(3)
            # print(r.shape)
            nir = nirimg.read(1)
            # print(nir.shape)
            # stackedarray = np.dstack((r,g,b,nir))
            meta = rgbimg.meta
            meta.update({
                'count': 4
            })
            with rasterio.open(outfPath, 'w', **meta) as dst:
                dst.write(r, 1)
                dst.write(g, 2)
                dst.write(g, 3)
                dst.write(nir, 4)

    def FCCFolderstack(self):
        rgbfolder = input(r"rgb image folder: ")
        nirfolder = input(r"nir image folder: ")
        outputfolder = input(r"output folder: ")
        # rgbfiles = glob.glob(rgbfolder+"/*.tif")
        rgbfiles = os.listdir(rgbfolder)
        filesAbsent = []
        for i in rgbfiles:
            if os.path.isfile(Path(nirfolder,i))==False:
                print(i)
                filesAbsent.append(i)
        try:
            len(filesAbsent) == 0
        except OSError:
            print("Following files are absent in nir folder")
            print(filesAbsent)
        for i in rgbfiles:
            rgbfPath = Path(rgbfolder, i)
            nirfPath = Path(nirfolder, i) 
            outfPath = Path(outputfolder, i)
            rgbimg = rasterio.open(rgbfPath)
            nirimg = rasterio.open(nirfPath)
            b = rgbimg.read(1)
            # print(b.shape)
            g = rgbimg.read(2)
            # print(g.shape)
            r = rgbimg.read(3)
            # print(r.shape)
            nir = nirimg.read(1)
            # print(nir.shape)
            # stackedarray = np.dstack((r,g,b,nir))
            meta = rgbimg.meta
            with rasterio.open(outfPath, 'w', **meta) as dst:
                dst.write(nir, 1)
                dst.write(r, 2)
                dst.write(b, 3)
            print(i)
