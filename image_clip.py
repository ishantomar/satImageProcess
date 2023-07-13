import rasterio
import matplotlib.pyplot as plt
import pathlib
import geopandas as gpd
import fiona
class ImageClip:
    def __init__(self)
        self.shpPath = input("Path to shape file: ")
        self.b8path = input("Band 8 path: ")
        self.b4path = input("Band 4 path: ")
        self.b8apath = input("Band 8A path: ")
        self.b11path = input("Band 11 path: ")

    def clip_raster(self, inputFIle = False, noData=-1)
        fileName = input("Path to image file to clip: ")
        outFname = input("Folder path to save the image file") 
        with fiona.open(self.shpPath, 'r') as shapeFile:
            for feature in shapeFile:
                shapes = [feature['geometry']]
        with rasterio.open(fileName) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True, filled=True, nodata=noData)
            out_meta = src.meta
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        with rasterio.open(outFname, 'w', **out_meta) as dst:
            dst.write(out_image)
    def getNDVI(self, inputFile = False):
        if inputFIle == False:
            nirB = input("Give file name for NIR Band: ")
            rB = input("Give the file name for Red Band ")
        else: 
            nirb = self.b8apath
            rb = self.b4path
        nirImg = rasterio.open(nirB)
        rImg = rasterio.open(rB)
        #Check if bounds are same 
        # Code here
        nirArray = nirImg.read(1)
        rArray = rImg.read(1)
        ndviArray = (nirArray-rArray)/(nirArray+rArray)
        out_meta = nirImg.meta
        with rasterio.open(outFname, 'w', **out_meta) as dst:
            dst.write(ndviArray)
    def getNDWI(self, inputFile=False):
        if inputFIle == False:
            b8a = input("Give file name for NIR Band: ")
            b11 = input("Give the file name for Red Band ")
        else: 
            b8a = self.b8apath
            b11 = self.b4path
        nirImg = rasterio.open(b8a)
        rImg = rasterio.open(b11)
        #Check if bounds are same 
        # Code here
        nirArray = nirImg.read(1)
        rArray = rImg.read(1)
        ndviArray = (nirArray-rArray)/(nirArray+rArray)
        out_meta = b8a.meta
        with rasterio.open(outFname, 'w', **out_meta) as dst:
            dst.write(ndviArray)