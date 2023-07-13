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

    def clip_raster(self, inputFIle = False)
        fileName = input("Path to image file to clip: ")
        outFname = input("Folder path to save the image file") 
        
        

