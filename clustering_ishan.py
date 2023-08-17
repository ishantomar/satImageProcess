import numpy as np
import rasterio as rio
from tslearn import clustering
from pathlib import Path
import matplotlib.pyplot as plt

fname = input(r"Input image file path: ")
outfolder = input(r"Input output folder: ")

with rio.open(fname) as dst:
    img3d = dst.read()
    meta = dst.meta
    img3dShape = img3d.shape

# Array to work on::
#flatImg = img3d.reshape(-1. img_shape[0],1)
#Not right
flatImg = img3d.reshape(img3dShape[0],-1).T
flatImg = flatImg.reshape(flatImg.shape[0],flatImg.shape[1],1)

n_clusters = 8 

#KMeans "dtw"

# km_dst = clustering.TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_init=3, n_jobs=5, max_iter=5,max_iter_barycenter=3,).fit(flatImg)
#Kmeans fit done
from datetime import datetime
start=datetime.now()
print("Kmeans starting")
km_dst = clustering.TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", n_init=3, n_jobs=5, max_iter=5,max_iter_barycenter=3,).fit(flatImg)
# km_dst = clustering.KShape(n_clusters=n_clusters, n_init=3).fit(flatImg)

print(datetime.now()-start)

start2 = datetime.now()
# flatCluster = km_dst.fit_predict(flatImg)
flatCluster = km_dst.labels_
print(datetime.now()-start2)
out = flatCluster.reshape(img3dShape[1:])
#Save Tiff Image
meta.update(count = 1)
outFname = input(r"Output file name: ")
outFpath = Path(outfolder, outFname)
with rio.open(outFpath, 'w', **meta) as dst:
    dst.write(out,1)
#Plot graph
plt.ioff()
i = 1
while i <= n_clusters:
    clusterIdxs = np.where(flatCluster==i)[0]
    randomIdxs = np.random.choice(clusterIdxs, size=10)
    for j in randomIdxs:
        plt.plot(flatImg[j,:,0])
    plotFpath = Path(outfolder, f"cluster_{i}.jpg")
    plt.savefig(plotFpath)
    plt.clf()
    i+= 1


def getElbowCurve(n_clusters, flatImg):
    clusterList = []
    inertiaList = []
    labelList = []
    i = 2
    while i <= n_clusters:
        print(i)
        km_dst = clustering.TimeSeriesKMeans(n_clusters=i, metric="euclidean", n_init=1, n_jobs=5, max_iter=1,max_iter_barycenter=1,).fit(flatImg)
        print("K means estimated")
        clusterList.append(i)
        inertiaList.append(km_dst.inertia_)
        labelList.append(km_dst.labels_)
        i+= 1
    return clusterList, inertiaList, labelList

# To save the images with clusters merged

def mergeCluster(clusterList, img, meta, imgShape, destFpath=None):
    zeroImg = np.zeros(clusterImgShape)
    meta.update(count=1)
    for i in clusterList:
        oneIdx = np.where(clusterImg==i)
        print(i)
    if destFpath is None:
        destFpath = input(r"Input dest file path: ")
    with rio.open(destFpath, 'w', **meta) as dst:
        dest.write(zeroImg)
