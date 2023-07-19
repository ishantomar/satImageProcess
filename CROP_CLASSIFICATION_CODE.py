# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:41:36 2023

@author: HP
"""

# %%importing libraries>

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import os
import numpy as np
import seaborn as sns
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import time
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from rasterio.mask import mask
import geopandas as gpd
import glob

# %% data location>

# polygonfile = r'D:\10007\research\crop_classification\training_data\polypoint(B).shp'
polygonfile = r'D:\10007\research\crop_classification\training_data\train_5.shp'
train_folder = r'D:\10007\research\crop_classification\savi\savi'

df = gpd.read_file(polygonfile)
df.head(5)

value_counts = df['Crop_Id_Ne'].value_counts()
print(value_counts)

df= df.dropna()

# Calculate statistic(s) for each polygon/points>
for root, folders, files in os.walk(train_folder):
    for file in files:
        f = os.path.join(root, file)
        if os.path.isfile(f) and f.endswith('.tif'):
            print(file)
            stats = ['mean']
            df2 = pd.DataFrame(zonal_stats(vectors=df['geometry'], raster=f, stats=stats))
            df2.columns = ['{0}_{1}'.format(stat, file.split('.')[0]) for stat in stats]
            df = df.join(df2)
df= df.dropna()


# %%


# save the datframe
csv_path=r"D:\10007\research\crop_classification\training_data\df_t_data.csv"
df.to_csv(csv_path, index=False)



# save the datframe
csv_path=r"D:\10007\research\crop_classification\training_data\df_t_data.csv"
df.to_csv(csv_path, index=False)


df1 = df.sample(n=int(len(df) * 0.12), random_state=42)
print(df1['Crop_Id_Ne'].value_counts())


# %%

polygonfile1 = r"D:\10007\research\crop_classification\training_data\df_t_data.csv"

df = pd.read_csv(polygonfile1)
df.head(5)

df1 = df.sample(n=int(len(df) * 0.12), random_state=42)
print(df1['Crop_Id_Ne'].value_counts())

# %%
# Separate the features (X) and the target variable (y)>
df1=df
X = df1.iloc[:, 5:]  
y = df1.iloc[:, 3]   

# %%

# Resampling Techniques Oversampling/Undersampling code  
     
# Creating an instance of the RandomOverSampler

ros = RandomOverSampler(random_state=42)

# ros = RandomUnderSampler(random_state=42)

# Performing random oversampling
X, y = ros.fit_resample(X, y)

# Checking the class distribution after oversampling
unique, counts = np.unique(y, return_counts=True)
print("Class distribution after oversampling:", dict(zip(unique, counts)))
# %%

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

# making a function for roc_auc score
from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(y_test,y_pred):
    y_test_new = LabelBinarizer().fit_transform(y_test)
    y_pred_new = LabelBinarizer().fit_transform(y_pred)
    return round(roc_auc_score(y_test_new,y_pred_new)*100,2)

def classification_model(model):
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    
    scoree = round(accuracy_score(y_test,y_pred)*100,2)
    
    f1_s = round(f1_score(y_test,y_pred,average='micro')*100,2)
    
    cross_v = cross_val_score(model,X,y,cv=10,scoring='accuracy').mean()
    
    roc_ = multiclass_roc_auc_score(y_test,y_pred)
    
    print ("Model:",str(model).split("(")[0])
    print ("Accuracy Score:",scoree)
    print ("f1 Score:",f1_s)
    print ("CV Score:",cross_v)
    print ("ROC_AUC Score:",roc_)
    
#shows the classification report
    class_report = classification_report(y_test,model.predict(X_test_scaled))
    print (class_report)
    
    
# shows the confusion matrix
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True,square=True)

            
# %%

# RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc_para = {
    'n_estimators': [100, 200, 300],  # Number of decision trees in the forest
    'max_depth': [None, 5, 10],       # Maximum depth of the decision trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
    'criterion':['gini','entropy']
}

gsCV_rfc = GridSearchCV(rfc,rfc_para,cv=3,scoring='accuracy')

gsCV_rfc.fit(X_train_scaled,y_train)

para_n=gsCV_rfc.best_params_

gsCV_rfc.best_estimator_

best_rbf_classifier=gsCV_rfc.best_estimator_

classification_model(gsCV_rfc.best_estimator_)

# %%
# PREDICT IMAGE
# prediction_path= r'D:\10007\research\crop_classification\pre_out\pre_ns_rendvi_try.tif'
   
# FUNCTION OF OUTPUT
def predict_class(model,folder):
    # b2 = rasterio.open(os.path.join(folder, 'rendvi1.tif')).read()
    # b2 = b2[0,:,:] 
    # b3 = rasterio.open(os.path.join(folder, 'rendvi2.tif')).read()
    # b3 = b3[0,:,:]
    # b4 = rasterio.open(os.path.join(folder, 'rendvi3.tif')).read()
    # b4 = b4[0,:,:]
    # b5 = rasterio.open(os.path.join(folder, 'rendvi5.tif')).read()
    # b5 = b5[0,:,:] 
    # b6 = rasterio.open(os.path.join(folder, 'rendvi6.tif')).read()
    # b6 = b6[0,:,:]
    # b7 = rasterio.open(os.path.join(folder, 'rendvi7.tif')).read()
    # b7 = b7[0,:,:]
    # b8 = rasterio.open(os.path.join(folder, 'rendvi8.tif')).read()
    # b8 = b8[0,:,:]    
    b9 = rasterio.open(os.path.join(folder, 'savi1.tif')).read()
    b9 = b9[0,:,:] 
    b10 = rasterio.open(os.path.join(folder, 'savi2.tif')).read()
    b10= b10[0,:,:]
    b11= rasterio.open(os.path.join(folder, 'savi3.tif')).read()
    b11= b11[0,:,:]
    b12= rasterio.open(os.path.join(folder, 'savi5.tif')).read()
    b12= b12[0,:,:] 
    b13= rasterio.open(os.path.join(folder, 'savi6.tif')).read()
    b13= b13[0,:,:]
    b14= rasterio.open(os.path.join(folder, 'savi7.tif')).read()
    b14= b14[0,:,:]
    b15= rasterio.open(os.path.join(folder, 'savi8.tif')).read()
    b15= b15[0,:,:]

    # bands = np.dstack((b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15))
    bands = np.dstack((b9,b10,b11,b12,b13,b14,b15))

    bands = bands.reshape(int(np.prod(bands.shape)/7),7)

    bands_scaled = scaler.transform(bands)

    predictions = model.predict(bands_scaled)

    predictions = predictions.reshape(b9.shape)

    b2src = rasterio.open(os.path.join(folder, 'b2.tif'))
    with rasterio.Env():
        profile = b2src.profile
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            compress='lzw')
        with rasterio.open('savi_ppolygon_TS_os.tif', 'w', **profile) as dst:
            dst.write(predictions.astype(rasterio.uint8), 1)               
            
folder=r'D:\10007\research\crop_classification\savi\savi_clipped'


predict_class(best_rbf_classifier,folder)

# %%



# time series curve



df2=df1.iloc[:, 6:]
df2=df2.drop('geometry', axis=1)
# Group the DataFrame by Crop ID
grouped = df2.groupby('Crop_Id_Ne')

# Define the months and corresponding column names
months = ['January', 'February', 'March', 'April', 'June', 'July', 'August']
columns = ['mean_rendvi1', 'mean_rendvi2', 'mean_rendvi3','mean_rendvi5','mean_rendvi6','mean_rendvi7','mean_rendvi8']

# Plotting the graph for each crop ID
for _, group in df2.groupby('Crop_Id_Ne'):
    crop_id = group['Crop_Id_Ne'].values[0]
    pixel_values = group[columns].mean().values
    plt.plot(months, pixel_values, label=crop_id)

plt.xlabel('Months')
plt.ylabel('Pixel Values')
plt.title('Mean rendvi Values for Different Crops')
plt.legend()
# Rotate and align the x-axis labels
plt.xticks(rotation=45, ha='right')
# Display the plot
plt.show()


# %%

# CLIPPING THE CLASSIFIED FILE TO THE SHAPEFILE>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# # Path to the input TIFF file
# input_tiff = r'D:\10007\research\crop_classification\pre_out\pre_ns_rendvi.tif'
prediction_path= r'D:\10007\research\crop_classification\pre_out\raster_kmeans.tif'
# Path to the polygon shapefile(shapefile including all the firm fields)
polygon_shapefile = r'D:\10007\research\crop_classification\test\full2.shp'

# Path to the output clipped TIFF file
output_tiff = r'D:\10007\research\crop_classification\pre_out\raster_kmeans_clip.tif'


# Read the polygon shapefile using geopandas
polygon = gpd.read_file(polygon_shapefile)

# Read the input TIFF file using rasterio
with rasterio.open(prediction_path) as src:
    # Clip the raster using the polygon
    out_image, out_transform = mask(src, polygon.geometry, nodata=src.nodata, crop=True)

    # Update metadata for the clipped raster
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Write the clipped raster to a new TIFF file
    with rasterio.open(output_tiff, "w", **out_meta) as dest:
        dest.write(out_image)
# %%

















        