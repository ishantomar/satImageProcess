# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:01:35 2023

@author: HP
"""

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



polygonfile = r'D:\10007\research\crop_classification\training_data\train_5.shp'
s2folder = r'D:\10007\research\crop_classification\rededge'


# Create geodataframe from the polygons
df = gpd.read_file(polygonfile)
df.head(5)

# Get the count of each unique value in the column
value_counts = df['Crop_Id_Ne'].value_counts()

# Print the count of each unique value
print(value_counts)

# # Define a dictionary to map old values to new values
# value_mapping = {'old_value1': 'new_value1', 'old_value2': 'new_value2', 'old_value3': 'new_value3'}

# # Replace the values in the 'column_name' column using the mapping dictionary
# df['column_name'] = df['column_name'].replace(value_mapping)


df= df.dropna()



# # Get the unique values in the column
# unique_values = df['LANDNAME'].unique()

# # Print the unique values
# print(unique_values)

# Get the count of each unique value in the column
value_counts = df['Crop_Id_Ne'].value_counts()

# Print the count of each unique value
print(value_counts)


# # encoding classes
# from sklearn.preprocessing import LabelEncoder

# # Assuming you have a dataframe named 'df' with a column named 'column_name' that you want to encode

# # Create an instance of LabelEncoder
# label_encoder = LabelEncoder()

# # Fit and transform the column you want to encode
# encoded_column = label_encoder.fit_transform(df['LANDNAME'])

# # Replace the original column with the encoded values
# df['LANDNAME'] = encoded_column



# Calculate statistic(s) for each polygon
for root, folders, files in os.walk(s2folder):
    for file in files:
        f = os.path.join(root, file)
        if os.path.isfile(f) and f.endswith('.tif'):
            print(file)
            stats = ['mean']
            df2 = pd.DataFrame(zonal_stats(vectors=df['geometry'], raster=f, stats=stats))
            df2.columns = ['{0}_{1}'.format(stat, file.split('.')[0]) for stat in stats]
            df = df.join(df2)
            
            
# # adding indices    
# import geopandas as gpd
# import pandas as pd
# from rasterstats import zonal_stats

# # Path to the NDVI raster
# ndvi_raster = 'path/to/ndvi.tif'


# # Calculate statistics for each point based on the NDVI raster
# stats = zonal_stats(df['geometry'], ndvi_raster, stats="mean")  # You can specify multiple statistics if needed

# # Create a new column in the GeoDataFrame to store the statistics
# df['NDVI_mean'] = [s['mean'] for s in stats]  # Access the specific statistic you need

# # Print the updated GeoDataFrame with the NDVI statistics
# print(df)
       
# If AttributeError: 'NoneType' object has no attribute 'get' error is showed (because of the none values) just APPLY dropna()  
   
df= df.dropna()




# # Get the count of each unique value in the column
# value_counts = df['LANDNAME'].value_counts()

# # Print the count of each unique value
# print(value_counts)


# Separate the features (X) and the target variable (y)
X = df.iloc[:, 5:]  
y = df.iloc[:, 3]   

# # Assuming you have the features X and the target variable y

# # Create an instance of the RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)

# # Perform random undersampling
# X_resampled, y_resampled = rus.fit_resample(X, y)

# # Check the class distribution after undersampling
# unique, counts = np.unique(y_resampled, return_counts=True)
# print("Class distribution after undersampling:", dict(zip(unique, counts)))

# Resampling Techniques Oversampling code

# Assuming you have the features X and the target variable y

# Create an instance of the RandomOverSampler
ros = RandomOverSampler(random_state=42)
# ros = RandomUnderSampler(random_state=42)
# Perform random oversampling
X, y = ros.fit_resample(X, y)

# Check the class distribution after oversampling
unique, counts = np.unique(y, return_counts=True)
print("Class distribution after oversampling:", dict(zip(unique, counts)))


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# hypertunning
from sklearn.model_selection import RandomizedSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
#Hyperparameter tuning
# from sklearn.model_selection import RandomizedSearchCV
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
#     'gamma': ['scale', 'auto'],
#     'degree': [2, 3, 4],
#     'coef0': [0.0, 0.5, 1.0]
# }
svm = SVC()
random_search = RandomizedSearchCV(estimator=svm, param_distributions=param_grid, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train_scaled, y_train)

print('Best parameters:', random_search.best_params_)
print('Best score:', random_search.best_score_)

best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test_scaled, y_test)
print('Test accuracy:', test_accuracy)


# Make predictions on the test set
y_pred = best_model.predict(X_test_scaled)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate accuracy
accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
print('Test accuracy:', accuracy)

# F1 SCORE
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)

print(cm)
#      shows the classification report
class_report = classification_report(y_test,y_pred)
print (class_report)
    







# ROC AUC CURVE 

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.svm import SVC
# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc
# import numpy as np

# # Assuming you have training data 'X_train' and corresponding labels 'y_train'
# # Assuming you have test data 'X_test' and corresponding labels 'y_test'

# # Train a multiclass SVM classifier using One-vs-Rest strategy
# svm_classifier = OneVsRestClassifier(SVC(probability=True))
# svm_classifier.fit(X_train_scaled, y_train)

# # Obtain the decision scores for each class
# decision_scores = svm_classifier.decision_function(X_test_scaled)

# # Binarily encode the true labels of the test set using one-hot encoding
# y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

# # Compute the ROC curve and micro-averaged AUC
# fpr, tpr, _ = roc_curve(y_test_bin.ravel(), decision_scores.ravel())
# auc_score = auc(fpr, tpr)

# # Print the micro-averaged AUC
# print("Micro-Averaged AUC:", auc_score)



# new roc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Assuming you have training data 'X_train' and corresponding labels 'y_train'
# Assuming you have test data 'X_test' and corresponding labels 'y_test'

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'estimator__C': [0.1, 1, 10],
    'estimator__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'estimator__gamma': ['scale', 'auto']
}

# Create the SVM classifier with probability=True for ROC curve
svm_classifier = OneVsRestClassifier(SVC(probability=True))

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)

# Obtain the best SVM classifier with the optimal hyperparameters
best_svm_classifier = grid_search.best_estimator_

# Obtain the decision scores for each class using the best classifier
decision_scores = best_svm_classifier.decision_function(X_test_scaled)

# Binarily encode the true labels of the test set using one-hot encoding
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))

# Get the unique class names from y
class_names = np.unique(y_train)

# Compute the ROC curve for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = y_test_bin.shape[1]

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], decision_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot the ROC curve for each class with class names
plt.figure()
colors = ['blue', 'red', 'green', 'orange', 'purple', 'pink']  # Choose colors for each class
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (AUC = {1:.2f})'.format(class_names[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Make predictions on the test set
y_pred = best_svm_classifier.predict(X_test_scaled)

# Compute the confusion matrix
cmroc = confusion_matrix(y_test, y_pred)


# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# F1 SCORE
f1 = f1_score(y_test, y_pred, average='weighted')
print('F1 score:', f1)

print(cm)
#      shows the classification report
class_report = classification_report(y_test,y_pred)
print (class_report)
    

# Get the count of each unique value in the column
value_counts = df['Crop_Id_Ne'].value_counts()

# Print the count of each unique value
print(value_counts)

















# ACCURACY

polygonfile1 = r'D:\10007\research\crop_classification\training_data\train_2.shp'
s2folder1 = r'D:\10007\research\crop_classification\rededge'


# Create geodataframe from the polygons
df1 = gpd.read_file(polygonfile1)
df1.head(5)


df1= df1.dropna()

# Calculate statistic(s) for each polygon
for root, folders, files in os.walk(s2folder1):
    for file in files:
        f = os.path.join(root, file)
        if os.path.isfile(f) and f.endswith('.tif'):
            print(file)
            stats = ['mean']
            df2 = pd.DataFrame(zonal_stats(vectors=df1['geometry'], raster=f, stats=stats))
            df2.columns = ['{0}_{1}'.format(stat, file.split('.')[0]) for stat in stats]
            df1 = df1.join(df2)
            
df1= df1.dropna()    
        
# Separate the features (X) and the target variable (y)
X1 = df.iloc[:, 5:]  
y1 = df.iloc[:, 3]   

# Scale the features using StandardScaler
scaler = StandardScaler()
X1 = scaler.fit_transform(X1)

# Make predictions on the test set
y_pred1 = best_model.predict(X1)

# Compute the confusion matrix
cm1 = confusion_matrix(y1, y_pred1)

# Calculate accuracy
accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
print('Test accuracy:', accuracy)

# F1 SCORE
f1_2 = f1_score(y1, y_pred1, average='weighted')
print('F1 score:', f1)


# prediction
pre_folder=r'D:\10007\research\crop_classification\resampled'


b2 = rasterio.open(os.path.join(s2folder, 'b2.tif')).read()
b2 = b2[0,:,:] 
b3 = rasterio.open(os.path.join(s2folder, 'b3.tif')).read()
b3 = b3[0,:,:]
b4 = rasterio.open(os.path.join(s2folder, 'b4.tif')).read()
b4 = b4[0,:,:]
b5 = rasterio.open(os.path.join(s2folder, 'b5.tif')).read()
b5 = b5[0,:,:] #Drop the first dimension that is created when using rasterio open
b6 = rasterio.open(os.path.join(s2folder, 'b6.tif')).read()
b6 = b6[0,:,:]
b7 = rasterio.open(os.path.join(s2folder, 'b7.tif')).read()
b7 = b7[0,:,:]
b8 = rasterio.open(os.path.join(s2folder, 'b8.tif')).read()
b8 = b8[0,:,:]

bands = np.dstack((b2,b3,b4,b5,b6,b7,b8))

bands = bands.reshape(int(np.prod(bands.shape)/7),7)

bands_scaled = scaler.transform(bands)

predictions = best_model.predict(bands_scaled)

predictions = predictions.reshape(b6.shape)

b2src = rasterio.open(os.path.join(s2folder, 'b2.tif'))
with rasterio.Env():
    profile = b2src.profile
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw')
    with rasterio.open('crop_classification_2.tif', 'w', **profile) as dst:
        dst.write(predictions.astype(rasterio.uint8), 1)
        

import matplotlib.pyplot as plt

# Assuming you have already obtained the 'predictions' array

# Plot the predicted result
plt.imshow(predictions, cmap='viridis')  # Adjust the colormap as per your preference
plt.colorbar(label='Class')
plt.title('Predicted Result')
plt.show()


