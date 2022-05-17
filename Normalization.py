#--->importing req libraries
try:
    import os
    import tensorflow as tf
    import cv2 
    import keras
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from PIL import Image
    import pandas as pd
    from keras import layers
    from keras.models import Sequential
    from sklearn.utils import class_weight
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.preprocessing import LabelBinarizer
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.metrics import multilabel_confusion_matrix
    import seaborn as sns
    from PIL import Image
    from tqdm import tqdm
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")
    

    
training_path = r'E:\FCAI.HU\Lev.3\Se.2\Selected - 2\PROJECT CNN\Training'
CATEGORIES = ["glioma", "meningioma","pituitary"]
imgSize = 227 

        
def create_training_data():
    for category in CATEGORIES:  # do glioma ,meningioma and pituitary
        path = os.path.join(training_path,category)  # create path to them
        class_num = CATEGORIES.index(category)  # get the classification  (0 or 1 or 2). 0=glioma 1=meningioma 2=pituitary

        for img in tqdm(os.listdir(path)):  # iterate over each images
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (imgSize, imgSize))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

training_data = []
create_training_data()
print('training data is :', len(training_data))

def print_max_min_pixelValues(folder_path):
    minPixelValues = []
    maxPixelValues = []
    for dirName, _, fileNames in os.walk(folder_path):
        for fileName in fileNames:
            min_max = Image.open(os.path.join(dirName, fileName), 'r').getextrema() # tuple of 3 tuples; for R, G and B
            if (np.ndim(min_max) == 1):
                minPixelValues.append(min_max[0])
                maxPixelValues.append(min_max[1])
            else:
                minPixelValues.append(min(min_max[i][0] for i in range(np.shape(min_max)[0])))
                maxPixelValues.append(max(min_max[i][1] for i in range(np.shape(min_max)[0])))
    print("Max pixel value: ", max(maxPixelValues), ", Min pixel value :", min(minPixelValues))

print_max_min_pixelValues(training_path)  

from sklearn.utils import shuffle

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
    
X = np.array(X).reshape(-1, imgSize, imgSize, 1) 
y = np.array(y)
X, y = shuffle(X, y)

print(X[12].reshape(-1, imgSize, imgSize, 1))

print (X.shape)
print (y.shape)

from sklearn.preprocessing import MinMaxScaler
x = np.array(X).reshape(-1, imgSize) 

# build the scaler model
scaler = MinMaxScaler()
# fit using the train set
scaler.fit(x)
# transform the test test
X_scaled = scaler.transform(x)
# Verify minimum value of all features
X_scaled.min(axis=0)
# array([0., 0., 0., 0.])
# Verify maximum value of all features
X_scaled.max(axis=0)
# array([1., 1., 1., 1.])
# Manually normalise without using scikit-learn
X_manual_scaled = (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
# Verify manually VS scikit-learn estimation
print(np.allclose(X_scaled, X_manual_scaled))
#True

scaler.fit(x)

print(scaler.transform(x))

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print ('Done .. ! ' )
