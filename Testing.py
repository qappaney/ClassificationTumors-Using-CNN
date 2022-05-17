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
    import pickle
    from tqdm import tqdm
    import tensorflow as tf
    import keras
    import matplotlib.pyplot as plt
    from keras import layers
    from keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from sklearn.utils import class_weight
    from tensorflow.keras import optimizers
    from tensorflow.keras import losses
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation ,BatchNormalization
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.model_selection import train_test_split
    import pickle
    print("Library Loaded Successfully ..........")
except:
    print("Library not Found ! ")

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)
print ( X.shape , y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , shuffle=True)



test_path = r'/content/drive/MyDrive/tumors/TumorsClassification/Testing'
CATEGORIES = ["glioma", "meningioma", "pituitary"]
imgSize = 227


def create_test_data():
    for category in CATEGORIES:  # do glioma ,meningioma and pituitary
        path = os.path.join(test_path, category)  # create path to them
        class_num = CATEGORIES.index(
            category)  # get the classification  (0 or 1 or 2). 0=glioma 1=meningioma 2=pituitary

        for img in tqdm(os.listdir(path)):  # iterate over each images
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (imgSize, imgSize))  # resize to normalize data size
                test_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

test_data = []
create_test_data()
print('testing data is :', len(test_data))

loaded_model = load_model("tarining.h5")
loss, accuracy = loaded_model.evaluate(x_test, y_test)
print('test accuracy : %.2f' % accuracy, "%")



def inverse_classes(num):
    if num==0:
        return 'Glioma Tumor'
    elif num==1:
        return 'Meningioma Tumor'
    elif num==2:
        return 'Pituitary Tumor'
    else:
        return 'Error !!'




# Prediction using CNN model
plt.figure(figsize=(15,12))
for i in range(4):
    plt.subplot(3,2,(i%12)+1)
    index=np.random.randint(394)
    pred_class=inverse_classes(np.argmax(model.predict(np.reshape(x_test[index],(-1,227,227,1))),axis=1))
    plt.title('This image is of {0} and is predicted as {1}'.format(inverse_classes(y_test[index]),pred_class),
              fontdict={'size':13})
    plt.imshow(np.squeeze(x_test[index]))
    plt.tight_layout()


pred = model.predict(test_X)
pred = np.argmax(pred,axis=1)
print(classification_report(test_y,pred))

cm = confusion_matrix(test_y, pred)
import itertools
from itertools import product


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print('Confusion matrix')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm,CATEGORIES)