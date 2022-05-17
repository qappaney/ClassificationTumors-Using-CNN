#--->importing req libraries
try:
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
    
    print("Libraries Loaded Successfully ..........")
except:
    print("Library not Found ! ")

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

print ( X.shape , y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , shuffle=True)

model = Sequential()

model.add (keras.Input(shape=(227,227,1)))
model.add(Conv2D(32,(5,5),padding="same"))
model.add(Activation('relu'))
model.add(BatchNormalization(axis = -1, name = 'bn0'))
model.add(MaxPooling2D(pool_size=(3,3)))
    
model.add(Conv2D(64,(5,5), padding="same"))
model.add(Activation('relu')) 
model.add(BatchNormalization(axis =-1, name = 'bn1'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Dropout(0.5))
model.add(Dense(32))
model.add(Flatten())
model.add(Dense(3))

model.summary()

# training the model
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "Adam",
              metrics =['accuracy'])

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

# # fitting the model
history = model.fit(x_train, y_train, batch_size = 10,epochs =10, verbose =1,validation_data =(x_test, y_test))


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()







