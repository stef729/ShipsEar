# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 10:20:40 2020

@author: Administrator
"""
# 8 layer VGGish data_original_1s, (7335, 2445)  Test accuracy: 0.9938
# 8 layer VGGish data_aug 1s, (36675, 2445) Test accuracy: 0.9673


import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler

# Libraries for Classification and building Models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical 

import os
import librosa
import librosa.display
import glob 
import skimage
import random

import specaugment

# 滑窗函数，可通过window_size控制步长
def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size)
        start = int(start)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def extract_features(path,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []

    for i in range(1956):
        file_name = path + current_path[i]
        label = df["ClassID"][i]
        sound_clip,s = librosa.load(file_name)
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.power_to_db(melspec)
                logspec_1 = logspec.T.flatten()[:, np.newaxis]
                log_specgrams.append(logspec_1)

                labels.append(label)
        print('The file of number %s has done' %i)
    features = np.asarray(log_specgrams).reshape(len(log_specgrams), melspec.shape[0], 41, order="F")
    return np.array(features), np.array(labels,dtype = np.int)

# =============================================================================
# # data Augmentation, the number of 4 augment
# =============================================================================
def data_augment(data, label):    
    X_train_aug = []
    labels = []
    for i in range(X_train.shape[0]):
        data = X_train[i,:,:]
        label = np.argmax(Y_train, axis=1)
        aug_flat = []
        for k in range(1, 5):
            k = k - k +1
            data_aug = specaugment.data_aug_main(data, k)
            data_aug = data_aug[0].T.flatten()[:, np.newaxis]
            aug_flat.append(data_aug)
        
        X_train_aug.append(data.T.flatten()[:, np.newaxis])
        X_train_aug.extend(aug_flat)
        labels.append(label[i])
        labels.append(label[i])
        labels.append(label[i])
        labels.append(label[i])
        labels.append(label[i])
        print('The file of number %s has done' %i)
    features = np.asarray(X_train_aug).reshape(len(X_train_aug), 60, 41, order="F")
    return np.array(features), np.array(labels,dtype = np.int)



        
if __name__ == '__main__':
    
    path = 'D:/Project/Sound/ShipsEar/Data_frame/'
    current_path = os.listdir(path)[:-4]
    df = pd.read_csv(path + 'label_DataFrame_2.csv', names= ["Name", "ClassID", "ClassID_2"] )
     
    if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/data_feature_1s.npy'): 
        data_features,labels = extract_features(path)
        data_labels = to_categorical(labels)
        np.save('data_feature_1s',data_features)
        np.save('data_label_1s', data_labels)
    else:
        data_features = np.load('D:/Project/Sound/ShipsEar/Feature model/data_feature_1s.npy')
        data_labels = np.load('D:/Project/Sound/ShipsEar/Feature model/data_label_1s.npy')        


X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state = 10)

# =============================================================================
# # 对训练集进行数据增广
# =============================================================================
# if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/data_feature_aug_1s.npy'):
#     train_aug_features, train_aug_labels = data_augment(X_train, Y_train)
#     train_aug_labels = to_categorical(train_aug_labels)
#     np.save('data_feature_aug_1s', train_aug_features)
#     np.save('data_label_aug_1s', train_aug_labels)
# else:
#     train_aug_features = np.load('D:/Project/Sound/ShipsEar/Feature model/data_feature_aug_1s.npy')
#     train_aug_labels = np.load('D:/Project/Sound/ShipsEar/Feature model/data_label_aug_1s.npy')   
# X_train = train_aug_features
# Y_train = train_aug_labels


X_train = X_train.reshape(X_train.shape[0], 60, 41, 1)
X_test = X_test.reshape(X_test.shape[0], 60, 41, 1)


# =============================================================================
# # 2 layer base_line
# =============================================================================
input_dim = (60, 41, 1)
model = Sequential()
model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu", input_shape = input_dim))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu", input_shape = input_dim))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation = "tanh"))
model.add(Dense(5, activation = "softmax"))


# =============================================================================
# # 8 layer VGGish 
# =============================================================================
# input_dim = (60, 41, 1)
# model = Sequential()

# model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu", input_shape = input_dim))
# model.add(BatchNormalization())
# model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(Conv2D(256, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# model.add(Conv2D(512, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(Conv2D(512, (3, 3), padding = "same", activation = "relu"))
# model.add(BatchNormalization())
# model.add(MaxPool2D(pool_size=(2, 2)))

# # model.add(Flatten())
# model.add(GlobalMaxPooling2D())
# model.add(Dense(5, activation = "softmax"))



model.summary()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 50, batch_size = 128, validation_data = (X_test, Y_test))
predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

# =============================================================================
# # 结果可视化
# =============================================================================
history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']


epochs = range(1, len(acc) + 1)

# “bo”代表 "蓝点"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b代表“蓝色实线”
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()   # 清除数字

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()





































