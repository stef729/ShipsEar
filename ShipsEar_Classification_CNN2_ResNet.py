# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:06:18 2020

@author: Administrator
"""

# 
# 迁移学习训练，利用ResNet50 + ImagneNet； 训练样本：MFCC+一阶差分+二阶差分
# 训练集（7335，60，41，3）； 测试集（2445，60，41，3）
# 训练集：loss: 0.5792 - accuracy: 0.7200 - val_loss: 5.5452 - val_accuracy: 0.2311
# layers.Dense(512)，sigmoid；测试集： [5.544306003047889, 0.23108384]



import pandas as pd
import numpy as np

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential

import os
import librosa
import librosa.display



def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def extract_features(path,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 512 * (frames - 1)
    log_specgrams = []

    for i in range(1956):
        file_name = path + current_path[i]
        label = df["ClassID_2"][i]
        sound_clip,s = librosa.load(file_name)
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                melspec = librosa.feature.melspectrogram(signal, n_mels = bands)
                logspec = librosa.power_to_db(melspec)
                logspec = logspec.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
        print('The file of number %s has done' %i)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams)), np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0], width=3, order=1) # 一阶差分
        features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0], width=3, order=2) # 二阶差分
  
    return np.array(features), np.array(labels,dtype = np.int)
        




# 特征提取
if __name__ == '__main__':
    
    path = 'D:/Project/Sound/ShipsEar/Data_frame/'
    current_path = os.listdir(path)[:-2]
    df = pd.read_csv(path + 'label_DataFrame.csv', names= ["Name", "ClassID", "ClassID_2"] )
    
    feature = []
    labels = []
    current_path = os.listdir(path)[:-2]
    
    if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/Transfer_ResNet50_feature.npy'): 
        data_features,labels = extract_features(path)
        data_labels = one_hot_encode(labels)
        np.save('Transfer_ResNet50_feature_12',data_features)
        np.save('Transfer_ResNet50_label_12', data_labels)
    else:
        data_features = np.load('D:/Project/Sound/ShipsEar/Feature model/Transfer_ResNet50_feature.npy')
        data_labels = np.load('D:/Project/Sound/ShipsEar/Feature model/Transfer_ResNet50_label.npy')
    
    
    

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state = 1)
X_train = X_train.reshape(X_train.shape[0], 60, 41, 3)
X_test = X_test.reshape(X_test.shape[0], 60, 41, 3)
input_dim = (60, 41, 3)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras import  layers
from tensorflow.keras.applications.resnet import ResNet50

# Creating Keras Model and Testing
covn_base = ResNet50(
        include_top=False, weights='imagenet', input_shape=(60,41,3),
        pooling='avg', classes=5)
covn_base.trainable = True
covn_base.summary()

model = Sequential()
model.add(covn_base)
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(5, activation='sigmoid'))


# 训练后30层
fine_tune_at = -30
for layer in covn_base.layers[:fine_tune_at]:
    layer.trainable = False
model.summary()

model.compile(optimizer = Adam(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 256, validation_data = (X_test, Y_test))

predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

# =============================================================================
# # 结果可视化
# =============================================================================
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

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












