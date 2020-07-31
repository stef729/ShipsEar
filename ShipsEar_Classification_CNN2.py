# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:06:18 2020

@author: Administrator
"""

# 特征向量：利用MFCC提取（60，41）维特征，再提取一阶差分，组成特征向量（60，41，2）
# 样本容量：采用滑窗，每段5s音频提取多个帧，帧移长度 window_size：512*40
# train：7335； test：2445
# 一维 Mel_spectrogram  90个epoch训练，Test accuracy 0.982
# 一阶差分 Mel Test accuracy 0.983
# 一阶差分 Mel， 12分类， Test accuracy 0.988



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
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.utils import to_categorical 

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import os
import librosa
import librosa.display
import glob 
import skimage
import random


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
    # features = log_specgrams
    # 一阶差分， 二维
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    # 二阶差分，三维
    # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams)), np.zeros(np.shape(log_specgrams))), axis = 3)
    for i in range(len(features)):
        # 一阶差分
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
        # 二阶差分
        # features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0], width=3, order=1) # 一阶差分
        # features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0], width=3, order=2) # 二阶差分
    return np.array(features), np.array(labels,dtype = np.int)
        




# 特征提取
if __name__ == '__main__':
    
    path = 'D:/Project/Sound/ShipsEar/Data_frame/'
    current_path = os.listdir(path)[:-4]
    df = pd.read_csv(path + 'label_DataFrame_2.csv', names= ["Name", "ClassID", "ClassID_2"] )
    
    feature = []
    labels = []
    
    if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/CNN2_feature_2d_12c.npy'): 
        data_features,labels = extract_features(path)
        data_labels = one_hot_encode(labels)
        np.save('CNN2_feature_2d_12c',data_features)
        np.save('CNN2_label_12', data_labels)
    else:
        data_features = np.load('D:/Project/Sound/ShipsEar/Feature model/CNN2_feature_2d_12c.npy')
        data_labels = np.load('D:/Project/Sound/ShipsEar/Feature model/CNN2_label_12.npy')
    
    
    

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state = 10)
X_train = X_train.reshape(X_train.shape[0], 60, 41,2)
X_test = X_test.reshape(X_test.shape[0], 60, 41,2)
input_dim = (60, 41, 2)

# Creating Keras Model and Testing
model = Sequential()

model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(12, activation = "softmax"))
model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 256, validation_data = (X_test, Y_test))

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

# =============================================================================
# # 热力图
# =============================================================================

y_pred = model.predict_classes(X_test)
# y_pred1 = []
# for j in y_pred:
#   vec = np.zeros(11)
#   ind = np.where(j == np.amax(j))
#   vec[ind] = 1.
#   y_pred1.append(vec)  
# y_pred1 = np.array(y_pred1)

y_test1 = []
for j in Y_test:
  ind = np.where(j == np.amax(j))
  y_test1.append(ind[0][0])
y_test1 = np.array(y_test1)


import sklearn.metrics as skm
class_names = ['nature noise', 'fishing boats', 'trawlers','mussel boats','tugboats ','dredgers','motorboats','pilot boats','sailboats','passenger','ocean liners', 'ro-ro vessels']
#class_names = ['Little boat', 'Moto boat', 'Passenger', 'Ocean boat', 'Nature Noise']
cm = skm.confusion_matrix(y_test1, y_pred)
print(cm)
print( skm.classification_report(y_test1,y_pred))
import seaborn as sn
#seaborn for heatmap of representation of the results from evaulation 
df_cm = pd.DataFrame(cm)
sn.set(font_scale=1.4)
plt.subplots(figsize=(10,8))
ax = sn.heatmap(df_cm, annot=True,fmt="g",xticklabels=class_names,yticklabels=class_names,annot_kws={"size": 16},cmap="YlGnBu")











