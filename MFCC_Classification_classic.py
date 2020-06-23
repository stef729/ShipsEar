# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:06:12 2020

@author: CS
"""

# mfcc: (1956，48,100),取均值，生成(1956,100)， Logistic Regression：0.32； Linear SVC is 0.34； Poly SVC is 0.36
# RBM SVC is 0.39;  K-NN is 0.41; Decision Tree is 0.34; Gradient Boosting is 0.41; 神经网络 0.33 .  mel_spec特征未见改善


# mfcc:(1956,48,100)，flatten，生成(1956,4800), LR Classifier: 0.53;  Linear SVC is 0.62; Poly SVC is 0.56
# RBM SVC is 0.76;  K-NN is 0.77;  Decision Tree is 0.44; SGD is 0.60; Gradient Boosting is 0.66; 神经网络 0.71  mel_spec特征未见改善



import tensorflow as tf
from tensorflow import keras

import os 
import librosa

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split 




path = 'D:/Project/Sound/ShipsEar/Data_frame/'
df = pd.read_csv(path + 'label_DataFrame.csv', names= ["Name", "ClassID", "ClassID_2"] )
print(df.head())



features = []
labels = []
logspec = []
all_mfcc = []
all_mfcc_spec = []
all_stft = []
all_CQT = []
current_path = os.listdir(path)[:-2]

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)
        start = int(start)


def extract_features(row):
    
    # all_label = pd.read_csv(path + 'label_DataFrame.csv')
    # label = all_label.iloc[:,1]
    for i in range(1956):
        file_name = path + current_path[i]
        # Here kaiser_fast is a technique used for faster extraction
        data, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        
        # (1) n_mfcc维度48，fft默认2048，hop_length为帧移
        mfcc_feature = librosa.feature.mfcc(y=data, sr=sample_rate,n_mfcc=48, hop_length=1024)
        # (2) mel_spectrogram
        melspec = librosa.feature.melspectrogram(data, n_mels = 48)
        logspec = librosa.power_to_db(melspec)
        # (3) STFT (1025,54)
        stft_feature = librosa.amplitude_to_db(np.abs(librosa.stft(data, hop_length=2048)), ref=np.max)
        # (4) CQT  (84,216)
        CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(data, sr=sample_rate)), ref=np.max)
        # 存入矩阵
        mfcc_feature = mfcc_feature[:,0:100]
        all_mfcc.append(mfcc_feature)
        logspec = logspec[:,0:100]
        all_mfcc_spec.append(logspec)
        
        stft_feature = stft_feature[0:1000,0:50]
        all_stft.append(stft_feature)
        
        CQT = CQT[0:80,0:200]
        all_CQT.append(CQT)
        
        labels.append(df["ClassID"][i])
        print('The number of %d has done' %i)
        
    all_mfcc_features = np.asarray(all_mfcc).reshape(len(all_mfcc),48,100)    
    all_mfcc_spec_features = np.asarray(all_mfcc_spec).reshape(len(all_mfcc_spec),48,100)
    
    all_stft_features = np.asarray(all_stft).reshape(len(all_stft),1000,50)    
    all_CQT_features = np.asarray(all_CQT).reshape(len(all_CQT),80,200)

    return np.array(all_mfcc_features), np.array(all_mfcc_spec_features), np.array(all_stft_features), np.array(all_CQT_features), np.array(labels,dtype = np.int)


if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/Classic_mfcc_features.npy'):
    all_mfcc_features, all_mfcc_spec_features, all_stft_features, all_CQT_features, labels = extract_features(df) 
    np.save('Classic_mfcc_features',all_mfcc_features)  
    np.save('Classic_mfcc_spec_features', all_mfcc_spec_features)
    np.save('Classic_stft_features', all_stft_features)
    np.save('Classic_CQT_features', all_CQT_features)
    np.save('Classic_mfcc_labels', labels)
all_mfcc = np.load('D:/Project/Sound/ShipsEar/Feature model/Classic_mfcc_features.npy', allow_pickle=True)
all_mfcc_spec = np.load('D:/Project/Sound/ShipsEar/Feature model/Classic_mfcc_spec_features.npy', allow_pickle=True)
all_mfcc_labels = np.load('D:/Project/Sound/ShipsEar/Feature model/Classic_labels.npy', allow_pickle=True)
all_stft_features = np.load('D:/Project/Sound/ShipsEar/Feature model/Classic_stft_features.npy', allow_pickle=True)
all_CQT_features = np.load('D:/Project/Sound/ShipsEar/Feature model/Classic_CQT_features.npy', allow_pickle=True)

# 降维，取均值
# all_mfcc_mean = np.mean(all_mfcc_spec.T,axis=1).transpose()
# 降维，flatten
all_feature = all_CQT_features.reshape(all_CQT_features.shape[0], -1)
data = pd.DataFrame(all_feature)
labels = pd.DataFrame(all_mfcc_labels)
data = (data - data.min())/(data.max() - data.min())
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state = 42)

from sklearn.preprocessing import StandardScaler
#标准化数据，保证每个维度特征数据方差为1，均值为0.使得预测结果不会被某些维度过大的特征值主导
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# =============================================================================
# 2 - 经典机器学习
# =============================================================================
import sys
# sys.path.append(r'D:\Project\Sound\ShipsEar\Samples')
# from def_Common_Machine_Learn import def_Common_Machine_Learn
# obj = def_Common_Machine_Learn(X_train, X_test, y_train, y_test)
# obj.Logistic_Regression()
# obj.Linear_SVC()
# obj.Poly_SVC()
# obj.RBF_SVC()
# obj.K_NN()
# obj.Decision_Tree()
# obj.SGD_Classifier()
# obj.Gradient_Boosting()

# =============================================================================
# 3 - 浅层神经网络训练
# 训练结果  train：0.69， test:0.59
# =============================================================================
from tensorflow.keras.layers import Flatten, Dense, LSTM

model = keras.Sequential()
# model.add(Dense, input_shape(100,))
model.add(Dense(256, input_shape=(16000,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=128, epochs=100)
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)
