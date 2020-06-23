# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:45:18 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:23:35 2020

@author: Administrator
"""

# 2020年3月24日23:23:39
# https://www.kaggle.com/prabhavsingh/urbansound8k-classification
# ShipsEar - Classification
# (128,216)维的MFCC特征，求均值，生成128维特征向量

# 90 epoch, train:0.91; test:0.77


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

path = 'D:/Project/Sound/ShipsEar/Data_frame/'
df = pd.read_csv(path + 'label_DataFrame.csv', names= ["Name", "ClassID", "ClassID_2"] )
print(df.head())

# =============================================================================
# # 1 - Using Librosa to analyse random sound sample - SPECTOGRAM
# =============================================================================
dat1, sampling_rate1 = librosa.load(path + '92__Natural noise_6.wav')
dat2, sampling_rate2 = librosa.load(path + '25__Ocean liner_15.wav')

plt.figure(figsize=(20, 10))
D = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
logspec = librosa.power_to_db(D)
plt.figure(figsize=(20, 10))
plt.subplot(4, 2, 1)
librosa.display.specshow(logspec, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Log MFCC spectrogram')




plt.figure(figsize=(20, 10))
D = librosa.power_to_db(np.abs(librosa.stft(dat2)), ref=np.max)
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('STFT power spectrogram')


# =============================================================================
# # 2 - Feature Extraction and Database Building
# =============================================================================
dat1, sampling_rate1 = librosa.load(path + '66__Mussel boat_22.wav')
arr = librosa.feature.melspectrogram(y=dat1, sr=sampling_rate1)
print(arr.shape)


feature = []
label = []
logspec = []
current_path = os.listdir(path)[:-2]

def parser(row):
    # Function to load files and extract features
    
    # all_label = pd.read_csv(path + 'label_DataFrame.csv')
    # label = all_label.iloc[:,1]
    for i in range(1956):
        file_name = path + current_path[i]
        # Here kaiser_fast is a technique used for faster extraction
        X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        
        # (1) We extract mfcc feature from data
        mels = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        logspec = librosa.power_to_db(mels)
        
        # （2） STFT
        # logspec = librosa.power_to_db(np.abs(librosa.stft(X)), ref=np.max)
        
        # (3) CQT
        # logspec = librosa.power_to_db(np.abs(librosa.cqt(X, sr=sample_rate)), ref=np.max)
        
        mels = np.mean(logspec.T,axis=0)        
        feature.append(mels)
        label.append(df["ClassID"][i])
        print('The number of %d has done' %i)
    return [feature, label]

if not os.path.isfile('D:/Project/Sound/ShipsEar/Feature model/CNN1_Mel_spec.npy'):
    CNN1_Mel_spec = parser(df)
    np.save('CNN1_Mel_spec',CNN1_Mel_spec)  
CNN1_Mel_spec = np.load('D:/Project/Sound/ShipsEar/Feature model/CNN1_Mel_spec.npy', allow_pickle=True)


temp = np.array(CNN1_Mel_spec)
data = temp.transpose()

X_ = data[:, 0]
Y = data[:, 1]
print(X_.shape, Y.shape)
X = np.empty([1956, 128])

for i in range(1956):
    X[i] = (X_[i])
Y = to_categorical(Y)

#Final Data
print(X.shape)
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 1)
X_train = X_train.reshape(1467, 16, 8, 1)
X_test = X_test.reshape(489, 16, 8, 1)
input_dim = (16, 8, 1)

# Creating Keras Model and Testing
model = Sequential()

model.add(Conv2D(64, (3, 3), padding = "same", activation = "tanh", input_shape = input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding = "same", activation = "tanh"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation = "tanh"))
model.add(Dense(5, activation = "softmax"))
model.summary()

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 90, batch_size = 50, validation_data = (X_test, Y_test))

predictions = model.predict(X_test)
score = model.evaluate(X_test, Y_test)
print(score)

preds = np.argmax(predictions, axis = 1)
result = pd.DataFrame(preds)
result.to_csv("ShipsEar_result_CNN1.csv")

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









































































