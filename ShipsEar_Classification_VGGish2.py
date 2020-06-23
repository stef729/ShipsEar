# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 22:06:18 2020

@author: Administrator
"""

# VGGish 


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
from tensorflow.keras.layers import Conv2D, Conv1D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization, LSTM
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.regularizers import l2

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import os
import sys
import librosa
import librosa.display
import glob 
import skimage
import random

sys.path.append("VGGish")
from VGGish import vggish_input
from VGGish import mel_features
from VGGish import vggish_keras
from VGGish import vggish_params
from VGGish import vggish_slim
from VGGish import vggish_postprocess



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


# Relative tolerance of errors in mean and standard deviation of embeddings.
rel_error = 0.1  # Up to 10%

# Paths to downloaded VGGish files.
checkpoint_path = 'D:/Project/Methods test/3-VGGish/2-Antoine-VGGish-Keras/vggish_weights.ckpt'
pca_params_path = 'D:/Project/Methods test/3-VGGish/2-Antoine-VGGish-Keras/vggish_pca_params.npz'

model = vggish_keras.get_vggish_keras()
model.load_weights(checkpoint_path)
num_secs = 5 


def extract_features(path,file_ext="*.wav",bands = 60, frames = 41):
    window_size = 22050*5
    log_specgrams = []

    for i in range(1956):
        file_name = path + current_path[i]
        label = df["ClassID_2"][i]
        sound_clip,s = librosa.load(file_name)
        for (start,end) in windows(sound_clip,window_size):
            if(len(sound_clip[start:end]) == window_size):
                signal = sound_clip[start:end]
                
                input_batch = vggish_input.waveform_to_examples(signal, s)
                np.testing.assert_equal(input_batch.shape,
                                        [num_secs, vggish_params.NUM_FRAMES, vggish_params.NUM_BANDS])
                embedding_batch = model.predict(input_batch[:,:,:,None])  # (num_sec,128)
                pproc = vggish_postprocess.Postprocessor(pca_params_path)
                postprocessed_batch = pproc.postprocess(embedding_batch)
                
                logspec = postprocessed_batch.T.flatten()[:, np.newaxis].T
                log_specgrams.append(logspec)
                labels.append(label)
        print('The file of number %s has done' %i)
    # list to array
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),5,128)
    # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)    
    return np.array(log_specgrams), np.array(labels,dtype = np.int)
                
                
                

    # log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams),bands,frames,1)
    # # 一阶差分， 二维
    # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis = 3)
    # # 二阶差分，三维
    # # features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams)), np.zeros(np.shape(log_specgrams))), axis = 3)
    # for i in range(len(features)):
    #     # 一阶差分
    #     features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])
    #     # 二阶差分
    #     # features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0], width=3, order=1) # 一阶差分
    #     # features[i, :, :, 2] = librosa.feature.delta(features[i, :, :, 0], width=3, order=2) # 二阶差分
    # return np.array(features), np.array(labels,dtype = np.int)
        


# 特征提取
if __name__ == '__main__':
    
    path = 'D:/Project/Sound/ShipsEar/Data_frame/'
    current_path = os.listdir(path)[:-2]
    df = pd.read_csv(path + 'label_DataFrame.csv', names= ["Name", "ClassID", "ClassID_2"] )
    
    feature = []
    labels = []
    
    if not os.path.isfile('D:/Project/Sound/ShipsEar/CNN_feature_VGGish_12c.npy'): 
        data_features,labels = extract_features(path)
        data_labels = one_hot_encode(labels)
        np.save('CNN_feature_VGGish_12c',data_features)
        np.save('CNN_label_VGGish_12c', data_labels)
    else:
        data_features = np.load('D:/Project/Sound/ShipsEar/CNN_feature_VGGish_12c.npy')
        data_labels = np.load('D:/Project/Sound/ShipsEar/CNN_label_VGGish_12c.npy')
    
    
    
# =============================================================================
# # 网络模型和输入
# =============================================================================

X_train, X_test, Y_train, Y_test = train_test_split(data_features, data_labels, random_state = 1)
X_train = X_train.reshape(1464, 128*5)
X_test = X_test.reshape(489, 128*5)
input_dim = (128*5, )

# Creating Keras Model and Testing
model = Sequential()

# model.add(Conv1D(64, 5, padding = "same", activation = "relu", input_shape = input_dim))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3)) 
# model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu", ))
# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.3))
# model.add(Flatten())

# model.add(Dense(1024, activation = "relu", input_shape = input_dim))
# model.add(Dropout(0.3))
model.add(Dense(512, activation = "relu", input_shape = input_dim))
model.add(Dropout(0.3)) 
model.add(BatchNormalization())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.3)) 
# model.add(BatchNormalization())
model.add(Dense(12, activation = "softmax"))
model.summary()

model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size = 256, validation_data = (X_test, Y_test))

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

acc = history_dict['acc']
val_acc = history_dict['val_acc']
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
# class_names = ['Little boat', 'Moto boat', 'Passenger', 'Ocean boat', 'Nature Noise']
cm = skm.confusion_matrix(y_test1, y_pred)
print(cm)
print( skm.classification_report(y_test1,y_pred))
import seaborn as sn
#seaborn for heatmap of representation of the results from evaulation 
df_cm = pd.DataFrame(cm)
sn.set(font_scale=1.4)
plt.subplots(figsize=(10,8))
ax = sn.heatmap(df_cm, annot=True,fmt="g",xticklabels=class_names,yticklabels=class_names,annot_kws={"size": 16},cmap="YlGnBu")











