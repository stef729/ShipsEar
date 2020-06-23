# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:12:30 2020

@author: CS
"""

# 2020年3月25日 
# ShipsEar波形显示

import pandas as pd
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt

# file_path = 'D:/Project/Sound/ShipsEar/Data/28__Trawler.wav'
file_path = 'F:/3-声学所/马力/2016.1海洋背景噪声/data_20160108_045223-045723/data_CH1_20160108_045223-045723.wav'
data, sampling_rate = librosa.load(file_path)
plt.figure(figsize=(20, 10))
plt.subplot(4,2,1)
librosa.display.waveplot(data, sr=sampling_rate)
        
    
    
data1, sampling_rate1 = librosa.load('D:/Project/Sound/ShipsEar/Data_frame/43__Passengers_8.wav')
plt.figure(figsize=(20, 10))
D = librosa.amplitude_to_db(np.abs(librosa.stft(data1)), ref=np.max)
plt.subplot(4,2,1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')


#melspec = librosa.feature.melspectrogram(data, sampling_rate, n_fft=1024, hop_length=512, n_mels=128)
#logmelspec = librosa.power_to_db(melspec)

# LibROSA调用了numpy.lib.stride_tricks.as_strided函数进行分帧（若不指定，帧长默认为2048，帧移默认为512）
arr = librosa.feature.mfcc(y=data, sr=sampling_rate,hop_length=2048)
plt.figure(figsize=(20, 10))
plt.subplot(4,2,1)
librosa.display.specshow(arr, x_axis='time', y_axis='mel')
plt.title('Mel spectrogram 2')
#plt.tight_layout()
#plt.show()
