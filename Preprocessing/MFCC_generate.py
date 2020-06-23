# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:06:34 2020

@author: CS
"""
# 批量生成MFCC特征矩阵，并存储

import pandas as pd
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt
import scipy


path = 'D:/Project/Sound/ShipsEar/Data_frame/'
current_path = os.listdir(path)[:-2]
num_MFCC = len(current_path)
mfcc = []
all_mfcc = []
mel_spec= []
all_mel_spec = []

for i in range(num_MFCC-1):
    file_path = path + current_path[i]
    data, sampling_rate = librosa.load(file_path)
    # n_mfcc维度48，fft默认2048，hop_length为帧移
    arr = librosa.feature.mfcc(y=data, sr=sampling_rate,n_mfcc=48, hop_length=512)
    # mel_spectrogram
    melspec = librosa.feature.melspectrogram(data, n_mels = 48)
    logspec = librosa.power_to_db(melspec)

    
#    # MFCC生成三维矩阵
#    mfcc = np.append(mfcc, arr)
#    dim = arr.shape
#    all_mfcc = mfcc.reshape((i+1),dim[0],dim[1])
#    np.save('all_mfcc', all_mfcc)
#    print('The Number of %d  has done' %i)
    
    
    # MFCC生成flatten二维矩阵
    mfcc = np.append(mfcc, arr)
    all_mfcc.append(mfcc)
    #all_mfcc = np.row_stack((all_mfcc, mfcc))
    mfcc = []
    
    # mel_spec
    mel_spec = np.append(mel_spec,logspec)
    all_mel_spec.append(mel_spec)
    mel_spec = []
    
    print ('The number of %d has done' %i)
    
    
    
# np.save('all_mfcc_list', all_mfcc)

all_mfcc_matrix = np.array(all_mfcc)
np.save('all_mfcc_matrix', all_mfcc_matrix)

all_mel_spec = np.array(all_mel_spec)
np.save('all_mel_spec', all_mel_spec)



    
 