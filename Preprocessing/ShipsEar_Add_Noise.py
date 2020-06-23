# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:34:57 2020

@author: CS
"""

import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob 
import os
from tqdm import tqdm
import pickle
import IPython.display as ipd  # To play sound in the notebook

fname = 'D:/Project/Sound/ShipsEar/Data_frame/6__Passengers_19.wav' 
data, sampling_rate = librosa.load(fname)
plt.figure(figsize=(8,4))
librosa.display.waveplot(data, sr=sampling_rate)
ipd.Audio(data, rate=sampling_rate)

# =============================================================================
# # 数据增广方法（Augmentations Methods）
# =============================================================================
# (1) 添加白噪声
def noise(data):
    noise_amp = 0.05*np.random.uniform() * np.amax(data)
    data = data.astype('float') + noise_amp * np.random.normal(size=data.shape[0])
    return data
# 在背景中添加静态噪声。
x = noise(data)
plt.figure(figsize=(8,4))
librosa.display.waveplot(x, sr=sampling_rate)
ipd.Audio(x , rate=sampling_rate)

# (2) 随机移动
def shift(data):
    s_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, s_range)

x = shift(data)
plt.figure(figsize=(8,4))
librosa.display.waveplot(x, sr=sampling_rate)


# (3) 时域拉伸
def stretch(data, rate=0.8):
    data = librosa.effects.time_stretch(data, rate)
    return data

x = stretch(data)
plt.figure(figsize=(8,4))
librosa.display.waveplot(x, sr=sampling_rate)


# (4) 音调变换，该方法强调高音
def pitch(data, sample_rate):
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2*(np.random.uniform())
    data = librosa.effects.pitch_shift(data.astype('float64'),
                                       sample_rate, n_steps=pitch_change,
                                       bins_per_octave=bins_per_octave)
    return data

x = pitch(data, sampling_rate)
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sampling_rate)


# (5) 动态随机变化
def dyn_change(data):
    dyn_change = np.random.uniform(low=-0.5 ,high=7)
    return (data * dyn_change)

x = dyn_change(data)
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sampling_rate)


# (6)加速(压缩)和音调变换
def speedNpitch(data):
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac =1.2 / length_change
    tmp = np.interp(np.arange(0,len(data), speed_fac), np.arange(0, len(data)), data)
    minlen = min(data.shape[0], tmp.shape[0])
    data *= 0
    data[0:minlen] = tmp[0:minlen]
    return data

x = speedNpitch(data)
plt.figure(figsize=(8, 4))
librosa.display.waveplot(x, sr=sampling_rate)