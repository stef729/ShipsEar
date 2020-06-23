# -*- coding: utf-8 -*-
"""
Created on Fri May 22 15:56:18 2020

@author: CS
"""

import pandas as pd
import numpy as np
import os
import librosa.display
import matplotlib.pyplot as plt

file_path = 'D:/Project/Sound/ShipsEar/Data_frame/43__Passengers_8.wav'
y, sr = librosa.load(file_path)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.waveplot(y, sr=sr)

#  Linear-frequency power spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

# on a logarithmic scale
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

# use a CQT scale
CQT = librosa.amplitude_to_db(np.abs(librosa.cqt(y, sr=sr)), ref=np.max)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

# Draw a chromagram with pitch classes
C = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(C, y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')


# 绘制时间标记
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log power spectrogram')

# tempogram with BPM节拍
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
Tgram = librosa.feature.tempogram(y=y, sr=sr)
librosa.display.specshow(Tgram, x_axis='time', y_axis='tempo')
plt.colorbar()
plt.title('Tempogram')
plt.tight_layout()
plt.show()

# MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()


# melspectrogram 梅尔频谱
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
plt.figure(figsize=(12, 8))
plt.subplot(4, 2, 1)
S_dB = librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(S_dB , x_axis='time', y_axis='mel', sr=sr, fmax=8000)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()

















