# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:41:06 2020

@author: CS
"""
# 2020年3月25日 
# ShipsEar数据集分帧，5s截取，形成数据集

from pydub import AudioSegment
import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

# 显示音频时域波形
file_path = 'D:/Project/Sound/ShipsEar/Data/96__Dredger.wav'
fname,ext = os.path.split(file_path)

input_music = AudioSegment.from_wav(file_path)

# 开始截取时间
start = 0 * 1000
end = 60 *1000
number = math.floor((end-start)/(5*1000))
for i in range(number):
     output_music = input_music[(start + i* 5*1000):(start + i*5*1000 + 5*1000)]
     output_music.export('D:/Project/Sound/ShipsEar/Data_frame/96__Dredger_' + str(i+1) + '.wav', format="wav")