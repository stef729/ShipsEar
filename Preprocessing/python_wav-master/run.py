# -*- coding:utf8 -*-

from wavTools import *

wav_path = "./voice_files/demo.wav"
pcm_path = "./voice_files/demo.pcm"
wav_path2 = "./voice_files/test.wav"

# 音频切割的文件路径
main_wav_path = "./part_voice_files/12.wav"
part_wav_path = "./part_voice_files/ms_part_voice.wav"
second_part_wav_path = "./part_voice_files/second_part_wav_path.wav"
minute_part_wav_path = "./part_voice_files/minute_part_wav_path.wav"

# 获取音频信息
ret = wav_infos(wav_path)
print(ret)

# 读取音频文件内容
ret = read_wav(wav_path)
print(ret)

# 获取音频时长(单位秒)
ret = get_wav_time(wav_path)
print(ret)


# 音频切片，获取部分音频 时间的单位是毫秒
start_time = 13950
end_time = 15200
get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path)


# 音频切片，获取部分音频 时间的单位是秒
start_time = 35
end_time = 38
get_second_part_wav(main_wav_path, start_time, end_time, second_part_wav_path)

# 音频切片，获取部分音频 时间的单位是分钟和秒 样式：0:12
start_time = "0:35"
end_time = "0:38"
get_minute_part_wav(main_wav_path, start_time, end_time, minute_part_wav_path)


# wav文件转为pcm文件
wav_to_pcm(wav_path, pcm_path)

# pcm文件转为wav文件
pcm_to_wav(pcm_path, wav_path2)

# 音频对应的波形图
wav_waveform(wav_path)



