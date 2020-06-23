# python_wav
对音频文件的处理：音频信息，读取内容，获取时长，切割音频，pcm与wav互转

文章简介：公众号-Python疯子
https://mp.weixin.qq.com/s/Kw_n3RgYfZCn_0ZOJpaxHg

###### 获取音频信息
ret = wav_infos(wav_path)
print(ret)

######  读取音频文件内容
ret = read_wav(wav_path)
print(ret)

######  获取音频时长(单位秒)
ret = get_wav_time(wav_path)
print(ret)


######  音频切片，获取部分音频 时间的单位是毫秒
start_time = 13950
end_time = 15200
get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path)


######  音频切片，获取部分音频 时间的单位是秒
start_time = 35
end_time = 38
get_second_part_wav(main_wav_path, start_time, end_time, second_part_wav_path)

######  音频切片，获取部分音频 时间的单位是分钟和秒 样式：0:12
start_time = "0:35"
end_time = "0:38"
get_minute_part_wav(main_wav_path, start_time, end_time, minute_part_wav_path)


######  wav文件转为pcm文件
wav_to_pcm(wav_path, pcm_path)

######  pcm文件转为wav文件
pcm_to_wav(pcm_path, wav_path2)
