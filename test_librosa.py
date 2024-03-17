import librosa

audio_data = 'D:/PycharmProjects/Temp1/data/audio_split/MELD/dev_video/dia9_utt5.wav'
x, sr = librosa.load(audio_data)
print(type(x), type(sr)) #<class 'numpy.ndarray'> <class 'int'>

librosa.load(audio_data, sr=)