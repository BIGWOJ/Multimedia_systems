import numpy as np
import matplotlib.pyplot as plt
import cv2
from useful_modules import generate_figures
import soundfile as sf

def quantize(data, bit, m=0, n=1):
    d = 2 ** bit - 1

    if not np.issubdtype(data.dtype, np.floating):
        DataF = data.astype(float)
    else:
        DataF = data

    DataF = (DataF - m) / (n - m)
    DataF = np.round(DataF * d) / d
    DataF = DataF * (n - m) + m

    return DataF.astype(data.dtype)

def decimation(data, factor, fs):
    new_fs = fs // factor
    return data[::factor], new_fs

signal, fs = sf.read('SIN/sin_60Hz.wav')
# print(np.issubdtype(signal.dtype,np.integer))
# print(np.issubdtype(signal.dtype,np.floating))
# generate_figures(signal, fs)
# generate_figures(quantize(signal, 8), fs)
# generate_figures(signal, fs)
decimated_signal, decimated_fs = decimation(signal, 150, fs)
generate_figures(decimated_signal, decimated_fs)
