import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack

def task_1():
    data, fs = sf.read('SOUND_INTRO/sound1.wav', dtype='float32')
    # sd.play(data, fs)
    # status = sd.wait()

    sound_L = data[:, 0]
    sound_R = data[:, 1]
    sound_mix = (sound_L + sound_R) / 2

    # sf.write('SOUND_INTRO/sound1_L.wav', sound_L, fs)
    # sf.write('SOUND_INTRO/sound1_R.wav', sound_R, fs)
    # sf.write('SOUND_INTRO/sound1_mix.wav', sound_mix, fs)
    #
    # x = np.arange(0, data.shape[0]) / fs
    #
    # plt.subplot(2,1,1)
    # plt.plot(x, data[:,0])
    #
    # plt.subplot(2,1,2)
    # plt.plot(x, data[:,1])
    # plt.show()
    #

    data, fs = sf.read('SIN/sin_440Hz.wav', dtype=np.int32)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, data.shape[0]) / fs, data)

    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(data)
    plt.plot(np.arange(0, fs, 1.0 * fs / (yf.size)), np.abs(yf))
    plt.show()

    # Decibel scale
    fsize = 2**8
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, data.shape[0]) / fs, data)
    plt.subplot(2, 1, 2)
    yf = scipy.fftpack.fft(data, fsize)
    plt.plot(np.arange(0, fs / 2, fs / fsize), 20 * np.log10(np.abs(yf[:fsize // 2])))
    plt.show()

def task_2(signal, fs, time_margin=[0, 0.02]):
    time = np.arange(0, len(signal)) / fs

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.xlabel('[s]')
    plt.ylabel('Amplitude')
    plt.xlim(time_margin)
    plt.title('Signal')

    yf = scipy.fftpack.fft(signal)
    xf = np.linspace(0, fs / 2, len(yf) // 2)

    plt.subplot(2, 1, 2)
    plt.plot(xf, 20 * np.log10(np.abs(yf[:len(yf) // 2])))
    plt.xlabel('[Hz]')
    plt.ylabel('[dB]')
    plt.title('1/2 Spectrum')
    plt.tight_layout()
    plt.show()

# task_1()(

data, fs = sf.read('SIN/sin_440Hz.wav', dtype='float32')
task_2(data, fs, )