import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from IPython.display import Audio
from numpy.fft import fft, ifft

freq = 44100
sgn_len = 4096

Fs, data = read("hello.wav")
data = data[:, 0]

for i in range(0, 1, 1):
    tmp = data[i*sgn_len:((i+1)*sgn_len)-1]
    for s in range(-20, 20, 1):     # laplace spaces
        lp_data = [tmp[i] * np.exp(-s*i/freq) for i in tmp]
        plt.plot(lp_data)
        plt.plot(tmp)

        fft_data = fft(tmp)
        fft_lp = fft(lp_data)

        plt.plot(abs(fft_data)**2)
        plt.plot(abs(fft_lp)**2)
    plt.show()

