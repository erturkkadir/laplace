# https://pythontic.com/visualization/signals/spectrogram
# https://stackoverflow.com/questions/42024817/plotting-a-continuous-stream-of-data-with-matplotlib
# https://stackoverflow.com/questions/35344649/reading-input-sound-signal-using-python

# GABOR https://analyticsindiamag.com/hands-on-tutorial-on-visualizing-spectrograms-in-python/

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import sounddevice as sd

fft_size = 1024

freq = 44100
dt = 1/freq
df = freq/(2*fft_size)
duration = int(2 * freq)

t = np.arange(0, 2, dt)
fig, ax1 = plt.subplots(nrows=1)


def update(frame):
    rec = sd.rec(duration, samplerate=freq, channels=2, dtype='float64')
    rec = rec[:, 0]
    sd.wait()
    # ax1.plot(t, rec)
    Pxx, freqs, bins, im = ax1.specgram(rec, NFFT=fft_size, Fs=df, noverlap=600)
    fig.gca().autoscale_view()


animation = FuncAnimation(fig, update, interval=2000)
plt.show()