import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

sampFreq, sound = wavfile.read('Police whistle.wav')

sound = sound / 2.0**15

length_in_s = sound.shape[0] / sampFreq
print(sound[0])
print("Time: ", length_in_s)
print(sound.dtype, sampFreq)
print(sound.shape)

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

signal = sound[:,0]
sample = []
subSample = []
for i in range (len(sound)):
    if abs(sound[i] > 0.1).all():
        subSample.append(sound[i])  #(index, amplitude) (sound[i] has 2 elements, 1 for left channel, 1 for right channel)
    elif abs(sound[i] <= 0.1).all():
        sample.append(subSample)
        subSample = []


# for take_sample in sample:
#     if len(take_sample) != 0:
#         signal_FFT = np.array(take_sample)

print(len(sample[0]))
fft_spectrum = np.fft.rfft(signal)
freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
print(len(fft_spectrum))

fft_spectrum_abs = np.abs(fft_spectrum)

f = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.xlabel("time, s [left channel]")
plt.ylabel("signal, relative units")

plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("time, s [right channel]")
plt.ylabel("signal, relative units")
plt.tight_layout()

g = plt.figure(2)
plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")

plt.show()
