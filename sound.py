import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment, silence

whistle_freq = 2000

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (9, 7)

FileName = "mix2.wav"
sampFreq, sound = wavfile.read(FileName)

sound = sound / 2.0**15

myaudio = AudioSegment.from_wav(FileName)

nonsilence = silence.detect_nonsilent(myaudio, min_silence_len=55, silence_thresh=-50)

nonsilence = [((start/1000),(stop/1000)) for start,stop in nonsilence] #convert to sec

singal_sample  = [] #list of interval, in which have nonsilence sound
amp = []
#sound: array of amplitudes
print(nonsilence)
for time_interval in nonsilence:
    print(time_interval)
    for cnt in range(int(time_interval[0]*sampFreq), int(time_interval[1]*sampFreq)):
        amp.append(sound[cnt])
    singal_sample.append(np.array(amp))
    amp = []


print("singal_sample[i] ", singal_sample[0])
print("Length of nonsilence general array: ", len(singal_sample))
# print("Length of nonsilence array 1 ", len(singal_sample[0]))
# print("Length of nonsilence array 2 ", len(singal_sample[1]))

length_in_s = sound.shape[0] / sampFreq
#print(type(sound[0]))
print("Time: ", length_in_s)
print(sound.dtype, sampFreq)
print(sound.shape)

time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s

print("sound ",sound)
signal = sound[:,0]
print("signal ", signal)
signal_list = []

for sig in singal_sample:
    signal_list.append(sig[:,0])
    
# print("signal_array[0]", signal_list[0])
# print("signal_array[1]", signal_list[1])

fft_spectrum = np.fft.rfft(signal)
fft_spectrum_list = []

freq = np.fft.rfftfreq(signal.size, d=1./sampFreq)
freq_list = []

print(len(fft_spectrum))

fft_spectrum_abs = np.abs(fft_spectrum)
fft_spectrum_abs_list = []

for i in range(len(singal_sample)):
    fft_spectrum_list.append(np.fft.rfft(signal_list[i]))
    freq_list.append(np.fft.rfftfreq(signal_list[i].size, d=1./sampFreq))
    fft_spectrum_abs_list.append(np.abs(fft_spectrum_list[i]))



f = plt.figure(1)
plt.subplot(2,1,1)
plt.plot(time, sound[:,0], 'r')
plt.xlabel("time, s [left channel]")
plt.ylabel("signal, relative units")

plt.subplot(2,1,2)
plt.plot(time, sound[:,1], 'b')
plt.xlabel("time, s [right channel]")
plt.ylabel("signal, relative units")
plt.tight_layout()

g = plt.figure(2)
plt.subplot(3,1,1)
plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")

for i in range(len(singal_sample)):
    plt.subplot(len(singal_sample) +1 ,1,i+2)
    plt.plot(freq_list[i], fft_spectrum_abs_list[i])
    if any(arr_element >= whistle_freq for arr_element in freq_list[i]):
        plt.xlabel("frequency, Hz  " + str(nonsilence[i]) + " -----> Whistle")
    else:
        plt.xlabel("frequency, Hz  " + str(nonsilence[i]) + " -----> Non whistle")
    #plt.xlabel("frequency, Hz  " + str(nonsilence[i]))
    plt.ylabel("Amplitude, units")

plt.show()