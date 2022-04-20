import numpy as np
import librosa
import audio_processing as ap
import matplotlib
import matplotlib.pyplot as plt
import soundfile as sf
import model_test as mt
from glob import iglob
import os
from tqdm import tqdm
import time

audio_file = "/Users/musk/dataset/VADTestSet/noisy/10dB/G07FM0210082_h_babble.wav"
# audio_file = "/Users/musk/Desktop/test.wav"
dst_file = "/Users/musk/Desktop/enhance.wav"

y, _ = librosa.load(audio_file, 16000)

def peak_finding_inTime(y_tmp, delay=512):
    # y_tmp = y[240000-512:240000]
    # delay = 512

    # t0 = time.time()
    n = np.zeros(delay)
    for d in range(delay):
        # r = 0
        # m = 0
        # for i in range(d, delay):
        #     r += y_tmp[i]*y_tmp[i-d]
        #     m += (y_tmp[i]**2+y_tmp[i-d]**2)
        r = np.sum(y_tmp[d:]*y_tmp[:512-d])
        m = np.sum(y_tmp[d:]**2+y_tmp[:512-d]**2)
        n[d] = 2*r/m
    # t1 = time.time()

    d = n[1:]*n[:-1]
    s = n[1:]-n[:-1]
    peak_tmp = 0
    peak_ind_tmp = 0
    peak_can = False
    peak_ind = []
    peak = []
    count = 0
    for i in range(len(d)):
        if peak_can == True and n[i+1]>peak_tmp:
            peak_tmp = n[i+1]
            peak_ind_tmp = i+1

        if d[i] < 0 and s[i]>0:
            peak_can = True

        if d[i] < 0 and s[i]<0:
            peak_can = False
            peak_ind.append(peak_ind_tmp)
            peak.append(peak_tmp)
            peak_tmp = 0
            peak_ind_tmp = 0
            # count += 1

        # if count == 2:
        #     # t2 = time.time()
        #     # print(t1 - t0)
        #     # print(t2 - t1)
        #     return peak_ind[-1]


    # print(peak_ind)
    # print(peak)
    #
    # plt.figure()
    # plt.plot(n)
    # plt.show()

    return peak_ind[1]

##### peak finding #####
stft = librosa.stft(y, 512, 256, 512)
mag = np.abs(stft)
pitch = np.zeros(len(mag[0]))

# y_tmp = y[40000:40512]
# peak = peak_finding_inTime(y_tmp)

for i in tqdm(range(256, len(y)-256, 256)):
    y_tmp = y[i-256:i+256]
    peak = peak_finding_inTime(y_tmp)
    # f0 = 8000/257*peak
    pitch[i//256] = peak

plt.figure(1)
plt.imshow(mag[::-1])
plt.figure(2)
plt.plot(pitch)
plt.show()

def peak_finding_inFreq(mag):
    peak = np.zeros(len(mag[0]))
    for n in range(len(mag[0])):
        mag_tmp = mag[:, n]
        e = np.zeros(30)
        for d in range(30):
            e[d] = np.mean(mag_tmp[:257-d]*mag_tmp[d:])
        peak[n] = np.argmax(e[3:10])+3
        mag[int(peak[n]), n] = 50

    plt.figure()
    plt.imshow(mag[::-1])
    plt.show()

def peak_finding_inFreq_v2(mag):
    peak = np.zeros(len(mag[0]))
    for n in range(len(mag[0])):
        mag_tmp = mag[:, n] #739
        mag_4k = mag_tmp[::2]
        mag_2k = mag_4k[::2]
        # mag_1k = mag_2k[::2]
        # mag_500 = mag_1k[::2]

        l = len(mag_2k)
        e = mag_tmp[:l]*mag_4k[:l]*mag_2k[:l] #*mag_1k[:l]*mag_500
        peak[n] = np.argmax(e[3:10])+3
        mag[int(peak[n]), n] = 50

    plt.figure(1)
    plt.plot(peak)
    plt.figure(2)
    plt.imshow(mag[::-1])
    plt.show()



def peak_denoise(mag):
    peak = np.zeros(len(mag[0]))
    for n in range(len(mag[0])):
        mag_tmp = mag[:, n]
        e = np.zeros(257)
        for d in range(0, 257):
            e[d] = np.mean(mag_tmp[:257-d]*mag_tmp[d:])
        i = np.argmin(e[:10])
        e[:i] = 0
        # peak[n] = np.argmax(e[3:10])+3
        e /= np.max(e)
        mag[:, n]*=np.sqrt(e)

    # plt.figure(2)
    # plt.imshow(mag[::-1])
    # plt.show()
    return mag

##### peak denoise #####
# stft = librosa.stft(y, 512, 256, 512)
# mag = np.abs(stft)
# phase = stft/mag
# enhance_stft = peak_denoise(mag)*phase
# enhance = librosa.istft(enhance_stft, 256, 512)
# sf.write("/Users/musk/Desktop/test.wav", enhance, 16000)
