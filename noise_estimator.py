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
import math

def cal_gain(xi, vk, order=4):
    intgral = np.zeros(vk.shape)
    for r in range(1, order+1):
        intgral += (-vk)**r/(math.factorial(r)*r)
    a = np.log(vk)
    b = intgral
    gain = xi/(1+xi)*np.exp(-(0.57721566+intgral+np.log(vk))/2)
    # gain = xi / (1 + xi) * np.exp(-intgral / 2)
    return gain

def array_max(array, n):
    # a = np.array([1, 2, 3, 4, 5])
    compare = lambda x: np.max((x, n))
    result = np.array(list(map(compare, array)))
    return result

def get_window(n):
    window = np.zeros(n)
    for i in range(n):
        window[i] = 0.54-0.46*np.cos(2*np.pi*i/(n-1))
    return window

## load data ##
clean_file = "/Users/musk/dataset/TCC300_rename/Dev/G07FM0210/G07FM0210082.wav"
audio_file = "/Users/musk/dataset/VADTestSet/noisy/-2dB/G07FM0210082_h_babble.wav"
dst_file = "/Users/musk/Desktop/test.wav"

clean, _ = librosa.load(clean_file, 16000)
noisy, _ = librosa.load(audio_file, 16000)

## stft parameter ##
window_size = 256
hop_length = int(window_size/2)

## stft ##
stft = librosa.stft(noisy, window_size, hop_length, window_size)
mag = np.abs(stft)
phase = stft/mag
freq_bins, frames = mag.shape

stft_clean = librosa.stft(clean, window_size, hop_length, window_size)
mag_clean = np.abs(stft_clean)
lambda_x = np.mean(mag_clean**2, axis=1)

## algorithm parameter ##
enhance_mag = np.zeros((freq_bins, frames))
alpha = 0.92 # priori SNR weighting of past frame
alpha_s = 0.9 # magnitude smooth weighting of past frame
alpha_d = 0.5 #0.85
obs_frame_num = 5 # observed frame number of S_hist
B_min = 1.66
garma_0 = 4.6
garma_1 = 3
zeta_0 = 1.67
beta = 1.47
freq_smooth_window_size = 10
post_SNR = np.zeros(freq_bins)
S_hist = np.zeros([freq_bins, obs_frame_num])
S_tilde_hist = np.zeros([freq_bins, obs_frame_num])
I = np.zeros(freq_bins) # speech present in each freq bin
S_tilde = np.zeros(freq_bins) # magnitude of speech present in each freq bin
Sf_tilde = np.zeros(freq_bins)
q_hat = np.zeros(freq_bins)

hamming_window = get_window(freq_smooth_window_size) # frequency smooth window
hamming_window /= np.sum(hamming_window) # summation of window = 1



test = np.zeros(mag.shape)

for t in range(obs_frame_num, frames):
    if t == obs_frame_num:
        enhance_mag[:, :obs_frame_num] = mag[:, :obs_frame_num]
        lambda_d_hat = mag[:,obs_frame_num]**2
        lambda_d_bar = mag[:, obs_frame_num] ** 2
        garma_last = mag[:, obs_frame_num] ** 2 / lambda_d_hat
        Sf = np.convolve(mag[:,obs_frame_num]**2, hamming_window, mode="same")
        S = Sf
        S_tilde = Sf
        Sf_tilde = Sf
        S_hist[:, :obs_frame_num] = mag[:, :obs_frame_num]
        S_tilde_hist[:, :obs_frame_num] = mag[:, :obs_frame_num]

    # for f in range(freq_bins):
    garma_current = mag[:,t]**2/lambda_d_hat
    xi = lambda_x/(lambda_d_hat+np.finfo(np.float).eps)
    vk = garma_current*xi/(1+xi)
    gain = cal_gain(xi, vk)
    xi_hat = alpha*gain**2*garma_last+(1-alpha)*array_max(garma_current-1, 0)
    garma_last = garma_current

    Sf = np.convolve(mag[:, t] ** 2, hamming_window, mode="same")
    S = alpha_s*S+(1-alpha_s)*Sf
    S_hist[:, :obs_frame_num-1] = S_hist[:, 1:]
    S_hist[:, obs_frame_num-1] = S
    S_min = np.amin(S_hist, axis=1)

    garma_min = mag[:, t]**2/(B_min*S_min)
    zeta = S/(B_min*S_min)
    # freq_bins_VAD = lambda r, z: 1 if (r < garma_0 and z < zeta_0) else 0
    for i in range(freq_bins):
        if garma_min[i] < garma_0 and zeta[i] < zeta_0:
            I[i] = 1
        else:
            I[i] = 0

    # test[:, t] = I

    mag_speech_present = mag[:, t]**2*I
    Smooth_I = np.convolve(I, hamming_window, mode="same")
    Smooth_mag = np.convolve(mag_speech_present, hamming_window, mode="same")
    for i in range(freq_bins):
        if Smooth_I[i] != 0:
            Sf_tilde[i] = Smooth_mag[i]/Smooth_I[i]
        else:
            Sf_tilde[i] = S_tilde[i]
    S_tilde = alpha_s*S_tilde + (1-alpha_s)*Sf_tilde
    S_tilde_hist[:, :obs_frame_num - 1] = S_tilde_hist[:, 1:]
    S_tilde_hist[:, obs_frame_num - 1] = S_tilde
    S_tilde_min = np.amin(S_tilde_hist, axis=1)

    garma_tilde_min = mag[:, t] ** 2 / (B_min * S_tilde_min)
    zeta_tilde = S / (B_min * S_tilde_min)
    # freq_bins_VAD = lambda r, z: 1 if (r < garma_0 and z < zeta_0) else 0
    for i in range(freq_bins):
        if garma_tilde_min[i] <= 1 and zeta_tilde[i] < zeta_0:
            q_hat[i] = 1
        elif garma_tilde_min[i] <= garma_1 and garma_tilde_min[i] > 1 and zeta_tilde[i] < zeta_0:
            q_hat[i] = (garma_1-garma_tilde_min[i])/(garma_1-1)
        else:
            q_hat[i] = 0

    # test[:, t] = q_hat

    S_present_probavility = 1/(1+q_hat/(1-q_hat+np.finfo(np.float).eps)*(1+xi_hat)*np.exp(-vk))
    alpha_d_tilde = alpha_d+(1-alpha_d)*S_present_probavility

    lambda_d_bar = alpha_d_tilde*lambda_d_bar + (1-alpha_d_tilde)*mag[:, t]**2

    lambda_d_hat = beta*lambda_d_bar
    lambda_x = xi_hat*lambda_d_hat

    enhance_mag[:, t] = mag[:, t]-np.sqrt(lambda_d_hat)


enhance_stft = enhance_mag*phase
enhance = librosa.istft(enhance_stft, 128, 256)
sf.write(dst_file, enhance, 16000)

# plt.figure(1)
# plt.imshow(mag[::-1])
# plt.figure(2)
# plt.imshow(test[::-1])
# plt.show()

