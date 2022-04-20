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

def wiener_filter(sig, noise_mu, update_filter, fs=16000):

    # calculation parameters
    len_ = 16 * fs // 1000  # frame size in samples
    PERC = 50  # window overlop in percent of frame
    len1 = len_ * PERC // 100  # overlop'length
    len2 = len_ - len1  # window'length - overlop'length

    # setting default parameters
    Thres = 3  # VAD threshold in dB SNRseg
    Expnt = 1.0
    G = 0.8

    # insign = win * x[k - 1:k + len_ - 1]
    # # compute fourier transform of a frame
    # spec = np.fft.fft(insign, nFFT)
    # # compute the magnitude
    # sig = abs(spec)


    # Posterior SNR (noisy SNR)
    SNRpos = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)

    # --- wiener filtering --- #

    # 1 spectral subtraction(Half wave rectification)
    sub_speech = sig ** Expnt - noise_mu ** Expnt
    #   When the pure signal is less than the noise signal power
    diffw = sig ** Expnt - noise_mu ** Expnt

    #   beta negative components
    def find_index(x_list):
        index_list = []
        for i in range(len(x_list)):
            if x_list[i] < 0:
                index_list.append(i)
        return index_list

    # z = find_index(diffw)
    # if len(z) > 0:
    #     sub_speech[z] = 0

    # Priori SNR
    SNRpri = 10 * np.log10(np.linalg.norm(sub_speech, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
    # parameter to deal mel
    mel_max = 15
    mel_0 = (1 + 4 * mel_max) / 5
    # mel_0 = (1 + 8 * mel_max) / 5
    s = 25 / (mel_max - 1)

    # deal mel
    def get_mel(SNR):
        if -5.0 <= SNR <= 20.0:
            a = mel_0 - SNR / s
        else:
            if SNR < -5.0:
                a = mel_max
            if SNR > 20:
                a = 1
        return a

    # setting mel
    mel = get_mel(SNRpri)

    # 2 gain function Gk
    # G_k = (sig ** Expnt - noise_mu ** Expnt) / sig ** Expnt
    G_k = sub_speech ** 2 / (sub_speech ** 2 + mel * noise_mu ** 2)
    # wf_speech = G_k * sub_speech ** (1 / Expnt)
    wf_speech = G_k * sig

    if update_filter:  # Update noise spectrum
        noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  # Smoothing processing noise power spectrum
        noise_mu = noise_temp ** (1 / Expnt)  # New noise amplitude spectrum

    return wf_speech, noise_mu

def denoise_v2(audio_file, dst, model, sess):
    y, _ = librosa.load(audio_file, sr=16000)
    stft = librosa.stft(y, 256, 128, center=False)
    mag = np.abs(stft)
    phase = stft/mag

    predict = model.predict(sess, audio_file=audio_file)[0,:,0]

    noise_mu = np.mean(mag[:, :5], 1)
    update_filter = False
    N = 0
    for i in range(len(predict)):
        if predict[i] < 0.8:
            if i == 0:
                # noise_obs[:, :] = np.expand_dims(mag[:, i], 1)
                continue
            else:
                frame = y[(i-1) * 128:(i-1) * 128 + 256]
                z = np.mean(frame ** 2)
                l = -0.691 + 10 * np.log10(z)
                if l > -40:
                    update_filter=True
        else:
            update_filter=False

        mag[:,i], noise_mu = wiener_filter(mag[:,i], noise_mu, update_filter)


    enhance_stft = mag*phase
    enhance = librosa.istft(enhance_stft, 128, 256)
    # sf.write("/Users/musk/Desktop/test.wav", enhance, 16000)
    sf.write(dst, enhance, 16000)

def denoise(audio_file, dst, model, sess):
    y, _ = librosa.load(audio_file, sr=16000)
    stft = librosa.stft(y, 256, 128, center=False)
    mag = np.abs(stft)
    phase = stft/mag

    predict = model.predict(sess, audio_file=audio_file)[0,:,0]

    obs_t = 5
    # noise_obs = np.zeros([129, obs_t])
    noise_obs = mag[:, :obs_t]

    N = 0
    for i in range(len(predict)):
        if predict[i] < 0.55:
            if i == 0:
                # noise_obs[:, :] = np.expand_dims(mag[:, i], 1)
                continue
            else:
                frame = y[(i-1) * 128:(i-1) * 128 + 256]
                z = np.mean(frame ** 2)
                l = -0.691 + 10 * np.log10(z)
                if l > -40:
                    noise_obs[:, N%obs_t] = mag[:, i]
                    N+=1
                    if N==obs_t:
                        N=0

            # mag[:, i] = 0
        mag[:,i] = mt.wiener_filter(mag[:,i], noise_obs)


    enhance_stft = mag*phase
    enhance = librosa.istft(enhance_stft, 128, 256)
    # sf.write("/Users/musk/Desktop/test.wav", enhance, 16000)
    sf.write(dst, enhance, 16000)


def batch_denoise(file_path):
    # file_path = "/Users/musk/dataset/VADTestSet/noisy_stationary/10dB/*"
    audio_file_list = [tag for tag in iglob(file_path)]
    model, sess = mt.build_model(model_saver="220126")

    for audio_file in tqdm(audio_file_list):
        root_path = "/".join(audio_file.split("/")[:5])
        # dst_file = "/".join("/".join(audio_file.split("/")[-2:]))
        dst_file_path = "{}/enhance/220126-2/{}".format(root_path, audio_file.split("/")[-2])
        dst_file = "{}/{}".format(dst_file_path, audio_file.split("/")[-1])

        if not os.path.exists(root_path+"/enhance/"):
            os.mkdir(root_path+"/enhance/")
        if not os.path.exists(root_path + "/enhance/220126-2/"):
            os.mkdir(root_path + "/enhance/220126-2/")
        if not os.path.exists(dst_file_path):
            os.mkdir(dst_file_path)

        denoise_v2(audio_file, dst_file, model, sess)

if __name__=="__main__":
    audio_file = "/Users/musk/dataset/VADTestSet/noisy/-2dB/G07FM0210082_h_babble.wav"
    # # audio_file = '/Users/musk/Userfile/compal/助聽器2.0/hearing aid recording/noisy_16k.wav'
    # # audio_file = "/Users/musk/Downloads/wiener_filtering_in_SNR0_sp01.wav"
    # audio_file = "/Users/musk/Desktop/test.wav"
    dst_file = "/Users/musk/Desktop/enhance.wav"
    model, sess = mt.build_model(model_saver="220126")
    denoise_v2(audio_file, dst_file, model, sess)
    # denoise(audio_file, dst_file, model, sess)

    # file_path = "/Users/musk/Desktop/NR_Test/NR0/*"
    # batch_denoise(file_path)

