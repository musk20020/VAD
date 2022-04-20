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
import model_test as mt

def filter(noisy_mag, noisy_phase, noise_mag_mu, noise_phase_pre, update_filter, arccos_pre):

    freq = np.linspace(0, 8000, 129) # Hz
    noise_angle_pre = np.angle(noise_phase_pre)
    noise_angle = ((8.0/1000.0*freq)%1)*(2*np.pi)+noise_angle_pre
    noise_phase = np.cos(noise_angle)+1j*np.sin(noise_angle)
    noisy_angle = np.angle(noisy_phase)

    if update_filter:
        # signal_mag = np.zeros(129)
        signal_mag = noisy_mag*0.5
        signal_phase = noisy_phase
        arccos = arccos_pre
        noise_phase = noisy_phase
        noise_mag_mu = 0.1*noise_mag_mu + 0.9*noisy_mag
        ######################################################
    else:
        c1 = noise_mag_mu**2-2*noise_mag_mu**2*np.sin(noise_angle-noisy_angle)**2-\
            noisy_mag**2
        c2 = -2*noise_mag_mu**2*np.sin(noise_angle-noisy_angle)*np.cos(noise_angle-noisy_angle)
        c3 = noise_mag_mu**2*np.sin(noise_angle-noisy_angle)**2

        a1 = (c1 + 2 * c3)
        a2 = (np.sqrt(c1**2+c2**2)+np.finfo(float).eps)
        for i in range(len(a1)):
            if a1[i]>a2[i]:
                # a1[i] = a2[i]
                a1[i] = 0
        arccos = np.arccos(a1/a2)
        # for i in range(1, len(a1)):
        #     if arccos[i]-arccos_pre[i] > 0: #TODO 把前一時刻的arccos帶入這個function
        #         arccos[i] *= -1

        arctan = np.arctan(c2 / (c1 + np.finfo(float).eps))
        for i in range(len(arctan)): # for every frequency bin
            # if c1[i] < 0 and c2[i] > 0:
            #     arctan[i] += np.pi
            # elif c1[i] < 0 and c2[i] < 0:
            #     arctan[i] += np.pi
            if c1[i] < 0:
                arctan[i] += np.pi


        # b = np.arctan(c2/(c1+np.finfo(float).eps))
        # signal_angle = (np.arccos((c1+2*c3)/(np.sqrt(c1**2+c2**2)+np.finfo(float).eps))-np.arctan(c2/(c1+np.finfo(float).eps)))/2+noisy_angle
        signal_angle = (arccos - arctan) / 2 + noisy_angle

        signal_mag = np.abs(-noise_mag_mu*np.sin(noise_angle-noisy_angle)/(np.sin(signal_angle-noisy_angle)+np.finfo(float).eps))

        # test = signal_mag-noisy_mag

        # noise_phase = noisy_phase
        signal_phase = np.cos(signal_angle)+1j*np.sin(signal_angle)


    return signal_mag, signal_phase, noise_mag_mu, noise_phase, arccos


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

def denoise_v2(audio_file, dst, model, sess):
    y, _ = librosa.load(audio_file, sr=16000)
    stft = librosa.stft(y, 256, 128, center=False)
    mag = np.abs(stft)
    phase = stft/mag

    enhance_mag = np.zeros(np.shape(mag))

    predict = model.predict(sess, audio_file=audio_file)[0,:,0]

    confidence_threshold = 0.55
    arccos_pre = np.zeros(129)
    noise_mu = np.mean(mag[:, :5], 1)
    noise_phase = phase[:, 0]
    update_filter = False
    N = 0
    for i in range(1, len(predict)):
        if predict[i] < confidence_threshold:
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

        enhance_mag[:, i], phase[:,i], noise_mu, noise_phase, arccos_pre = filter(mag[:, i], phase[:,i], noise_mu, noise_phase, update_filter, arccos_pre)

    enhance_stft = enhance_mag*phase
    enhance = librosa.istft(enhance_stft, 128, 256)
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
    # audio_file = "/Users/musk/dataset/VADTestSet/noisy/10dB/G07FM0210082_h_babble.wav"
    audio_file = "/Users/musk/Desktop/test.wav"
    dst_file = "/Users/musk/Desktop/enhance.wav"
    model, sess = mt.build_model(model_saver="220126")
    denoise_v2(audio_file, dst_file, model, sess)