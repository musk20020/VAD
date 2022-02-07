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


def batch_denoise():
    file_path = "/Users/musk/dataset/VADTestSet/noisy_stationary/10dB/*"
    audio_file_list = [tag for tag in iglob(file_path)]
    model, sess = mt.build_model(model_saver="220126")

    for audio_file in tqdm(audio_file_list):
        root_path = "/".join(audio_file.split("/")[:5])
        # dst_file = "/".join("/".join(audio_file.split("/")[-2:]))
        dst_file_path = "{}/enhance_stationary/220126/{}".format(root_path, audio_file.split("/")[-2])
        dst_file = "{}/{}".format(dst_file_path, audio_file.split("/")[-1])

        if not os.path.exists(dst_file_path):
            os.mkdir(dst_file_path)

        denoise(audio_file, dst_file, model, sess)


audio_file = "/Users/musk/dataset/VADTestSet/noisy/0dB/G07FM0210082_h_babble.wav"
# audio_file = "/Users/musk/Downloads/wiener_filtering_in_SNR0_sp01.wav"
dst_file = "/Users/musk/Desktop/test.wav"
model, sess = mt.build_model(model_saver="220126")
denoise(audio_file, dst_file, model, sess)