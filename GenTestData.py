import librosa
import numpy as np
import scipy
from glob import iglob
import time
from tqdm.auto import tqdm
import soundfile as sf
import os

def gen_noisy(clean_file, noise_file, save_dir, SNR, sr_clean, sr_noise):
    if not os.path.exists('/'.join(save_dir.split('/')[:-2])):
        os.mkdir('/'.join(save_dir.split('/')[:-2]))
    if not os.path.exists('/'.join(save_dir.split('/')[:-1])):
        os.mkdir('/'.join(save_dir.split('/')[:-1]))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    clean_name = clean_file.split( '/' )[-1].split( '.' )[0]
    noise_name = noise_file.split( '/' )[-2]
    y_clean, sr_clean = librosa.load( clean_file, sr_clean, mono=True )

    clean_pwr = sum( abs( y_clean ) ** 2 ) / len( y_clean )
    y_noise, sr_noise = librosa.load( noise_file, sr_noise, mono=True )

    if len( y_noise ) < len( y_clean ):
        tmp = (len( y_clean ) // len( y_noise )) + 1
        y_noise = np.array( [x for j in [y_noise] * tmp for x in j] )
        y_noise = y_noise[:len( y_clean )]
    else:
        y_noise = y_noise[:len( y_clean )]

    y_noise -= np.mean(y_noise)

    noise_variance = clean_pwr / (10 ** (SNR / 10))
    noise_pwr = sum(abs(y_noise) ** 2) / len(y_noise)
    noise = np.sqrt( noise_variance ) * y_noise / np.sqrt(noise_pwr)
    # noise = np.sqrt(noise_variance) * y_noise / np.std(y_noise)

    y_noisy = y_clean + noise

    maxv = np.iinfo( np.int16 ).max

    save_name = '{}_{}.wav'.format( clean_name, noise_name )
    sf.write('/'.join( [save_dir, save_name] ), y_noisy/ np.max( np.abs( y_noisy )), 16000)

    # librosa.output.write_wav(
    #     '/'.join( [save_dir, save_name] ), (y_noisy / np.max( np.abs( y_noisy ) ) * maxv).astype( np.int16 ), 16000 )

if __name__ == "__main__" :
    noise_path = '/Users/musk/dataset/noise_train_16k(ESC-50)/*'
    noise_list = [tag for tag in iglob(noise_path)]
    clean_file_f = '/Users/musk/dataset/TCC300_rename/Dev/G07FM0210/G07FM0210082.wav'
    clean_file_m = '/Users/musk/dataset/TCC300_rename/Dev/G08MM0210/G08MM0210012.wav'
    save_dir = '/Users/musk/dataset/VADTestSet/noisy/10dB/'
    SNR = 10
    sr_clean = sr_noise = 16000
    for noise_type in noise_list:
        noise = [tag for tag in iglob(noise_type+'/*')][3]
        gen_noisy(clean_file_f, noise, save_dir, SNR, sr_clean, sr_noise)
        gen_noisy(clean_file_m, noise, save_dir, SNR, sr_clean, sr_noise)