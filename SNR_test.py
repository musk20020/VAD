import os
import librosa
import numpy as np
from glob import iglob
import subprocess
import soundfile as sf


def pesq(clean_path, target_path, sr_clean=16000, sr_noisy=16000, time=0):
    target, _ = librosa.load( target_path, sr=sr_noisy, mono=True )
    clean, _ = librosa.load( clean_path, sr=sr_clean, mono=True )
    if sr_noisy != 16000:
        target = librosa.resample( target, sr_noisy, 16000 )
    if sr_clean != 16000:
        clean = librosa.resample( clean, sr_clean, 16000 )
    # if time != 0:
    #     if time >0:
    #         diff = int(16000*time)
    #         target = target[diff:]
    #     else:
    #         diff = int( 16000 * -time )
    #         clean = clean[diff:]
    #     clean = clean[:len(target)]

    pesq_app = "/Users/musk/Userfile/pesq/P862/Software/source/PESQ"
    sf.write('/Users/musk/Userfile/pesq/clean.wav', clean, 16000)
    sf.write('/Users/musk/Userfile/pesq/degrade.wav', target, 16000)

    # maxv = np.iinfo( np.int16 ).max
    # librosa.output.write_wav(
    #     '/Users/musk/Userfile/pesq/clean.wav', (clean*maxv).astype( np.int16 ), 16000 )
    # librosa.output.write_wav(
    #     '/Users/musk/Userfile/pesq/degrade.wav', (target * maxv).astype( np.int16 ), 16000 )
    a = subprocess.run( [pesq_app, '+16000', '/Users/musk/Userfile/pesq/clean.wav', '/Users/musk/Userfile/pesq/degrade.wav'] )
    print("break point")

def segmental_SNR(clean_path, target_path, sr=16000, window_size=256):
    target, _ = librosa.load( target_path, sr=sr )
    max_SNR = -10

    for i in range(0, 1):
        # target = target[448:]

        clean_ori, _ = librosa.load( clean_path, sr=sr )
        clean = clean_ori[i:]
        if len(clean)>len(target):
            clean = clean[:len(target)]
        else:
            target = target[:len( clean )]

        # clean = clean[256*1000:]
        # target = target[256*1000:]

        power_t = np.mean( target ** 2 )
        power_c = np.mean( clean ** 2 )

        # target = target / power_t * power_c
        # target = target / np.sqrt( power_t ) * np.sqrt( power_c )
        SNR = 10*np.log10(power_c/(np.mean((target-clean)**2)))

        if SNR>max_SNR:
            max_SNR = SNR
            index = i
        # index=0

    return SNR, index


def segmental_SNR_v1(clean_path, degrade_path, sr=16000, window_size=256):
    degrade, _ = librosa.load( degrade_path, sr=16000 )
    clean, _ = librosa.load( clean_path, sr=16000 )
    power_t = np.sum( target ** 2 )
    power_c = np.sum( clean ** 2 )
    target = target / np.sqrt( power_t ) * np.sqrt( power_c )
    SNR = 10*np.log(power_c/np.sum((target-clean)**2))

    return SNR


if __name__ == '__main__':
    # print("====== test SNR ======")
    # test_data_route = '/Users/musk/dataset/VADTestSet/noisy_stationary/10dB/*'
    # # test_data_route = '/Users/musk/dataset/VADTestSet/enhance_stationary/220126/10dB/*'
    # test_list = [tag for tag in iglob(test_data_route+'.wav')]
    # test_list = np.sort(test_list)
    #
    # count = 0
    # SNR_tmp = 0
    # for filename in test_list:
    #     print(filename.split("/")[-1])
    #     ##############################    data parser    ##############################
    #     # ######  Aurora4  ######
    #     # clean_name = filename.split( '_' )[-1].split( '.' )[0]
    #     # speaker = clean_name[:3]
    #     # clean_path = '/Volumes/Transcend/Download/aurora4/validate_clean'
    #     # clean_file = '{}/{}_16k/{}..wav'.format(clean_path, speaker, clean_name)
    #     # #######################
    #     #
    #     # ######  TCC300 split ######
    #     # clean_name = filename.split( '_' )[-1].split( '.' )[0]
    #     # speaker = clean_name[:9]
    #     # # clean_path = '/Volumes/TRANSCEND 1/TCC300/Dev/Female/'
    #     # clean_path = '/Users/musk/dataset/TCC300_split/Dev/'
    #     # clean_file = '{}/{}/{}.wav'.format( clean_path, speaker, clean_name )
    #     # #######################
    #
    #     # ######  TCC300 rename  ######
    #     clean_name = filename.split('/')[-1].split('_')[0]
    #     speaker = clean_name[:9]
    #     # clean_path = '/Volumes/TRANSCEND 1/TCC300/Dev/Female/'
    #     clean_path = '/Users/musk/dataset/TCC300_rename/Dev/'
    #     clean_file = '{}/{}/{}.wav'.format(clean_path, speaker, clean_name)
    #     # #######################
    #
    #     ######  pixel3 record  ######
    #     # clean_file = '/Volumes/TRANSCEND 1/TCC300_split/Dev/Female/G03FM0110/G03FM011002004.wav'
    #     #######################
    #
    #
    #     ##############################    criteria test    ##############################
    #     ######  SNR test  ######
    #     # SNR_result, index = segmental_SNR(clean_file, filename)
    #     # SNR_tmp += SNR_result
    #     # count += 1
    #     # print(SNR_result, index)
    #
    #     ######  PESQ test  ######
    #     # pesq(clean_file, filename)
    #
    # # print('avg_SNR : {}'.format(SNR_tmp/count))

    clean_file = '/Users/musk/Desktop/0208/clean.wav'
    # filename = '/Users/musk/Desktop/0208/enhance_ori.wav'
    # filename = '/Users/musk/Desktop/0208/AWS_enhance.wav'
    filename = '/Users/musk/Desktop/0208/noisy.wav'
    pesq(clean_file, filename)
    # SNR_result, index = segmental_SNR(clean_file, filename)
    # print(SNR_result, index)


