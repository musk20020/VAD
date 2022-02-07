
import librosa
import numpy as np
import scipy
import os
import h5py
from glob import iglob
from shutil import copy2
from os.path import join
import xlrd
import random
# import pyroomacoustics as pra
#from pysndfx import AudioEffectsChain
import scipy.io.wavfile as wavfile
import warnings
import time
from scipy.optimize import least_squares
import audio_processing as ap

warnings.filterwarnings('ignore')

#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

epsilon = np.finfo(float).eps

def get_embedding(file_name, excel_index, gender=False):
    sheet = xlrd.open_workbook( file_name + '.xls' ).sheet_by_index( 0 )
    label = [sheet.cell( tag, excel_index ).value for tag in range( 0, sheet.nrows )]
    embedding = np.load( file_name + '.npy' )
    if gender:
        gender_label = [sheet.cell( tag, -1 ).value for tag in range( 0, sheet.nrows )]

    #dis_embed = np.zeros([len(embedding), len(embedding)])
    #for i in range(len(embedding)):
    #    for j in range(len(embedding)):
    #        if i == j:
    #            dis_embed[i][j] = 0
    #        else:
    #            dis_embed[i][j] = 1/np.sqrt(np.sum(np.square(embedding[i]-embedding[j])))
    #    dis_embed[i] /= np.sum(dis_embed[i])
    if gender:
        return embedding, label, gender_label 
    return embedding, label

def get_dist_table(file_name):
    embedding = np.load( file_name + '.npy' )
    userNum = len(embedding)
    sumArray = np.zeros((userNum, userNum),dtype = np.float32)
    for i in range(userNum):
        for j in range(userNum):
            if i==j:
                sumArray[i, j]=0
            else:
                sub = embedding[i] - embedding[j]
                #sum_ = np.sqrt(np.sum(sub**2))
                sum_ = np.sqrt(np.sum(sub**2))
                sumArray[i, j] = 1/sum_
    return sumArray

def np_batch(data1, batch_size, data_len):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            #yield data1[n_start:]
            n_start = 0
            n_end = batch_size
            yield data1[n_start:n_end]
        else:
            yield data1[n_start:n_end]
        n_start = n_end
        n_end += batch_size

def np_REG_batch(data1, data2, batch_size, data_len, data3=None, data4=None):
    n_start = 0
    n_end = batch_size
    l = data_len
    while True:
        if n_end >= l:
            #yield data1[n_start:]
            #yield data2[n_start:]
            #if data3 is not None:
            #    yield data3[n_start:]
            #if data4 is not None:
            #    yield data4[n_start:]
            #    #yield data4[n_start:]
            n_start = 0
            n_end = batch_size
        
        yield data1[n_start:n_end]
        yield data2[n_start:n_end]
        if data3 is not None:
            yield data3[n_start:n_end]
        if data4 is not None:
            yield data4[n_start:n_end]
            #yield data4[n_start:n_end]
        n_start = n_end
        n_end += batch_size


def search_wav(data_path):
    file_list = []
    #for filename in iglob('{}/-5*.wav'.format(data_path), recursive=True):
    #    file_list.append(str(filename))
    for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
        file_list.append(str(filename))
    return file_list

def split_list(alist, wanted_parts=20):
    length = len(alist)
    return [alist[i * length // wanted_parts: (i + 1) * length // wanted_parts]
            for i in range(wanted_parts)]

def room_sim(y, sr):
    corners = np.array([[0,0], [0,6], [6,6], [6,0]]).T
    room = pra.Room.from_corners(corners, fs=16000, max_order=1)
    room.extrude(3.)
    source = [np.random.random()*5+0.5, np.random.random()*5+0.5, 0.5]
    mic = [np.random.random()*5+0.5, np.random.random()*5+0.5, 0.5]
    mic_arr = [[mic[0], mic[0]+0.1], [mic[1], mic[1]], [mic[2], mic[2]]]
    room.add_source(source, signal=y)
    room.add_microphone_array(pra.MicrophoneArray(mic_arr, 16000))
    room.image_source_model(use_libroom=True)
    room.simulate()
    output = room.mic_array.signals[0, :]/np.max(np.abs(room.mic_array.signals[0, :]))
    try:
        dis = len(output)-len(y)
        output = output[dis:]
    except:
        output = np.concatenate((output, output[len(output)-len(y):]))
    return output

def EQ(init_freq=60, max_freq=8000):
    def generate_data(t, a1, a2, a3, a4, a5, a6):
        y = a1 + a2*t + a3*t**2 + a4*t**3+ a5*t**4 + a6*t**5
        return y

    def fun(x, t, y):
        return (x[0] + x[1]*t + x[2]*t**2 + x[3]*t**3+ x[4]*t**4 + x[5]*t**5) - y


    x0 = np.ones( 6 )

    center_freq = []
    freq_res = []
    freq = init_freq
    while freq < max_freq:
        center_freq.append(freq)
        freq *= 2**(1/3)

    matter_point = np.ceil(len(center_freq)/5)

    low_bound = np.array(np.random.randint(-5, 0)).reshape([1, -1])
    mid_bound = np.random.randint(-3, 3, size=3).reshape([1, -1])
    up_bound = np.array(np.random.randint(-5, 5)).reshape([1, -1])
    bound = np.concatenate((low_bound, mid_bound, up_bound), axis=1)

    # print(1)
    for i in range(len(center_freq)):
        index = int(np.round(i/matter_point))
        freq_res.append(np.random.randint(-2, 2) + bound[0, index])

    res_lsq = least_squares(fun, x0, args=(np.log10(np.array(center_freq)), np.array(freq_res)))

    #o = generate_data(np.log10(np.array(center_freq)), *res_lsq.x)

    return res_lsq

def audio_eq(audio_path, EQ, sr=16000, window=256, y=None):
    def generate_data(t, a1, a2, a3, a4, a5, a6):
        y = a1 + a2*t + a3*t**2 + a4*t**3+ a5*t**4 + a6*t**5
        return y

    max_freq = sr/2
    freq_resolution = window/2+1
    # band_width = max_freq/freq_resolution
    band = np.linspace(1, max_freq, freq_resolution)
    eq = generate_data(np.log10(np.array(band)), *EQ.x)
    eq[0] = 0
    eq = 10**(eq/10)


    stft = librosa.stft(y, 256, 128, 256)

    eq_stft = np.multiply(stft, np.reshape(eq, [-1, 1]))

    eq_y = librosa.istft(eq_stft, 128, 256)
    eq_y = librosa.util.fix_length(eq_y, len(y), mode='edge')

    # output_file = '/Users/edwin/Desktop/test.wav'
    # maxv = np.iinfo( np.int16 ).max
    # librosa.output.write_wav(
    #     output_file, (eq_y / np.max( np.abs( eq_y) ) * maxv).astype( np.int16 ), 16000 )
    return eq_y

def audio2spec(y, forward_backward=None, SEQUENCE=None, norm=True, hop_length=256, frame_num=None, mel_freq=False, angle=False,
               under4k_dim=0, gender=None, threshold=False, featureMap=None):
 
    NUM_FFT = np.int(hop_length*2)

    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=NUM_FFT,
                     window=scipy.signal.hamming,
                     center=False)

    if angle:
        Sxx = np.angle(D)
    elif under4k_dim:
        D = D[:under4k_dim]
        Sxx = abs(D)
    else:
        Sxx = abs(D)

    if mel_freq:
        Sxx = librosa.feature.melspectrogram(S=Sxx, sr=16000, n_fft=NUM_FFT, hop_length=hop_length,
                                           n_mels=24, fmax=8000, power=1)
        # print(Sxx.shape)
    if norm:
        # Sxx_mean = np.mean( Sxx, 1, keepdims=True)
        # Sxx_var = np.sqrt(np.var(Sxx, 1, keepdims=True))
        # Sxx_r = np.zeros(Sxx.shape[])

        mean = np.zeros(Sxx.shape[0])
        var = np.zeros(Sxx.shape[0])
        if featureMap is not None:
            Df = librosa.stft(featureMap,
                              n_fft=NUM_FFT,
                              hop_length=hop_length,
                              win_length=NUM_FFT,
                              window=scipy.signal.hamming)
            Sxxf = abs(Df)

            for i in range(Sxx.shape[1]):
                if i == 0:
                    mean = Sxx[:,i]
                    var = np.sqrt((Sxx[:,i]-mean)**2)
                else:
                    mean = 0.001*Sxx[:,i]+0.999*mean
                    var = np.sqrt(0.001*((Sxx[:,i]-mean)**2)+0.999*(var**2))
                    Sxx[:,i] = (Sxx[:,i]-mean)/var
                    Sxxf[:,i] = (Sxxf[:,i]-mean)/var
            return Sxx.T, Sxxf.T
        else:
            for i in range(Sxx.shape[1]):
                mean = 0.001*Sxx[:,i]+0.999*mean
                var = np.sqrt(0.001*((Sxx[:,i]-mean)**2)+0.999*(var**2))
                Sxx[:,i] = (Sxx[:,i]-mean)/var
            return Sxx

    else:
        Sxx_r = np.array(Sxx)
    Sxx_r = np.array(Sxx_r).T
    #Sxx_r = Sxx_r[NUM_FRAME+1:-NUM_FRAME]
    shape = Sxx_r.shape

    if threshold:
        threshold = np.max(Sxx_r)/200
        mask = Sxx_r > threshold
        Sxx_r *= Sxx_r*mask

    if SEQUENCE:
        return Sxx_r.reshape(shape[0], 1, shape[1])
    else:
        return Sxx_r

def phasespec(y, forward_backward=None, SEQUENCE=None, norm=True, hop_length=256, frame_num=None, mel_freq=False, angle=False):
    if frame_num is None:
        NUM_FRAME = 3  # number of backward frame and forward frame
    else:
        NUM_FRAME = frame_num
    NUM_FFT = 512
    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=512,
                     window=scipy.signal.hann)
    if angle:
        Sxx = np.angle(D)
    else:
        #D_re = Sxx.real
        #D_imag = Sxx.imag
        Sxx = D

    Sxx_r = np.array(Sxx)
    idx = 0
    if forward_backward:
        Sxx_r = Sxx_r.T
        return_data = np.empty(
            (3000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT / 2) + 1), dtype=complex)
        frames, dim = Sxx_r.shape

        for num, data in enumerate(Sxx_r):
            idx_start = idx - NUM_FRAME
            idx_end = idx + NUM_FRAME
            if idx_start < 0:
                null = np.zeros((-idx_start, dim))
                tmp = np.concatenate((null, Sxx_r[0:idx_end + 1]), axis=0)
            elif idx_end > frames - 1:
                null = np.zeros((idx_end - frames + 1, dim))
                tmp = np.concatenate((Sxx_r[idx_start:], null), axis=0)
            else:
                tmp = Sxx_r[idx_start:idx_end + 1]

            return_data[idx] = tmp
            idx += 1
        shape = return_data.shape
        if SEQUENCE:
            return return_data[:idx]
        else:
            return return_data.reshape(shape[0], shape[1] * shape[2])[:idx]

    else:
        Sxx_r = np.array(Sxx_r).T
        shape = Sxx_r.shape
        if SEQUENCE:
            return Sxx_r.reshape(shape[0], 1, shape[1])
        else:
            return Sxx_r


def phase_sensitive_spec(y_noisy, y_clean, hop_length=None,
                                 frame_num=None):
    if frame_num is None:
        NUM_FRAME = 3  # number of backward frame and forward frame
    else:
        NUM_FRAME = frame_num
    NUM_FFT = 1024
    
    D_noisy = librosa.stft(y_noisy,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=1024,
                     window=scipy.signal.hann)

    D_clean = librosa.stft(y_clean,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=1024,
                     window=scipy.signal.hann)

    D_noisy = D_noisy[:257]
    D_clean = D_clean[:257]

    spec_noisy = abs(D_noisy)
    Re_sy = spec_noisy*((D_clean/D_noisy).real)

    Sxx_clean = np.array(Re_sy).T

    Sxx_noisy = np.array(spec_noisy)
    Sxx_noisy = Sxx_noisy.T
    return_data = np.empty(
                (3000, np.int32(NUM_FRAME * 2) + 1, np.int32(NUM_FFT/2 / 2) + 1))
    frames, dim = Sxx_noisy.shape
    idx = 0
    for num, data in enumerate(Sxx_noisy):
        idx_start = idx - NUM_FRAME
        idx_end = idx + NUM_FRAME
        if idx_start < 0:
            null = np.zeros((-idx_start, dim))
            tmp = np.concatenate((null, Sxx_noisy[0:idx_end + 1]), axis=0)
        elif idx_end > frames - 1:
            null = np.zeros((idx_end - frames + 1, dim))
            tmp = np.concatenate((Sxx_noisy[idx_start:], null), axis=0)
        else:
            tmp = Sxx_noisy[idx_start:idx_end + 1]

        return_data[idx] = tmp
        idx += 1
    shape = return_data.shape
    return_data = return_data.reshape(shape[0], shape[1] * shape[2])[:idx]
    return return_data, Sxx_clean, Sxx_noisy

def ToWaveformData(y, window_size):
    #y, sr = librosa.load(wavfile, sr, mono=True)
    diss_y = len(y)%window_size
    #print(diss_y)
    #if diss_y > 1:
    y = y[diss_y:]
    hop_length = int(window_size/2)
    frames = int(len(y)/hop_length-1)
    Sxx_r = np.empty([frames, window_size])
    Sxx_r_even = np.array(y).reshape([-1, window_size])
    Sxx_r_odd = np.array(y[hop_length:-hop_length]).reshape([-1, window_size])
    Sxx_r[::2] = Sxx_r_even
    Sxx_r[1:-1:2] = Sxx_r_odd
    #print(Sxx_r.shape)
    return Sxx_r

def copy_file(input_file, output_file):
    copy2(input_file, output_file)

def scipy_load_wav(path):
    try:
        r, y = wavfile.read( path )
        maxy = np.max((y.max(), -y.min()))
        datatype = y.dtype
        shape = y.shape
        if datatype != np.float32:
            y = y.astype(np.float32)
        if len(shape)>1:
            #print(shape)
            y = y[:, 1]/maxy
        
        y = y-np.mean(y)
        y = y/maxy
        return y
    except:
        print('{} : {}'.format('Load wav fail', path))

def check_audio(y, speaker1='speaker1', speaker2='speaker2'):
    rechoose_flag = False
    #print('len y : {}, len(y) < 131*256 : {}'.format(len(y), len(y) < 131*256))
    if len(y) < 100*512:
        rechoose_flag = True
    if not np.isfinite(y).all():
        rechoose_flag = True
    if speaker1==speaker2:
        rechoose_flag = True
    return rechoose_flag

def gen_noisy(target_file, noise_file, SNR, preEnhance=False, getNoise=False):
    def check_min_length(y, min_length):
        len_y = len(y)
        if len_y < min_length:
            tmp = (min_length // len_y) + 1
            #y = np.array( [x for j in [y] * tmp for x in j] )
            y = np.tile(y, tmp)
            y = y[:min_length]
        return y

    def same_length(fix_len_data, flex_len_data, mode='target'):
        if mode=='target':
            if len( flex_len_data ) < len(fix_len_data):
                #tmp = (len( fix_len_data ) // len( flex_len_data)) + 1
                #flex_len_data = np.array( [x for j in [flex_len_data] * tmp for x in j] )
                flex_len_data = check_min_length(flex_len_data, len(fix_len_data))
                flex_len_data = flex_len_data[:len( fix_len_data )]
            else:
                flex_len_data = flex_len_data[:len( fix_len_data )]
            return flex_len_data
        else:
            if len( flex_len_data ) < len(fix_len_data):
                #tmp = (len( fix_len_data ) // len( flex_len_data)) + 1
                #flex_len_data = np.array( [x for j in [flex_len_data] * tmp for x in j] )
                #flex_len_data = flex_len_data[:len( fix_len_data )]
                fix_len_data = fix_len_data[:len( flex_len_data )]
            else:
                flex_len_data = flex_len_data[:len( fix_len_data )]
            return fix_len_data, flex_len_data
    
    y_target = scipy_load_wav(target_file)
    y_target = check_min_length(y_target, 160000)
    target_pwr = sum( abs( y_target ) ** 2 ) / len( y_target )
    
    noise = scipy_load_wav(noise_file)
    noise = same_length( y_target, noise )
    noise -= np.mean( noise )
    #y_target_var = target_pwr / (10 ** (SNR / 10))
    noise_target_var = target_pwr / (10 ** (SNR / 10))
    noise = np.sqrt( noise_target_var ) * noise / np.std( noise )
    
    prob = np.random.rand()
    # if prob < 0.01:
    #     y_noisy = y_target
    # else:
    #     y_noisy = y_target+noise

    y_noisy = y_target + noise
    max_y = np.max(np.abs(y_noisy))
    y_target /= max_y
    y_noisy /= max_y
    noise /= max_y

    if preEnhance:
        y_target[:-1] = y_target[:-1]-0.7*y_target[1:]
        y_noisy[:-1] = y_noisy[:-1]-0.7*y_noisy[1:]

    if getNoise:
        return y_target, y_noisy, noise
    else:
        return y_target, y_noisy


def _preprocessing(y, y1=None):
    window = 256
    max = 0.5
    tar_dB = -17
    l = tar_dB
    Gi = 0
    for i in range( 0, len( y ), window ):
        du_time = 1  # dutime = 256/16000/0.003
        du_time_weight = window / 16000 / du_time
        z = np.mean( (y[i:i + window]) ** 2 )
        l = l * (1 - du_time_weight) + (-0.691 + 10 * np.log10( z )) * du_time_weight
        e = l - tar_dB
        if e + Gi < -0.5:
            Gi_1 = Gi + 0.6
        elif e + Gi > 0.5:
            Gi_1 = Gi - 1
        else:
            Gi_1 = Gi

        GdB = np.array( [(j * (Gi_1 - Gi) / len( y[i:i + window] ) + Gi) for j in range( len( y[i:i + window] ) )] )
        Glin = 10 ** (GdB / 20)
        y[i:i + window] *= Glin
        if y1 is not None:
            y1[i:i + window] *= Glin

        Gi = Gi_1

    if y1 is not None:
        return y, y1
    else:
        return y

def wiener_filter(mag_src, mag_noise):

    N_FFT = 512
    frameshift = 256
    max_att = 15
    max_att_init = 1
    att_lin = 10 ** (-max_att / 20.0)
    overest = 2

    # S_bb = np.mean(np.square(mag_noise), axis=1)
    # S_bb = overest * S_bb

    filtered_src = np.zeros(np.shape(mag_src))


    for k in range(0, mag_noise.shape[0]):
        if k < 120:
            att_lin = 10 ** (-max_att+(max_att-max_att_init)*k/120 / 20.0)
        else:
            att_lin = 10 ** (-max_att / 20.0)
        H = np.maximum(att_lin, 1 - np.divide(np.square(mag_noise[k, :])*overest, np.square(mag_src[k, :])))
        filtered_src[k, :] = np.multiply(H, mag_src[k, :])

    return filtered_src

def _gen_denoise_training_data_runtime(clean_file_list, noise_file_list, config, train=True, num=None):
    def wgn(x, maximum=0.1):
        noise = np.random.random_sample(len(x))-0.5
        return x+noise*maximum

    clean_file = clean_file_list[num]
    noise_path = noise_file_list[num]
    noise_list = [tag for tag in iglob(noise_path+'/*')]
    noise_file = noise_list[np.random.randint(len(noise_list))]
    speaker = clean_file.split( '/' )[-2] # TCC300
    #speaker = clean_file.split( '/' )[-2].split('_')[0] # Aurora4 speaker
    #speaker = clean_file.split('/')[-2] # aidatatang speaker
    #speaker = clean_file.split( '/' )[-1].split('.')[0] # mozilla speaker

    if train:
        SNR_noise = np.random.randint(-3, 3) #3~6
        voice_path = config.voice_path
    else:
        SNR_noise = 0
        voice_path = config.dev_voice_path

    if config.wienerFilter:
        y_clean, y_noisy, y_noise = gen_noisy(clean_file, noise_file, SNR_noise, config.preEnhance, config.wienerFilter)
        noise_spec = audio2spec(y_noise, forward_backward=False, SEQUENCE=False, norm=False,
                                hop_length=config.hop_length)
    else:
        y_clean, y_noisy = gen_noisy(clean_file, noise_file, SNR_noise, config.preEnhance, config.wienerFilter)
    #white_noise = wgn(y_noisy)

    ##########################      data preprocessing      #############################
    if config.CRNN:
        noisy_spec = audio2spec( y_noisy, forward_backward=False, SEQUENCE=False, norm=False, hop_length=config.hop_length, under4k_dim=config.under4k)
        clean_spec = audio2spec( y_clean, forward_backward=False, SEQUENCE=False, norm=False, hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False)

        ###   stoi shape  ###
        shift = random.randint(0, 10)
        time_step = noisy_spec.shape
        clean_dim = clean_spec.shape
        residual = (time_step[0] - shift) % config.stoi_correlation_time
        noisy_spec = noisy_spec[shift:time_step[0] - residual]
        noisy_spec = noisy_spec.reshape( [-1, config.stoi_correlation_time, time_step[1]] )
        
        clean_spec = clean_spec[shift:time_step[0] - residual]
        clean_spec = clean_spec.reshape( [-1, config.stoi_correlation_time, clean_dim[1]] )
        #clean_spec = clean_spec[:,3:-3,:]

        # mask = clean_spec/noisy_spec
        # mask[mask>1] = 1

        noisy_spec = np.expand_dims(noisy_spec.transpose([0, 2, 1]), axis=3)
        

    elif config.RNN:
        noisy_spec = audio2spec( y_noisy, forward_backward=False, SEQUENCE=False, norm=False, hop_length=config.hop_length, under4k_dim=config.under4k)
        clean_spec = audio2spec( y_clean, forward_backward=False, SEQUENCE=False, norm=False, hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False)

        ###   stoi shape  ###
        shift = random.randint(0, 10)
        time_step = noisy_spec.shape
        clean_dim = clean_spec.shape
        residual = (time_step[0] - shift) % config.stoi_correlation_time
        noisy_spec = noisy_spec[shift:time_step[0] - residual]
        noisy_spec = noisy_spec.reshape( [-1, config.stoi_correlation_time, time_step[1]] )
        
        clean_spec = clean_spec[shift:time_step[0] - residual]
        clean_spec = clean_spec.reshape( [-1, config.stoi_correlation_time, clean_dim[1]] )
    else:
        noisy_spec = audio2spec( y_noisy, forward_backward=True, SEQUENCE=False, norm=False,
                                 hop_length=config.hop_length,
                                 frame_num=config.input_frame_num, mel_freq=config.mel_freq )
        clean_spec = audio2spec( y_clean, forward_backward=False, SEQUENCE=False, norm=False, hop_length=config.hop_length,
                                 frame_num=config.input_frame_num )


    ##########################      data return      ##############################   
    if config.norm:
        noisy_spec_norm, clean_spec_norm = audio2spec( y_noisy, forward_backward=False, SEQUENCE=False, norm=True, hop_length=config.hop_length, under4k_dim=config.under4k, featureMap=y_clean)
        # noisy_spec_norm = audio2spec(y_noisy, forward_backward=False, SEQUENCE=False, norm=True,
        #                                               hop_length=config.hop_length, under4k_dim=config.under4k)
        noisy_spec_norm = noisy_spec_norm[shift:time_step[0] - residual]
        noisy_spec_norm = noisy_spec_norm.reshape( [-1, config.stoi_correlation_time, time_step[1]] )
        noisy_spec_norm = np.expand_dims(noisy_spec_norm.transpose([0,2,1]), axis=3)

        clean_spec_norm = clean_spec_norm[shift:time_step[0] - residual]
        clean_spec_norm = clean_spec_norm.reshape([-1, config.stoi_correlation_time, time_step[1]])
        clean_spec_norm = np.expand_dims(clean_spec_norm.transpose([0, 2, 1]), axis=3)

        # b,h,w,c = noisy_spec_norm.shape
        # noisy_spec_norm_expend = np.zeros([b, h+66, w+6, c])
        # noisy_spec_norm_expend[:,33:-33,6:,:] = noisy_spec_norm
        # return noisy_spec, clean_spec, noisy_spec_norm
        return noisy_spec, clean_spec, noisy_spec_norm, clean_spec_norm
        # return noisy_spec[:,:,:-4,:], clean_spec[:,:-4,:], noisy_spec_norm
    else: 
        return noisy_spec, clean_spec

def _gen_audio(target_file, noise_file, SNR, preEnhance=False):
    def check_min_length(y, min_length):
        len_y = len(y)
        if len_y < min_length:
            tmp = (min_length // len_y) + 1
            # y = np.array( [x for j in [y] * tmp for x in j] )
            y = np.tile(y, tmp)
            y = y[:min_length]
        return y

    def same_length(fix_len_data, flex_len_data, mode='target'):
        if mode == 'target':
            if len(flex_len_data) < len(fix_len_data):
                # tmp = (len( fix_len_data ) // len( flex_len_data)) + 1
                # flex_len_data = np.array( [x for j in [flex_len_data] * tmp for x in j] )
                flex_len_data = check_min_length(flex_len_data, len(fix_len_data))
                flex_len_data = flex_len_data[:len(fix_len_data)]
            else:
                flex_len_data = flex_len_data[:len(fix_len_data)]
            return flex_len_data
        else:
            if len(flex_len_data) < len(fix_len_data):
                # tmp = (len( fix_len_data ) // len( flex_len_data)) + 1
                # flex_len_data = np.array( [x for j in [flex_len_data] * tmp for x in j] )
                # flex_len_data = flex_len_data[:len( fix_len_data )]
                fix_len_data = fix_len_data[:len(flex_len_data)]
            else:
                flex_len_data = flex_len_data[:len(fix_len_data)]
            return fix_len_data, flex_len_data

    y_target = scipy_load_wav(target_file)
    y_target = check_min_length(y_target, 160000)
    target_pwr = sum(abs(y_target) ** 2) / len(y_target)

    noise = scipy_load_wav(noise_file)
    noise = same_length(y_target, noise)
    noise -= np.mean(noise)
    # y_target_var = target_pwr / (10 ** (SNR / 10))
    noise_target_var = target_pwr / (10 ** (SNR / 10))
    noise = np.sqrt(noise_target_var) * noise / np.std(noise)

    return y_target, noise

def _stream_norm(noisySpec, cleanSpec=None):
    mean = np.zeros((257))
    var = np.zeros((257))

    for i in range(noisySpec.shape[0]):
        mean = 0.001 * noisySpec[i, :] + 0.999 * mean
        var = np.sqrt(0.001 * ((noisySpec[i, :] - mean) ** 2) + 0.999 * (var ** 2))
        noisySpec[i, :] = (noisySpec[i, :] - mean) / (var+np.finfo(np.float32).min)
        if cleanSpec is not None:
            cleanSpec[i, :] = (cleanSpec[i, :] - mean) / (var+np.finfo(np.float32).min)
            return noisySpec, cleanSpec
    return noisySpec


def _gen_denoise_training_data_runtime_v2(clean_file_list, noise_file_list, config, train=True, num=None):
    def wgn(x, maximum=0.1):
        noise = np.random.random_sample(len(x)) - 0.5
        return x + noise * maximum

    clean_file = clean_file_list[num]
    noise_path = noise_file_list[num]
    noise_list = [tag for tag in iglob(noise_path + '/*')]
    noise_file = noise_list[np.random.randint(len(noise_list))]

    if train:
        SNR_noise = np.random.randint(-5, 5)  # 3~6
        voice_path = config.voice_path
    else:
        SNR_noise = 0
        voice_path = config.dev_voice_path

    y_clean, y_noisy = ap.gen_training_data(clean_file, noise_file, SNR_noise)
    clean_is_not_finite = len(np.where(np.isfinite(y_clean)==False)[0])>0
    noisy_is_not_finite = len(np.where(np.isfinite(y_noisy) == False)[0]) > 0
    while clean_is_not_finite or noisy_is_not_finite :
        clean_file = clean_file_list[np.random.randint(0, len(clean_file_list))]
        clean_file = noise_list[np.random.randint(0, len(noise_list))]
        y_clean, y_noisy = ap.gen_training_data(clean_file, noise_file, SNR_noise)
        clean_is_not_finite = len(np.where(np.isfinite(y_clean) == False)[0]) > 0
        noisy_is_not_finite = len(np.where(np.isfinite(y_noisy) == False)[0]) > 0


    ##########################      data preprocessing      #############################
    noisy_spec = audio2spec(y_noisy, forward_backward=False, SEQUENCE=False, norm=False,
                            hop_length=config.hop_length, under4k_dim=config.under4k)
    clean_spec = audio2spec(y_clean, forward_backward=False, SEQUENCE=False, norm=False,
                            hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False)
    # noisy_spec_norm = audio2spec(y_noisy_norm, forward_backward=False, SEQUENCE=False, norm=False,
    #                         hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False)
    # clean_spec_norm = audio2spec(y_clean_norm, forward_backward=False, SEQUENCE=False, norm=False,
    #                         hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False)
    noisy_spec_norm, clean_spec_norm = audio2spec(y_noisy, forward_backward=False, SEQUENCE=False, norm=True,
                            hop_length=config.hop_length, under4k_dim=config.under4k, threshold=False, featureMap=y_clean)

    ###   stoi shape  ###
    shift = random.randint(0, 10)
    time_step = noisy_spec.shape
    clean_dim = clean_spec.shape
    residual = (time_step[0] - shift) % config.stoi_correlation_time

    noisy_spec = noisy_spec[shift:time_step[0] - residual]
    noisy_spec = noisy_spec.reshape([-1, config.stoi_correlation_time, time_step[1]])

    clean_spec = clean_spec[shift:time_step[0] - residual]
    clean_spec = clean_spec.reshape([-1, config.stoi_correlation_time, clean_dim[1]])

    noisy_spec_norm = noisy_spec_norm[shift:time_step[0] - residual]
    noisy_spec_norm = noisy_spec_norm.reshape([-1, config.stoi_correlation_time, clean_dim[1]])

    clean_spec_norm = clean_spec_norm[shift:time_step[0] - residual]
    clean_spec_norm = clean_spec_norm.reshape([-1, config.stoi_correlation_time, clean_dim[1]])

    noisy_spec_norm = np.expand_dims(noisy_spec_norm.transpose([0, 2, 1]), axis=3)
    clean_spec_norm = np.expand_dims(clean_spec_norm.transpose([0, 2, 1]), axis=3)

    ##########################      data return      ##############################
    return noisy_spec, clean_spec, noisy_spec_norm, clean_spec_norm

def _gen_VAD_training_data_runtime(clean_file_list, noise_file_list, config, train=True, num=None):
    def wgn(x, maximum=0.1):
        noise = np.random.random_sample(len(x)) - 0.5
        return x + noise * maximum

    clean_file = clean_file_list[num]
    noise_path = noise_file_list[num]
    noise_list = [tag for tag in iglob(noise_path + '/*')]
    noise_file = noise_list[np.random.randint(len(noise_list))]

    if train:
        SNR_noise = np.random.randint(-5, 5)  # 3~6
        voice_path = config.voice_path
    else:
        SNR_noise = -2
        voice_path = config.dev_voice_path

    y_clean, y_noisy = ap.gen_training_data(clean_file, noise_file, SNR_noise)
    clean_is_not_finite = len(np.where(np.isfinite(y_clean)==False)[0])>0
    noisy_is_not_finite = len(np.where(np.isfinite(y_noisy) == False)[0]) > 0
    while clean_is_not_finite or noisy_is_not_finite :
        clean_file = clean_file_list[np.random.randint(0, len(clean_file_list))]
        clean_file = noise_list[np.random.randint(0, len(noise_list))]
        y_clean, y_noisy = ap.gen_training_data(clean_file, noise_file, SNR_noise)
        clean_is_not_finite = len(np.where(np.isfinite(y_clean) == False)[0]) > 0
        noisy_is_not_finite = len(np.where(np.isfinite(y_noisy) == False)[0]) > 0

    ###  energy VAD  ###
    _, e = ap.cal_frame_loudness(y_clean, config.hop_length * 2, config.hop_length)

    ##########################      data preprocessing      #############################
    # noisy_spec = audio2spec(y_noisy, forward_backward=False, SEQUENCE=False, norm=False,
    #                         hop_length=config.hop_length, under4k_dim=config.under4k, mel_freq=True)
    # clean_spec = audio2spec(y_clean, forward_backward=False, SEQUENCE=False, norm=False,
    #                         hop_length=config.hop_length, under4k_dim=config.under4k, mel_freq=True)
    # clean_spec = clean_spec[:len(e)]
    # noisy_spec = noisy_spec[:len(e)]

    noisy_spec_norm = audio2spec(y_noisy, forward_backward=False, SEQUENCE=False, norm=True,
                            hop_length=config.hop_length, under4k_dim=config.under4k, mel_freq=True)

    noisy_spec_norm = noisy_spec_norm[500:len(e)] # 500 for streaming normalize, 100 of 500 for VAD energy init
    e = (e>0)[500:]


    ###  stoi shape  ###
    shift = random.randint(0, 10)
    time_step = noisy_spec_norm.shape
    # clean_dim = clean_spec.shape
    residual = (time_step[0] - shift) % config.stoi_correlation_time

    # noisy_spec = noisy_spec[shift:time_step[0] - residual]
    # noisy_spec = noisy_spec.reshape([-1, config.stoi_correlation_time, time_step[1]])

    # clean_spec = clean_spec[shift:time_step[0] - residual]
    # clean_spec = clean_spec.reshape([-1, config.stoi_correlation_time, clean_dim[1]])

    noisy_spec_norm = noisy_spec_norm[shift:time_step[0] - residual]
    noisy_spec_norm = noisy_spec_norm.reshape([-1, config.stoi_correlation_time, time_step[1]])

    e = e[shift:time_step[0] - residual]
    e = e.reshape([-1, config.stoi_correlation_time])

    # clean_spec_norm = clean_spec_norm[shift:time_step[0] - residual]
    # clean_spec_norm = clean_spec_norm.reshape([-1, config.stoi_correlation_time, clean_dim[1]])

    noisy_spec_norm = np.expand_dims(noisy_spec_norm.transpose([0, 2, 1]), axis=3)
    e = np.expand_dims(e, axis=2)
    # clean_spec_norm = np.expand_dims(clean_spec_norm.transpose([0, 2, 1]), axis=3)

    ##########################      data return      ##############################
    return noisy_spec_norm, e

