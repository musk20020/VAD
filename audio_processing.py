import librosa
import numpy as np
import pyroomacoustics as pra
import scipy.io.wavfile as wavfile
import soundfile as sf

def scipy_load_wav(path):
    # try:
    r, y = wavfile.read(path)
    # if r != 16000:
    #     y = librosa.resample(y, r, 16000)
    maxy = np.max((y.max(), -y.min()))
    datatype = y.dtype
    shape = y.shape
    if datatype != np.float32:
        y = y.astype(np.float32)
    if len(shape) > 1:
        # print(shape)
        y = y[:, 1] / maxy

    y = y - np.mean(y)
    y = y / maxy
    return y
    # except:
    #     print('{} : {}'.format('Load wav fail', path))

def WSOLA(y, speed=1):
    # y, _ = librosa.load("/Users/musk/dataset/TCC300_split/Dev/G01FM0110/G01FM011001001.wav", sr=16000)
    # y, _ = librosa.load(audioPath, sr=16000)
    # assert speed_max>=speed_min, "max speed must greater than min speed"

    frame_t = 50 # million second
    windowSize = int(frame_t/1000*16000)
    hoplength = int(windowSize/2)
    deltaRange = 50 # sample point number (search for period alignment)

    scale = speed ## speech speed coefficient
    shift = int(scale*hoplength)

    frameNum = (len(y)-windowSize)//shift + 1

    new_y = np.zeros([frameNum+1600, windowSize])

    #### hamming window
    window = np.zeros([windowSize])
    for i in range(windowSize):
        window[i] = 0.54 - 0.46*np.cos(2*np.pi*i/(windowSize-1))

    i = 0
    new_y[0] = y[:windowSize]

    frameIndex = 1
    while i+shift+windowSize+deltaRange<len(y) and i+hoplength+windowSize+deltaRange<len(y) :
        i_1 = i+shift
        referenceFrame = y[i+hoplength:i+hoplength+windowSize]

        maxSim = 0
        maxindex = 0
        for j in range(-deltaRange, deltaRange):
            newFrame = y[i_1+j:i_1+j+windowSize]
            sim = np.dot(referenceFrame, newFrame)
            if sim > maxSim:
                maxindex = j
                maxSim = sim
        new_y[frameIndex] = y[i_1+maxindex:i_1+maxindex+windowSize]

        frameIndex += 1
        i += (shift+maxindex)

    new_y = new_y[:frameIndex]

    newYInWavw = np.zeros((len(new_y)+1)*hoplength)
    for i in range(len(new_y)):
        newYInWavw[i*hoplength:i*hoplength+windowSize] += new_y[i]*window

    # librosa.output.write_wav("/Users/musk/Desktop/test.wav", newYInWavw, 16000)
    return newYInWavw

def reverb(y, delay):
    # y, _ = librosa.load(audioPath, sr=16000)
    # assert delay_max >= delay_min, "max delay must greater than min delay"

    rt60 = delay
    # room_dim = [9, 7.5, 3.5]
    room_dim = [5, 5, 3]
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)

    source = [0.5, 0.5, 1.5]
    noise_p = [1, 0.5, 1.5]
    mic = [1., 1., 1.5]
    room.add_source(source, signal=y)

    room.add_microphone(mic, fs=16000)
    room.simulate()
    output = room.mic_array.signals[0, :]
    # librosa.output.write_wav("/Users/musk/Desktop/test.wav", output, 16000)
    return output[:len(y)]

def wiener_filter_demo():
    denoiseAI = Denoise_AI()
    sess = denoiseAI.model_init()
    input_filename = '/Users/musk/dataset/DenoiseTestSet/noisy/-5dB/G01FM0110010_h_babble.wav'
    clean_mask = (denoiseAI.maskOutput(sess, input_filename)).T
    noise_mask = 1 - clean_mask
    observed, _ = librosa.load(input_filename, 16000)

    noise_file = '/Users/musk/dataset/noise_train_16k(ESC-50)/h_babble/01ec020y-1.6442-01zo030g--1.6442.wav'
    noise, _ = librosa.load(noise_file, 16000)
    noise /= np.max(np.abs(noise))

    N_FFT = 512
    frameshift = 256
    max_att = 15
    max_att_init = 1
    att_lin = 10 ** (-max_att / 20.0)
    overest = 2

    stft_obs = librosa.stft(observed, N_FFT, frameshift, N_FFT)
    mag_obs = np.abs(stft_obs)
    phase = stft_obs / mag_obs

    noise_stft = librosa.stft(noise, N_FFT, frameshift, N_FFT)
    noise_mag_obs = np.abs(noise_stft)
    noise_phase = noise_stft / noise_mag_obs

    mag_enhance = mag_obs * clean_mask
    mag_noise = noise_mag_obs

    S_bb = np.mean(np.square(mag_noise), axis=1)
    S_bb = overest * S_bb

    s_s_hat = np.zeros(np.shape(mag_enhance), dtype=np.complex_)

    for k in range(0, mag_noise.shape[1] - 1):

        if k < 120:
            att_lin = 10 ** (-(max_att_init + float(max_att - max_att_init) * k / 120) / 20.0)
        else:
            att_lin = 10 ** (-max_att / 20.0)
        H = np.maximum(att_lin, 1 - np.divide(np.square(mag_noise[:, k]) * overest, np.square(mag_obs[:, k])))
        s_s_hat[:, k] = np.multiply(H, stft_obs[:, k])

    est_y = librosa.istft(s_s_hat, frameshift, N_FFT)

    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav("/Users/musk/Desktop/test.wav", (est_y / np.max(est_y) * maxv).astype(np.int16), 16000)

def wiener_filter(mag_src, mag_noise):

    N_FFT = 512
    frameshift = 256
    max_att = 15
    att_lin = 10 ** (-max_att / 20.0)
    overest = 2

    # S_bb = np.mean(np.square(mag_noise), axis=1)
    # S_bb = overest * S_bb
    # H = np.maximum(att_lin, 1 - np.divide(np.square(mag_noise[:, k]) * overest, np.square(mag_src[:, k])))
    # filtered_src[:, k] = np.multiply(H, mag_src[:, k])

    filtered_src = np.zeros(np.shape(mag_src))
    for k in range(0, mag_noise.shape[1]):
        H = np.maximum(att_lin, 1 - np.divide(np.square(mag_noise[:, k])*overest, np.square(mag_src[:, k])))
        filtered_src[:, k] = np.multiply(H, mag_src[:, k])

    return filtered_src


def second_order_filter(y):
    # input_filename = '/Users/musk/dataset/DenoiseTestSet/noisy/-5dB/G01FM0110010_h_babble.wav'
    # noisy, _ = librosa.load(input_filename, 16000)

    # stft = librosa.stft(noisy, 2048, 1024, 2048)
    # mag = np.abs(stft)
    # phase = stft / mag
    # mean_mag = np.mean(mag, axis=1)
    filter_y = np.zeros(np.shape(y))
    tmp_y_1 = 0
    tmp_y_2 = 0

    b0 = 1.53512485958697
    b1 = -2.69169818940638
    b2 = 1.19839281085285
    a1 = -1.69065929318241
    a2 = 0.73248077421585

    for i in range(2, len(y)):
        tmp_y = y[i]-a1*tmp_y_1-a2*tmp_y_2
        filter_y[i] = b0*tmp_y+b1*tmp_y_1+b2*tmp_y_2
        tmp_y_2 = np.copy(tmp_y_1)
        tmp_y_1 = np.copy(tmp_y)


    tmp_y_1 = 0
    tmp_y_2 = 0

    b0 = 1.0
    b1 = -2.0
    b2 = 1.0
    a1 = -1.99004745483398
    a2 = 0.99007225036621

    for i in range(2, len(filter_y)):
        tmp_y = filter_y[i]-a1*tmp_y_1-a2*tmp_y_2
        filter_y[i] = b0*tmp_y+b1*tmp_y_1+b2*tmp_y_2
        tmp_y_2 = np.copy(tmp_y_1)
        tmp_y_1 = np.copy(tmp_y)

    # mag_f = np.abs(librosa.stft(filter_y, 2048, 1024, 2048))
    # mean_mag_f = np.mean(mag_f, axis=1)

    # dB = 10*np.log10(mean_mag**2)
    # dB_f = 10*np.log10(mean_mag_f**2)
    # delta = dB_f-dB
    # delta = mean_mag_f/mean_mag
    # np.save("freq_gain_2048.npy", delta)

    # plt.figure()
    # plt.plot(delta)
    # plt.show()
    # librosa.output.write_wav("/Users/musk/Desktop/test.wav", (filter_y/np.max(np.abs(filter_y))*np.iinfo(np.int16).max).astype(np.int16), sr=16000)
    return filter_y

def second_order_filter_freq(y):
    stft = librosa.stft(y, 512, 256, 512)
    mag = np.abs(stft)
    phase = stft / (mag+np.finfo(np.float32).eps)

    gain = np.load("freq_gain.npy")
    mag *= np.expand_dims(gain, 1)
    filtered_stft = mag * phase
    filter_y = librosa.istft(filtered_stft, 256, 512)
    # filter_y = librosa.util.fix_length(filter_y, len(y), mode='edge')
    return filter_y

def cal_frame_loudness(y, window, hop_length):
    t = 300  # 4800(ms)/16
    e = np.zeros([t])
    e_gate = np.zeros([t])
    n = (len(y) - window) // hop_length
    l_KG = np.zeros([n])
    e_hist = np.zeros([n])
    f = 0
    for i in range(0, n):
        frame = y[i*hop_length:i*hop_length + window]
        z = np.mean(frame ** 2)
        l = -0.691 + 10 * np.log10(z)

        if l > -70:
            e[f % t] = z
        else:
            e[f % t] = 0

        if np.sum(e) != 0:
            gamma_r = -0.691 + 10 * np.log10(np.sum(e) / np.sum(e > 0)) - 10
        else:
            gamma_r = -np.finfo(np.float32).max

        if l > gamma_r:
            e_gate[f % t] = z
            e_hist[f] = z
        else:
            e_gate[f % t] = 0
            e_hist[f] = 0


        l_KG[f] = -0.691 + 10 * np.log10(np.sum(e_gate) / np.sum(e_gate > 0))
        f += 1
    return l_KG, e_hist

def loudness_normalize(noisy, enhance):
    t = 300  # 80/16
    n = (len(noisy) - 256) // 256
    output = np.zeros([len(noisy)])

    ## noisy loudness
    e = np.zeros([t])
    e_gate = np.zeros([t])
    l_KG = np.zeros([n])
    f = 0
    for i in range(0, len(noisy) - 256, 256):
        frame = noisy[i:i + 256]
        z = np.mean(frame ** 2)
        l = -0.691 + 10 * np.log10(z)
        if l > -70:
            e[f % t] = z
        else:
            e[f % t] = 0

        if np.sum(e) != 0:
            gamma_r = -0.691 + 10 * np.log10(np.sum(e) / np.sum(e > 0)) - 10
        else:
            gamma_r = -np.finfo(np.float32).max
        if l > gamma_r:
            e_gate[f % t] = z
        else:
            e_gate[f % t] = 0

        if np.sum(e_gate) != 0:
            l_KG[f] = -0.691 + 10 * np.log10(np.sum(e_gate) / np.sum(e_gate > 0))
        else:
            l_KG[f] = -np.finfo(np.float32).max
        f += 1

    ## noisy loudness
    e_e = np.zeros([t])-15
    e_gate_e = np.zeros([t])-15
    l_KG_e = np.zeros([n])
    f = 0
    for i in range(0, len(enhance) - 256, 256):
        frame = enhance[i:i + 256]
        z = np.mean(frame ** 2)
        l = -0.691 + 10 * np.log10(z)
        if l > -70:
            e_e[f % t] = z
        else:
            e_e[f % t] = 0
        if np.sum(e_e) != 0:
            gamma_r = -0.691 + 10 * np.log10(np.sum(e_e) / np.sum(e_e > 0)) - 10
        else:
            gamma_r = -np.finfo(np.float32).max

        if l > gamma_r:
            e_gate_e[f % t] = z
        else:
            e_gate_e[f % t] = 0

        if np.sum(e_gate_e) != 0:
            l_KG_e[f] = -0.691 + 10 * np.log10(np.sum(e_gate_e) / np.sum(e_gate_e > 0))
        else:
            l_KG_e[f] =  -np.finfo(np.float32).max

        if l > gamma_r:
            A = np.sqrt(10**(l_KG_e[f]/10))
            A_target = np.sqrt(10**(l_KG[f]/10))
            output[i:i + 256] = enhance[i:i + 256] / (A+np.finfo(np.float32).eps) * A_target
            maxv = np.max(np.abs(output[i:i + 256]))
            if maxv>1:
                output[i:i + 256] /= maxv
        else:
            output[i:i + 256] = enhance[i:i + 256]

        f += 1

    return output

def loudness_normalize_to_dB(noisy, dB, ref=None):
    t = 300  # 80/16
    n = (len(noisy) - 256) // 256 +1
    output = np.zeros([len(noisy)])

    ## noisy loudness
    e = np.zeros([t])
    e_gate = np.zeros([t])
    l_KG = np.zeros([n])
    f = 0
    for i in range(0, len(noisy) - 256, 256):
        frame = noisy[i:i + 256]
        z = np.mean(frame ** 2)
        l = -0.691 + 10 * np.log10(z)
        if l > -70:
            e[f % t] = z
        else:
            e[f % t] = 0

        if np.sum(e) != 0:
            gamma_r = -0.691 + 10 * np.log10(np.sum(e) / np.sum(e > 0)) - 10
        else:
            gamma_r = -np.finfo(np.float32).max
        if l > gamma_r:
            e_gate[f % t] = z
        else:
            e_gate[f % t] = 0

        if np.sum(e_gate) != 0:
            l_KG[f] = -0.691 + 10 * np.log10(np.sum(e_gate) / np.sum(e_gate > 0))
        else:
            l_KG[f] = -np.finfo(np.float32).max

        if l > gamma_r:
            A = np.sqrt(10**(l_KG[f]/10))
            A_target = np.sqrt(10**(dB/10))
            output[i:i + 256] = noisy[i:i + 256] / A * A_target
            if ref is not None:
                ref[i:i + 256] = ref[i:i + 256] / A * A_target
            maxv = np.max(np.abs(output[i:i + 256]))
            if maxv>1:
                output[i:i + 256] /= maxv
                if ref is not None:
                    ref[i:i + 256] /= maxv
        else:
            output[i:i + 256] = noisy[i:i + 256]

        f += 1

    if ref is not None:
        return output, ref
    else:
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

    o = generate_data(np.log10(np.array(center_freq)), *res_lsq.x)

    return res_lsq

def wgn(x, SNR=5.0, noise=None):
    snr = 10 ** (SNR / 10.0)
    xpower = np.sum( x ** 2 ) / len( x )
    tpower = xpower / snr
    if noise is None:
        noise = np.random.randn( len( x ) )
    else:
        noise = noise[:len(x)]
    npower = np.sum( noise ** 2 ) / len( noise )
    return  x+noise/npower*tpower

def audio_eq(audio_path, EQ, sr=16000, window=256, y=None):
    def generate_data(t, a1, a2, a3, a4, a5, a6):
        y = a1 + a2*t + a3*t**2 + a4*t**3+ a5*t**4 + a6*t**5
        return y

    max_freq = sr/2
    freq_resolution = window/2+1
    band = np.linspace(1, max_freq, freq_resolution)
    eq = generate_data(np.log10(np.array(band)), *EQ.x)
    eq[0] = 0
    eq = 10**(eq/10)

    if y is None:
        y, _ = librosa.load(audio_path, sr=sr)
    stft = librosa.stft(y, 256, 128, 256)

    eq_stft = np.multiply(stft, np.reshape(eq, [-1, 1]))

    eq_y = librosa.istft(eq_stft, 128, 256)

    return eq_y

def loudness_normalize_demo():
    input_filename = '/Users/musk/dataset/DenoiseTestSet/noisy/-5dB/G01FM0110010_h_babble.wav'
    # input_filename = '/Users/musk/dataset/TCC300/Dev/G01FM0110/G01FM0110010.wav'
    enhance_filename = '/Users/musk/Desktop/enhance.wav'

    noisy, _ = librosa.load(input_filename, 16000)
    enhance, _ = librosa.load(enhance_filename, 16000)

    filter_noisy = second_order_filter_freq(noisy)
    filter_enhance = second_order_filter_freq(enhance)

    # l_n_y = loudness_normalize(filter_noisy, filter_enhance)
    l_n_y = loudness_normalize_to_dB(filter_noisy, -14)
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        "/Users/musk/Desktop/test.wav", (l_n_y * maxv).astype(np.int16), 16000)


def gen_noisy(y_target, noise, SNR, preEnhance=False, getNoise=False):
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

    y_target = check_min_length(y_target, 160000)
    target_pwr = sum(abs(y_target) ** 2) / len(y_target)

    noise = same_length(y_target, noise)
    noise -= np.mean(noise)
    # y_target_var = target_pwr / (10 ** (SNR / 10))
    noise_target_var = target_pwr / (10 ** (SNR / 10))
    noise = np.sqrt(noise_target_var) * noise / np.std(noise)

    prob = np.random.rand()
    if prob < 0.01:
        y_noisy = y_target
    else:
        y_noisy = y_target+noise

    max_y = np.max(np.abs(y_noisy))
    y_target /= max_y
    y_noisy /= max_y

    return y_target, y_noisy

def gen_noisy_v2(clean, noise, SNR=0):
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
                flex_len_data = check_min_length(flex_len_data, len(fix_len_data))
                flex_len_data = flex_len_data[:len(fix_len_data)]
            else:
                flex_len_data = flex_len_data[:len(fix_len_data)]
            return flex_len_data
        else:
            if len(flex_len_data) < len(fix_len_data):
                fix_len_data = fix_len_data[:len(flex_len_data)]
            else:
                flex_len_data = flex_len_data[:len(fix_len_data)]
            return fix_len_data, flex_len_data

    noise = same_length(clean, noise)

    l_KG, e_hist = cal_frame_loudness(clean)

    n = (len(noise) - 256) // 256 + 1
    f = 0
    frame_num = np.sum(e_hist > 0)
    noise_e = np.zeros([frame_num])
    clean_e = np.zeros([frame_num])

    for i in range(0, len(noise) - 256, 256):
        if e_hist[int(i / 256)] > 0:
            frame_n = noise[i:i + 256]
            frame_s = clean[i:i + 256]
            z_n = np.mean(frame_n ** 2)
            z_s = np.mean(frame_s ** 2)
            noise_e[f] = z_n
            clean_e[f] = z_s
            f += 1

    src_a = np.mean(noise_e)
    tar_a = np.mean(clean_e) * (10 ** (-SNR / 10))
    noise = noise / (np.sqrt(src_a)+np.finfo(np.float32).eps) * np.sqrt(tar_a)
    noisy = clean + noise

    max_y = np.max(np.abs(noisy))
    noisy /= (max_y+np.finfo(np.float32).eps)

    # clean /= max_y
    # noise /= max_y
    # maxv = np.iinfo(np.int16).max
    # librosa.output.write_wav("/Users/musk/Desktop/test1.wav", (clean * maxv).astype(np.int16), sr=16000)
    # librosa.output.write_wav("/Users/musk/Desktop/test2.wav", (noise * maxv).astype(np.int16), sr=16000)

    # maxv = np.iinfo(np.int16).max
    # librosa.output.write_wav("/Users/musk/Desktop/test.wav", (noisy * maxv).astype(np.int16), sr=16000)

    return noisy, max_y

def gen_training_data(clean_file, noise_file, SNR=0):
    # clean_file = "/Users/musk/dataset/TCC300_rename/Dev/G01FM0110/G01FM0110010.wav"
    # noise_file = "/Users/musk/dataset/noise_train_16k(ESC-50)/h_babble/01aa010k-1.3053-01po0310--1.3053.wav"

    #### load wav
    # clean, _ = librosa.load(clean_file, sr=16000, mono=True)
    # clean /= np.max(np.abs(clean))
    # noise, _ = librosa.load(noise_file, sr=16000, mono=True)
    # noise /= np.max(np.abs(noise))
    file_type = clean_file.split(".")[-1]
    if file_type == "flac":
        clean, _ = sf.read(clean_file)
    else:
        clean = scipy_load_wav(clean_file)

    noise = scipy_load_wav(noise_file)
    # noise, _ = librosa.load(noise_file, sr=16000, mono=True)

    # #### 2nd filter to caculate correct loudness
    clean = second_order_filter_freq(clean)
    noise = second_order_filter_freq(noise)

    #### speech speed tuning
    speed_min = 0.9
    speed_max = 1.3
    speed = speed_min + (speed_max - speed_min) * np.random.rand()
    clean = WSOLA(clean, speed)

    #### speech reverb preporcessing (data aug)
    # if np.random.rand() < 0.01:
    #     delay_min = 0.3
    #     delay_max = 0.5
    #     delay = delay_min + (delay_max-delay_min)*np.random.rand()
    #     clean_reverb = reverb(clean, delay)
    # else:
    #     clean_reverb = clean

    #### generate noisy
    clean, noisy = gen_noisy(clean, noise, SNR=SNR)

    # #### speech data calibration by max value
    # clean /= max_y

    #### input data normalization
    # noisy_norm, clean_norm = loudness_normalize_to_dB(noisy, dB=-14, ref=clean)

    # return clean, noisy, noisy_norm, clean_norm
    return clean, noisy

