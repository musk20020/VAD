import numpy as np
from resampy import resample
import tensorflow as tf
# from configuration import get_config
from tqdm import tqdm
from glob import iglob
import os
from os.path import join
import xlrd
import librosa
import scipy
# from basicRNN_poly_activate_impl import PolyBasicRNNCell
# from gru_poly_cell import GRUPolyCell
from scipy.optimize import least_squares
# from OpenStateLSTMCell import LSTMCell
import struct
import warnings

# config = get_config()
epsilon = np.finfo(float).eps

def search_wav(data_path, cond=None):
    file_list = []
    #for filename in iglob('{}/-5*.wav'.format(data_path), recursive=True):
    #    file_list.append(str(filename))
    if cond is None:
        for filename in iglob('{}/**/*.wav'.format(data_path), recursive=True):
            file_list.append(str(filename))
    else:
        for filename in iglob( cond.format( data_path ), recursive=True ):
            file_list.append( str( filename ) )
    return file_list

def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)

def gen_noisy(clean_file_list, noise_file_list, save_dir, snr, sr_clean, sr_noise):
    SNR = float( snr.split( 'dB' )[0] )
    for i in range(len(clean_file_list)):
        clean_file = clean_file_list[i]
        noise_file = noise_file_list[i]
        clean_name = clean_file.split( '/' )[-1].split( '.' )[0]
        noise_name = noise_file.split( '/' )[-1].split( '.' )[0]
        y_clean, sr_clean = librosa.load( clean_file, sr_clean, mono=True )
        if sr_clean != 16000:
            y_clean = librosa.resample(y_clean, sr_clean, 16000)
        #### scipy cannot conver TIMIT format ####

        clean_pwr = sum( abs( y_clean ) ** 2 ) / len( y_clean )
        y_noise, sr_noise = librosa.load( noise_file, sr_noise, mono=True )
        y_noise = y_noise[8000:]
        if sr_noise != 16000:
            y_noise = librosa.resample(y_noise, sr_noise, 16000)

        tmp_list = []
        if len( y_noise ) < len( y_clean ):
            tmp = (len( y_clean ) // len( y_noise )) + 1
            y_noise = np.array( [x for j in [y_noise] * tmp for x in j] )
            y_noise = y_noise[:len( y_clean )]
        else:
            y_noise = y_noise[:len( y_clean )]

        y_noise = y_noise - np.mean( y_noise )
        noise_variance = clean_pwr / (10 ** (SNR / 10))

        noise = np.sqrt( noise_variance ) * y_noise / np.std( y_noise )
        #noise = np.sqrt( noise_variance / np.mean(y_noise**2)) * y_noise

        y_noisy = y_clean + noise

        maxv = np.iinfo( np.int16 ).max

        save_name = '{}_{}.wav'.format( clean_name, noise_name )
        librosa.output.write_wav(
            '/'.join( [save_dir, save_name] ), (y_noisy / np.max( np.abs( y_noisy ) ) * maxv).astype( np.int16 ), 16000 )



def phase_iter(spec, phase):
    reverse = np.multiply( spec, phase.T )
    result = librosa.istft( reverse,
                            hop_length=256,
                            win_length=512,
                            window=scipy.signal.hann )
    min_delta = 0.0001
    delta_delta_phase = 1
    delta_phase_tmp = 2
    while delta_delta_phase>min_delta:
        D = librosa.stft( result,
                          n_fft=512,
                          hop_length=256,
                          win_length=512,
                          window=scipy.signal.hamming )
        iter_phase = D/np.abs(D)
        iter_phase = iter_phase.T
        delta_phase = np.mean(np.abs(iter_phase-phase))
        delta_delta_phase = delta_phase_tmp-delta_phase
        delta_phase_tmp = delta_phase
        phase = iter_phase
        reverse = np.multiply( spec, phase.T )
        result = librosa.istft( reverse,
                                hop_length=256,
                                win_length=512,
                                window=scipy.signal.hann )
    return result

def spec2wav(phase, output_filename, spec=None, hop_length=None, noisy_file=None):

    phase_predict = phase[:, :hop_length+1] + phase[:, hop_length+1:]*1j
    phase_predict /= np.abs(phase_predict)
    if spec is None:
        spec = np.sqrt(np.square(phase[:, :257]) + np.square(phase[:, 257:]))
        Sxx_r_tmp = np.array( spec )
    else:
        Sxx_r_tmp = np.array( spec )
        #Sxx_r_tmp = np.sqrt( 10 ** Sxx_r_tmp )

    Sxx_r = Sxx_r_tmp.T
    reverse = np.multiply(Sxx_r, phase_predict.T)

    result = librosa.istft(reverse,
                           hop_length=hop_length,
                           win_length=hop_length*2,
                           window=scipy.signal.hamming,
                           center=False)
    #result = phase_iter(Sxx_r, phase_predict)

    # y, _ = librosa.load( noisy_file, 16000, mono=True, dtype=np.float16 )
    y, _ = librosa.load( noisy_file, 16000, mono=True )
    y_out = librosa.util.fix_length(result, len(y)-128, mode='edge')

    #y = y * np.sqrt(0.11 * np.mean( np.square( y_out ) ) / np.mean( np.square( y ) ) ) # 1dB mix
    #y = y*np.sqrt(0.10251*np.mean(np.square(y_out))/np.mean(np.square(y))) # 10dB mix
    #y = y * np.sqrt( 0.08 * np.mean( np.square( y_out ) ) / np.mean( np.square( y ) ) )  # 10dB mix
    #y_out += y
    #y_out = result / np.max( np.abs( result ) )
    #y_out = y_out/np.max(np.abs(y_out))
    maxv = np.iinfo(np.int16).max
    librosa.output.write_wav(
        output_filename, (y_out * maxv).astype(np.int16), 16000)

def wav2spec(wavfile, sr, hop_length=256, phase=False, high_pass=False, under_4k=None, center=True, norm=True, eq=None):
    # Note:This function return three different kind of spec for training and
    # testing
    y, sr = librosa.load(wavfile, sr, mono=True)

    if eq is not None:
        y = audio_eq(None, eq, sr=16000, window=256, y=y)

    if high_pass:
        y[1:] -= 0.7 * y[:-1]

    NUM_FFT = hop_length*2
    y -= np.mean( y )
    if norm:
        y = y/np.max( np.abs( y ) )

    D = librosa.stft(y,
                     n_fft=NUM_FFT,
                     hop_length=hop_length,
                     win_length=NUM_FFT,
                     window=scipy.signal.hann,
                     center=center)

    if phase:
        Sxx_r = D
        # D = D + epsilon
        # a = np.angle( D )
        # Sxx_r = np.exp( 1j * a )
    else:
        Sxx = abs(D + epsilon)
        Sxx_r = np.array(Sxx)

    if under_4k:
        Sxx_r = Sxx_r[:under_4k]

    Sxx_r = np.array(Sxx_r).T
    shape = Sxx_r.shape

    return Sxx_r

def ToWaveformData(wavfile, window_size):
    #y, sr = librosa.load(wavfile, sr, mono=True)
    y, sr = librosa.load( wavfile, 16000, mono=True )
    y -= np.mean( y )
    y = y / np.max( np.abs( y ) )
    diss_y = len(y)%window_size
    #print(diss_y)
    #if diss_y > 1:
    y = y[diss_y:]
    Sxx_r = np.array(y).reshape([-1, window_size])
    #print(Sxx_r.shape)
    return Sxx_r

def SaveWave(y_out, output_filename):
    y_out = y_out / np.max( np.abs( y_out ) )
    maxv = np.iinfo( np.int16 ).max
    librosa.output.write_wav(
        output_filename, (y_out * maxv).astype( np.int16 ), 16000 )

def _batch_norm_v2(input_x, dimension, layer_num, decay=0.99):
    with tf.variable_scope( "batch_normalize_layer_" + layer_num ):
        epsilon = np.finfo( float ).eps
        mean = tf.get_variable( "mean_" + str( layer_num ), shape=[1, dimension],
                                initializer=tf.constant_initializer( value=0, dtype=tf.float32 ),
                                trainable=False )
        var = tf.get_variable( "var_" + str( layer_num ), shape=[1, dimension],
                               initializer=tf.constant_initializer( value=1, dtype=tf.float32 ),
                               trainable=False )
        shift = tf.get_variable( "shift_" + str( layer_num ), shape=[1, dimension],
                                 initializer=tf.constant_initializer( value=0, dtype=tf.float32 ) )
        scale = tf.get_variable( "scale_" + str( layer_num ), shape=[1, dimension],
                                 initializer=tf.constant_initializer( value=1, dtype=tf.float32 ) )
        mean_scale = tf.reduce_mean( mean )
        var_scale = tf.reduce_mean( var )
        shift_scale = tf.reduce_mean( shift )
        scale_scale = tf.reduce_mean( scale )
        tf.summary.scalar( "mean_" + str( layer_num ), mean_scale )
        tf.summary.scalar( "var_" + str( layer_num ), var_scale )
        tf.summary.scalar( "shift_" + str( layer_num ), shift_scale )
        tf.summary.scalar( "scale_" + str( layer_num ), scale_scale )
        # if is_training:
        mean_batch, var_batch = tf.nn.moments( input_x, 1 )
        mean_batch = tf.reduce_mean( mean_batch, 0 )
        var_batch = tf.reduce_mean( var_batch, 0 )
        train_mean = tf.assign( mean,
                                mean * decay + mean_batch * (1 - decay) )
        train_var = tf.assign( var,
                               var * decay + var_batch * (1 - decay) )
        with tf.control_dependencies( [train_mean, train_var] ):
            return tf.nn.batch_normalization( input_x, mean_batch, var_batch,
                                              shift, scale, epsilon )

def _batch_norm(input_x, dimension, channal, mean_layer, layer_num):
    epsilon = np.finfo( float ).eps
    mean = tf.get_variable( "mean_" + str( mean_layer ), shape=[1, dimension, 1, channal],
                            initializer=tf.constant_initializer( value=0, dtype=tf.float32 ),
                            trainable=False )
    var = tf.get_variable( "var_" + str( layer_num ), shape=[1, dimension, 1, channal],
                           initializer=tf.constant_initializer( value=1, dtype=tf.float32 ),
                           trainable=False )
    shift = tf.get_variable( "shift_" + str( layer_num ), shape=[1, dimension, 1, channal],
                             initializer=tf.constant_initializer( value=0, dtype=tf.float32 ) )
    scale = tf.get_variable( "scale_" + str( layer_num ), shape=[1, dimension, 1, channal],
                             initializer=tf.constant_initializer( value=1, dtype=tf.float32 ) )
    return tf.nn.batch_normalization( input_x, mean, var, shift, scale, epsilon )

def _add_2dfc_layer(input_, neural_number, output_number, layer_num, activate_function):
    with tf.name_scope( "fc_layer_" + layer_num):
        w = tf.get_variable("W_"+layer_num, shape=[neural_number, output_number], initializer=tf.contrib.layers.xavier_initializer() )
        b = tf.get_variable("B_"+layer_num, shape=[1, output_number], initializer=tf.constant_initializer( value=0, dtype=tf.float32))
        output = activate_function( tf.add( tf.matmul(input_, w ), b ) )
    return output

def _add_3dfc_layer(input_, neural_number, output_number, layer_num, activate_function=tf.nn.relu6,
                    trainable=True, keep_prob=1, dtype=tf.float32):
    with tf.variable_scope( "fc_layer_" + layer_num ):
        w = tf.get_variable( "W_" + layer_num, shape=[neural_number, output_number],
                             initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable, dtype=dtype )
        b = tf.get_variable( "B_" + layer_num, shape=[1, output_number],
                             initializer=tf.constant_initializer( value=0, dtype=dtype ), trainable=trainable )
        # w = tf.cast(w, tf.float16)
        # b = tf.cast(b, tf.float16)
        # output = tf.cond(tf.not_equal(str(activate_function), 'tf.nn.relu6'),
        #                 lambda:activate_function(tf.add(tf.einsum('bij,jk->bik', input_, w), b)),
        #                 lambda:tf.add(tf.einsum('bij,jk->bik', input_, w), b))
        # output = tf.add( tf.einsum( 'bij,jk->bik', input_, w ), b )
        output = activate_function( tf.add( tf.einsum( 'bij,jk->bik', input_, w ), b ) )
        #output = tf.nn.dropout( output, keep_prob )
    return output

def _add_gru(input_, gru_hidden_size, layer_num, keep_prob=1, trainable=True, activate_function=tf.nn.tanh):
    with tf.variable_scope( "gru_layer_" + layer_num ):
        gru_cell = tf.nn.rnn_cell.GRUCell( num_units=gru_hidden_size, activation=activate_function,
                                           trainable=trainable )
        outputs, _ = tf.nn.dynamic_rnn( cell=gru_cell, inputs=input_, dtype=tf.float32,
                                        time_major=False )  # for TI-VS must use dynamic rnn
    return outputs

# def _add_poly_gru(input_, gru_hidden_size, layer_num, keep_prob=1, trainable=True,
#                   activate_function=tf.nn.tanh):
#     with tf.variable_scope( "poly_gru_layer_" + layer_num ):
#         gru_cell = GRUPolyCell( num_units=gru_hidden_size, activation=activate_function, trainable=trainable )
#         outputs, _ = tf.nn.dynamic_rnn( cell=gru_cell, inputs=input_, dtype=tf.float32,
#                                         time_major=False )  # for TI-VS must use dynamic rnn
#     return outputs

def _add_bidir_lstm(_input, lstm_hidden_size, layer_num):
    # with tf.name_scope( "bidir_LSTM_layer_" + layer_num ):
    with tf.variable_scope( "bidir_LSTM_layer_" + layer_num ):
        encoder_f_cell = tf.nn.rnn_cell.LSTMCell( lstm_hidden_size )
        encoder_b_cell = tf.nn.rnn_cell.LSTMCell( lstm_hidden_size )
        (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
            tf.nn.bidirectional_dynamic_rnn( cell_fw=encoder_f_cell,
                                             cell_bw=encoder_b_cell,
                                             inputs=_input,
                                             dtype=tf.float32,
                                             time_major=True )
        # input has shape [batch, time, feature_dim] (default time major == False)
        # fw_output = tf.nn.embedding_lookup(tf.transpose( encoder_fw_outputs, [1, 0, 2] ), self.time_step-1)
        # bw_output = tf.nn.embedding_lookup(tf.transpose( encoder_bw_outputs, [1, 0, 2] ), self.time_step-1)
        encoder_outputs = tf.concat( (encoder_fw_outputs, encoder_bw_outputs), 2 )
    return encoder_outputs

def _add_lstm(input_, lstm_hidden_size, proj, layer_num, activation_function=tf.nn.tanh, time_major=False):
    with tf.variable_scope( "lstm_layer_" + layer_num ):
        lstm_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size, num_proj=proj, activation=activation_function)

        outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=input_, dtype=tf.float32,
                                        time_major=time_major )  # for TI-VS must use dynamic rnn
    return outputs

# def _add_lstm_OpenState(input_, lstm_hidden_size, proj, layer_num, activation_function=tf.nn.tanh, time_major=False, dtype=tf.float32):
#     with tf.variable_scope( "lstm_layer_" + layer_num ):
#         lstm_cell = LSTMCell( lstm_hidden_size, num_proj=proj, activation=activation_function, dtype=dtype)
#
#         outputs, state = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=input_, dtype=tf.float32,
#                                         time_major=time_major )  # for TI-VS must use dynamic rnn
#     return outputs, state

def _add_Blstm(input_, lstm_hidden_size, proj, layer_num, keep_prob=1, trainable=True):
    encoder_f_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size, num_proj=proj, trainable=trainable )
    encoder_b_cell = tf.nn.rnn_cell.LSTMCell( num_units=lstm_hidden_size, num_proj=proj, trainable=trainable )
    # c_state = tf.zeros( shape=(self.config.batch_size, neural_number) )
    # h_state = tf.zeros( shape=(self.config.batch_size, neural_number) )
    # init_state = LSTMStateTuple( c_state, h_state )
    (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) = \
        tf.nn.bidirectional_dynamic_rnn( cell_fw=encoder_f_cell,
                                         cell_bw=encoder_b_cell,
                                         inputs=input_,
                                         dtype=tf.float32
                                         )
    # fw_output = tf.transpose( encoder_fw_outputs, [1, 0, 2] )
    # bw_output = tf.transpose( encoder_bw_outputs, [1, 0, 2] )
    outputs = tf.concat( (encoder_fw_outputs, encoder_bw_outputs), 2 )
    return outputs

def _add_basicRNN(input_, hidden_size, layer_num, keep_prob=1, trainable=True, activate_function=tf.nn.tanh):
    with tf.variable_scope( "basicRNN_layer_" + layer_num ):

        lstm_cell = tf.nn.rnn_cell.BasicRNNCell( num_units=hidden_size, activation=activate_function,
                                                 trainable=trainable )
        outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=input_, dtype=tf.float32,
                                        time_major=False )  # for TI-VS must use dynamic rnn
    return outputs

# def _add_basicRNN_poly(input_, hidden_size, layer_num, keep_prob=1, trainable=True, activate_function=tf.nn.tanh):
#     with tf.variable_scope( "poly_basicRNN_layer_" + layer_num ):
#
#         lstm_cell = PolyBasicRNNCell( num_units=hidden_size, activation=activate_function,
#                                                  trainable=trainable )
#         outputs, _ = tf.nn.dynamic_rnn( cell=lstm_cell, inputs=input_, dtype=tf.float32,
#                                         time_major=False )  # for TI-VS must use dynamic rnn
#     return outputs

def _add_conv_layer(_input, layer_num, filter_h, filter_w, input_c, output_c, strides=[1, 1, 1, 1],
                        dilation=[1, 1, 1, 1], activate=tf.nn.tanh, padding='VALID', dtype=tf.float32, trainable=True):
    with tf.variable_scope( "conv_layer_" + layer_num ):
        # CNN output default is "NHWC"
        weights = tf.get_variable( 'conv_layer_w' + layer_num, shape=[filter_h, filter_w, input_c, output_c]
                                   , initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype, trainable=trainable )
        # weights = tf.cast(weights, tf.float16)
        conv = tf.nn.conv2d( _input, weights, strides=strides,
                             padding=padding, dilations=dilation )
        output = activate( conv )
        # output = conv
    return output

def _add_conv_add_layer(_input, layer_num, filter_h, filter_w, input_c, output_c, bias_h,
                        strides=[1, 1, 1, 1], dilation=[1, 1, 1, 1], activate=tf.nn.tanh,
                        padding='SAME', keep_prob=1, trainable=False):
    with tf.variable_scope("conv_layer_" + layer_num):
        # CNN output default is "NHWC"
        weights = tf.get_variable('conv_layer_w' + layer_num, shape=[filter_h, filter_w, input_c, output_c]
                                  , initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable,
                                  dtype=tf.float32)
        conv = tf.nn.conv2d(_input, weights, strides=strides,
                            padding=padding, dilations=dilation)
        bias = tf.get_variable('conv_layer_b' + layer_num, shape=[1, bias_h, 1, output_c], trainable=trainable,
                               dtype=tf.float32)
        bias = tf.reduce_mean(bias, 2, True)
        output = activate(conv + bias)
        output = tf.nn.dropout(output, keep_prob)
    return output

def _add_interaction_layer(source, reference, layer_num, input_c, bias_h):
    '''
    SN-Net model interaction layer
    (2 module's concolution layer exchange information for)

    :param source: convolution source tensor witch has shape [batch, H, W, C]
    :param reference: convolution reference tensor witch has shape [batch, H, W, C]
    :param layer_num: number of layer (string type)
    :param input_c: channel number of source tensor
    :return: return a tensor has same shape as input tensorflow
    '''

    with tf.variable_scope("interaction_layer_" + layer_num):
        concat_layer = tf.concat((source, reference), axis=3)
        mask = _add_conv_add_layer(concat_layer, layer_num, 5, 1, input_c * 2, input_c, activate=tf.nn.sigmoid,
                                    padding="SAME", bias_h=bias_h)
        rederence_info = mask * reference
        interaction_out = source + rederence_info

    return interaction_out

def _add_deconv_layer(input, layer_num, kernal_H, kernal_W, input_c, output_c, output_shape,
                      bias_h, stride=None,
                      activation=tf.nn.leaky_relu, trainable=False):
    with tf.variable_scope("deconv_layer_" + layer_num):
        kernal = tf.get_variable('deconv_kernal_' + layer_num, shape=[kernal_H, kernal_W, output_c, input_c]
                                 , initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable,
                                 dtype=tf.float32)
        bias = tf.get_variable('deconv_kernal_b'+layer_num, shape=[1, bias_h, 1, output_c],
                               trainable=trainable, dtype=tf.float32)
        bias = tf.reduce_mean(bias, 2, True)
        deconv = tf.nn.conv2d_transpose(input, kernal, output_shape=output_shape, strides=stride)
        output = activation(deconv+bias)
    return output

def _add_decoder_layer(input, skip, layer_num, deconv_kernal_H, deconv_kernal_W, deconv_input_c,
                       deconv_output_c, output_shape, conv_kernal_H, conv_kernal_W, bias_h,
                       stride=None, activation=tf.nn.leaky_relu):
    '''
    SN_Net decoder module
    (U-Net structure, recevie input feature and skip feature to combine local and global feature together)

    :param input: local feature input tensor witch has shape [batch, H, W, C]
    :param skip: skip feature(global feature) input tensor witch has shape 'output_shape'
    :param layer_num: number of layer (string type)
    :param deconv_kernal_H: same as encoder_kernal_H
    :param deconv_kernal_W: same as encoder_kernal_W
    :param deconv_input_c: input channel number
    :param deconv_output_c: same as input channel of encoder input tensor
    :param output_shape: same shape as the shape of encoder input tensor
    :param conv_kernal_H: convolution kernal H
    :param conv_kernal_W: convolution kernal W
    :param stride: same as encoder stride
    :param activation: activation function
    :return: decoder output tensor
    '''
    with tf.variable_scope("decoder_layer_" + layer_num):
        deconv = _add_deconv_layer(input, layer_num, deconv_kernal_H, deconv_kernal_W, deconv_input_c,
                                        deconv_output_c, output_shape, bias_h, stride, activation)
        concat_skip = tf.concat((deconv, skip), 3)
        mask = _add_conv_add_layer(concat_skip, layer_num + "-1", conv_kernal_H, conv_kernal_W,
                                    deconv_output_c * 2, deconv_output_c, activate=tf.nn.leaky_relu,
                                    padding="SAME", bias_h=bias_h, )
        skip_info = tf.multiply(skip, mask)
        concat_deconv = tf.concat((skip_info, deconv), 3)
        decoder_info = _add_conv_add_layer(concat_deconv, layer_num + "-2", conv_kernal_H, conv_kernal_W,
                                            deconv_output_c * 2, deconv_output_c, activate=activation,
                                            padding="SAME", bias_h=bias_h)
        decoder_output = tf.add(decoder_info, deconv)
    return decoder_output

def _linear(x):
    return x

def _poly_sigmoid(inputs):
    inputs = 0.2 * inputs + 0.5
    inputs = tf.clip_by_value( inputs, 0, 1 )
    return inputs

def _poly_tanh(inputs):
    inputs = tf.clip_by_value( inputs, -1, 1 )
    return inputs

def segmental_SNR(clean_path, target_path, sr=16000, window_size=256):
    target, _ = librosa.load( target_path, sr=16000 )
    max_SNR = -10

    for i in range(0, 40):
        # target = target_ori[int(i*16):]
        clean_ori, _ = librosa.load( clean_path, sr=16000 )
        clean = clean_ori[int(i*16):]
        if len(clean)>len(target):
            clean = clean[:len(target)]
        else:
            target = target[:len( clean )]

        # residual = len( target ) % window_size
        # target = target[residual:]
        power_t = np.sum( target ** 2 )


        # clean = clean[residual:]
        #clean = clean[:-128][residual:]
        #clean /= np.max( np.abs( clean ) )
        power_c = np.sum( clean ** 2 )

        target = target / np.sqrt( power_t ) * np.sqrt( power_c )

        SNR = 10*np.log(power_c/np.sum((target-clean)**2))

        if SNR>max_SNR:
            max_SNR = SNR
            index = i

    return max_SNR, index

def byteLog2wav(filepath, sr, datatype, channal):


    # f = open('/Users/musk/Desktop/readCbuf0826_16k.txt', 'rb')
    f = open(filepath, 'rb')
    bytes = f.read()
    byteArray = np.array([byte for byte in bytes])

    if datatype == "int16":
        dataByteNumber = 2
    elif datatype == "int8":
        dataByteNumber = 2
    else:
        warnings.warn('please amke sure "datatype" is either "int16" or "int8" string')


    wav = np.zeros([int(len(bytes) / 4)])
    for n in range(0, len(a), 4):
        uint16 = struct.unpack('h', a[n + 2:n + 4])
        wav[int(n / 4)] = uint16[0]

    # c = b.astype(np.int16)
    c = wav
    for i in range(len(c)):
        if c[i] > 127:
            c[i] = c[i] - 256

    # d = c[1::2]
    librosa.output.write_wav('/Users/musk/Desktop/logs.wav', c.astype(np.int16), 16000)
