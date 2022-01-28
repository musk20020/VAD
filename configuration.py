import argparse
import numpy as np

parser = argparse.ArgumentParser()    # make parser


# get arguments
def get_config():
    config, unparsed = parser.parse_known_args()
    return config

# return bool type of argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Data Parameters
train_arg = parser.add_argument_group('Data')

###   machine setting
train_arg.add_argument('--note', type=str, default='VAD', help="describe your model")
train_arg.add_argument('--date', type=str, default='220128', help="training model date")
train_arg.add_argument('--gpu_index', type=str, default="1", help="training model date")
train_arg.add_argument('--thread_num', type=int, default=8, help="parallel processing number")
# train_arg.add_argument('--read_ckpt', type=str, default='saver_moduleModel/211230-humanVoice', help="read ckpt file path(/ read_ckpt /)")
train_arg.add_argument('--read_ckpt', type=str, default=None, help="read ckpt file path(/ read_ckpt /)")
train_arg.add_argument('--log_path', type=str, default='/AudioProject/VAD/model/logs', help="save log path")
train_arg.add_argument('--saver_path', type=str, default='/AudioProject/VAD/model/saver', help="save model path")

train_arg.add_argument('--voice_path', type=str, default='/AudioProject/dataset/TCC300_rename/Train/*/*'
                       , help="clean voice data path")
train_arg.add_argument('--dev_voice_path', type=str, default='/AudioProject/dataset/TCC300_rename/Dev/*/*'
                       , help="clean voice data path")
train_arg.add_argument('--noise_path', type=str, default='/AudioProject/dataset/noise/mix_train/*'
                       , help="clean voice data path")
train_arg.add_argument('--dev_noise_path', type=str, default='/AudioProject/dataset/noise/mix_dev/*'
                       , help="clean voice data path")

###   data format setting
train_arg.add_argument('--shuffle_data_time', type=int, default=3, help="shuffle training data each time")
train_arg.add_argument('--progressive', type=bool, default=False, help="progressive training model")
train_arg.add_argument('--mel_freq_num', type=int, default=100, help="input as log-mel frequency")
train_arg.add_argument('--input_frame_num', type=int, default=2, help="enhance a frame with N frame input(one side)")
train_arg.add_argument('--hop_length', type=int, default=200, help="frame shift")
train_arg.add_argument('--clean_sr', type=int, default=16000, help="clean voice sample rate")
train_arg.add_argument('--noise_sr', type=int, default=16000, help="noise sample rate")
train_arg.add_argument('--embedding_file', type=str, default='TCC300_CRNN_DVec', help="embedding file need to read")
train_arg.add_argument('--embedding_similarity_file', type=str, default='gender', help="embedding file to caculate speaker similarity")
train_arg.add_argument('--stoi_correlation_time', type=int, default=60, help="stoi evaluate correlation time step")
train_arg.add_argument('--preEnhance', type=bool, default=False, help="pre-enhance to enhance high frequency intensity")
train_arg.add_argument('--under4k', type=int, default=0, help="input model dimension")
train_arg.add_argument('--norm', type=bool, default=True, help="magnitude mormalization")
train_arg.add_argument('--wienerFilter', type=bool, default=True, help="wienerFilter preprocessing on training data")

### dataset
train_arg.add_argument('--mozilla_train_path', type=str, default='/AudioProject/dataset/mozilla_speech_tainwan/zh-TW/clips/train_wav/*.wav'
                       , help="clean voice data path")
train_arg.add_argument('--librispeech_train_path', type=str, default='/AudioProject/dataset/LibriSpeech/train-clean-100/*/*/*.flac'
                       , help="clean voice data path")
train_arg.add_argument('--vocalset_train_path', type=str, default='/AudioProject/dataset/vocalset_singing/train/*/*/*/*.wav'
                       , help="clean voice data path")
train_arg.add_argument('--mozilla_dev_path', type=str, default='/AudioProject/dataset/mozilla_speech_tainwan/zh-TW/clips/dev_wav/*.wav'
                       , help="clean voice data path")
train_arg.add_argument('--librispeech_dev_path', type=str, default='/AudioProject/dataset/LibriSpeech/dev-clean/*/*/*.flac'
                       , help="clean voice data path")
train_arg.add_argument('--vocalset_dev_path', type=str, default='/AudioProject/dataset/vocalset_singing/dev/*/*/*/*.wav'
                       , help="clean voice data path")

train_arg.add_argument('--audioset_path', type=str, default='/AudioProject/dataset/Audioset/audioset-processing-master/output/pure_noise/*'
                       , help="clean voice data path")


### data preprocessing type (choose one)
train_arg.add_argument('--CRNN', type=bool, default=True, help="CRNN denoise model")
train_arg.add_argument('--CRNN_separation', type=bool, default=False, help="CRNN separation model")
train_arg.add_argument('--RNN', type=bool, default=False, help="RNN denoise model")

###   model setting
train_arg.add_argument('--min_delta_loss', type=float, default=0.001, help="save model delta loss threshold")
train_arg.add_argument('--init_learning_rate', type=float, default=0.001, help="initial learning rate")
train_arg.add_argument('--epochs', type=int, default=1000, help="iteration times")
train_arg.add_argument('--batch_size', type=int, default=128, help="iteration times")
train_arg.add_argument('--audio_batch', type=int, default=128, help="a training batch size of audio")
train_arg.add_argument('--is_training', type=bool, default=True, help="is training or not")

config = get_config()
print(config)           # print all the arguments
