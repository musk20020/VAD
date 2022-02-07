import tensorflow as tf
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
from model import REG
from os.path import join
from configuration import get_config
import librosa
import scipy
import audio_processing as ap
from glob import iglob
from tqdm import tqdm

def f1_score(predict, e, T=0.5):
    predict = predict[:, :len(e), :]
    predict_speech = predict[0,:,0]>T
    predict_noise = predict[0,:,0]<T
    speech_ground_truth = e>0
    noise_ground_truth = e==0
    TP = np.sum(predict_speech*speech_ground_truth) # speech detect to speech
    FP = np.sum(predict_speech)-TP # noise detect to speech
    TN = np.sum(predict_noise*noise_ground_truth) # noise detect to noise
    FN = np.sum(predict_noise)-TN # speech detect to noise
    precision = TP/(TP+FP)
    recall = TP/(TP+FN) # speech hit rate
    noise_hit_rate = TN/(TN+FP)
    f1_score = 2*precision*recall/(precision+recall)
    FPR = FP/(FP+TN)
    # print("F1 score : "+str(f1_score))
    return f1_score, recall, noise_hit_rate

def build_model(model_saver):
    config = get_config()
    log_path = config.log_path
    saver_dir = config.saver_path
    note = config.note
    date = config.date
    gpu_index = config.gpu_index
    model = REG(log_path, saver_dir, date, gpu_index, note, config)

    # model_saver = "220124"
    model.build(reuse=True)
    sess = model.init(model_saver)
    return model, sess

def get_f1_score(predict, clean_file, T=0.5):
    clean, _ = librosa.load(clean_file, sr=16000)
    _, e = ap.cal_frame_loudness(clean, window=256, hop_length=128)

    f1, TP, FP = f1_score(predict, e, T)
    return f1, TP, FP

def batch_test(model, sess, T=0.5):
    clean_root = "/Users/musk/dataset/TCC300_rename/Dev/"
    folder = "/Users/musk/dataset/VADTestSet/noisy/10dB/*"
    file_list = [tag for tag in iglob(folder)]
    f1_sum = 0
    TP_sum = 0
    FP_sum = 0
    for file in tqdm(file_list):
        clean_name = file.split('/')[-1].split('_')[0]
        clean_folder = clean_name[:9]
        clean_file = "{}{}/{}.wav".format(clean_root, clean_folder, clean_name)
        predict = model.predict(sess, audio_file=file)
        f1, TP, FP = get_f1_score(predict, clean_file, T)
        f1_sum+=f1
        TP_sum+=TP
        FP_sum+=FP
    f1_avg = f1_sum/len(file_list)
    TP_avg = TP_sum/len(file_list)
    FP_avg = FP_sum/len(file_list)
    # print("f1 score : "+str(f1_avg))
    return f1_avg, TP_avg, FP_avg

def wiener_filter(mag_src, mag_noise):
    max_att = 15 #15
    att_lin = 10 ** (-max_att / 20.0)
    overest = 5 #5

    S_bb = np.mean(np.square(mag_noise), axis=1)
    S_bb = overest * S_bb
    H = np.maximum(att_lin, 1 - np.divide(S_bb, np.square(mag_src)))
    filtered_src = np.multiply(H, mag_src)
    return filtered_src

if __name__=='__main__':
    model, sess = build_model(model_saver="220126")
    # file = open("f1_score_220126_NHR.txt", "w")
    for t in range(55, 57):
        f1, TP, FP = batch_test(model, sess, t/100)
        # file.write("{},{},{},{}\n".format(t/100, f1, TP, FP))
        print(1)

