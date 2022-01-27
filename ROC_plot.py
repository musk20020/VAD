import numpy as np
import librosa
import audio_processing as ap
import matplotlib
import matplotlib.pyplot as plt
import csv

file = open("f1_score_220124.txt", "r").readlines()

T = np.zeros([len(file)])
f1 = np.zeros([len(file)])
FP = np.zeros([len(file)])
TP = np.zeros([len(file)])
for i in range(len(file)):
    T_s, f1_s, TP_s, FP_s = file[i].replace("\n", "").split(",")
    T[i] = float(T_s)
    f1[i] = float(f1_s)
    TP[i] = float(TP_s)
    FP[i] = float(FP_s)

boundry = np.arange(0, 101)/100

plt.figure()
plt.plot(FP, TP)
plt.plot(boundry, boundry)
plt.show()