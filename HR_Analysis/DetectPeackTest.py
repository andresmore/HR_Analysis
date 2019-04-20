import pprint

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from scipy.signal import butter, lfilter

import wfdb
from wfdb import processing
from oct2py import octave

def plot_detection_data(signal, R, Q, S, T, qrs_inds):
    """
    Method responsible for plotting detection results.
    :param bool show_plot: flag for plotting the results and showing plot
    """

    def plot_data(axis, data, title='', fontsize=10):
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which='both', axis='both', linestyle='--')
        axis.plot(data, color="salmon", zorder=1)

    def plot_points(axis, values, indices, c):
        axis.scatter(x=indices, y=values[indices], c=c, s=50, zorder=2)

    plt.close('all')
    fig, axarr = plt.subplots(2, sharex=True, figsize=(15, 18))

    plot_data(axis=axarr[0], data=signal, title='Raw ECG measurements')
    plot_points(axis=axarr[0], values=signal, indices=R, c="red")
    plot_points(axis=axarr[0], values=signal, indices=Q, c="green")
    plot_points(axis=axarr[0], values=signal, indices=S, c="blue")
    plot_points(axis=axarr[0], values=signal, indices=T, c="black")
    plot_data(axis=axarr[1], data=signal, title='QRS_inds')
    plot_points(axis=axarr[1], values=signal, indices=qrs_inds, c="black")

    plt.tight_layout()
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    # nyq = 0.5 * fs
    low = lowcut * 2.0 / fs
    high = highcut * 2.0 / fs
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


octave.addpath('matlab')
octave.eval('pkg load signal')

# signal = np.load('signal.np.npy')
record2 = wfdb.rdrecord('p000020-2183-04-28-17-47', pb_dir='mimic3wdb/matched/' + 'p00/p000020/', channel_names=['II'])
signal = record2.p_signal.reshape(-1)
if len(signal) >= 1000000:  # last 4.4 hours
    signal = signal[-1000000:]

# Note: we don't expect nan because it has been sampled ... but?
signal = record2.p_signal.reshape(-1)
#use -300000 for T error 
#use -310000 for no error
start = -300000
signal = signal[start:start+100000]
fs = record2.fs
lowcut = 5.0
highcut = 15.0

filtered_ecg = butter_bandpass_filter(signal.reshape(-1), lowcut, highcut, fs, order=3)
filtered_ecg = np.nan_to_num(filtered_ecg)
print('Filtered, detecting peaks')
xqrs = processing.XQRS(sig=filtered_ecg, fs=125)
xqrs.detect()

# wfdb.plot_items(signal=signal, ann_samp=[xqrs.qrs_inds], fs=125, figsize=(18,8))

# [R_i,R_amp,S_i,S_amp,T_i,T_amp]=octave.peakdetect(signal,125,5, nout=6)
# wfdb.plot_items(signal=signal, ann_samp=[R_i,S_i,T_i], fs=125, figsize=(18,8))
R, Q, S, T, P_w = octave.MTEO_qrst(filtered_ecg, 125, False, nout=5, verbose=True)

R = R[:, 0].astype(int)
Q = Q[:, 0].astype(int)
S = S[:, 0].astype(int)
T = T[:, 0].astype(int)
print("Checking")

for i in range(len(T)):
    if R[i] < Q[i]:
        print("R , Q " + str(i))
    if S[i] < Q[i]:
        print("s , Q " + str(i))
    if S[i] < R[i]:
        print("S , R " + str(i))
    if T[i] < S[i]:
        print("T , S " + str(i))
    if T[i] > R[i+1]:
        print("T bigger than R" + str(i))
    if S[i] > R[i+1]:
        print("S bigger than R" + str(i))
print("End of check")

use_rows = np.min([len(R), len(Q), len(S), len(T)])
rr = np.diff(R[:use_rows+1, ])
st = T[:use_rows,]-S[:use_rows,]
plot_detection_data(filtered_ecg[:1300,], R[:10,], Q[:10,], S[:10,], T[:10,], xqrs.qrs_inds[:12,])

misaligned_idx=np.argwhere(T>np.roll(R,-1)[:len(T),])