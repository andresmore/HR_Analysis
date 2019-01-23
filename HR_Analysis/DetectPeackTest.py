import sys
import pprint

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import wfdb
from wfdb import processing
from oct2py import octave
import QRSDetectorOffline


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


octave.addpath('matlab')
octave.eval('pkg load signal')

# signal = np.load('signal.np.npy')
record2 = wfdb.rdrecord('p000107-2121-11-30-20-03', pb_dir='mimic3wdb/matched/' + 'p00/p000107/', channel_names=['II'])
signal = record2.p_signal.reshape(-1)
if len(signal) >= 1000000:  # last 4.4 hours
    signal = signal[-1000000:]
a = octave.test_nout(signal, 2, nout=10)
print(a)

xqrs = processing.XQRS(sig=signal, fs=125)
xqrs.detect()

# wfdb.plot_items(signal=signal, ann_samp=[xqrs.qrs_inds], fs=125, figsize=(18,8))

# [R_i,R_amp,S_i,S_amp,T_i,T_amp]=octave.peakdetect(signal,125,5, nout=6)
# wfdb.plot_items(signal=signal, ann_samp=[R_i,S_i,T_i], fs=125, figsize=(18,8))
print(signal)
R, Q, S, T, P_w = octave.MTEO_qrst(signal, 125, False, nout=5, verbose=True)

R = R[:, 0].astype(int)
Q = Q[:, 0].astype(int)
S = S[:, 0].astype(int)
T = T[:, 0].astype(int)
plot_detection_data(signal, R, Q, S, T, xqrs.qrs_inds)

# wfdb.plot_items(signal=signal, ann_samp=[R,Q,S,T], fs=125, figsize=(18,8))

# qrs_detector = QRSDetectorOffline.QRSDetectorOffline(signal, verbose=True,
#                                  log_data=False, plot_data=True, show_plot=True)
