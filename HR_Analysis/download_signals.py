#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:00:30 2019

@author: paula
"""

import wfdb
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter

from oct2py import octave
octave.addpath('matlab')
octave.eval('pkg load signal')

#Read data
dead_signals = pd.read_csv('died_data.csv', index_col=[0])




def butter_bandpass(lowcut, highcut, fs, order=5):
    #nyq = 0.5 * fs
    low = lowcut *2.0/ fs
    high = highcut *2.0/ fs
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_distances(Q,R,S,T, use_rows):
    '''
    Receives qrst in vector form
    @return Distances vector rr qr rs st and T wave height
    '''
    print(use_rows)
    rr = np.diff(R[:use_rows,])
    print(rr.shape)
    qr = R[:use_rows,] - Q[:use_rows,]
    print(qr.shape)
    rs = S[:use_rows,] - R[:use_rows,]
    print(rs.shape)
    st = T[:use_rows,] - S[:use_rows,]
    print(st.shape)
    t = T[:use_rows,]
    print(t.shape)
    resp = np.concatenate((rr, qr, rs, st, t))
    return resp.reshape((-1, 5), order='F')
    


for _, row in dead_signals.iterrows():
   #Download signal 
   print(row.file[:3]+'/'+row.file[:7]+'/')
   #'p000107-2121-11-30-20-03'
   dir_path = row.file[:3]+'/'+row.file[:7]+'/'
   record = wfdb.rdrecord(row.file, pb_dir='mimic3wdb/matched/' + dir_path, sampfrom=row.download_start_idx, sampto=row.download_end_idx, channel_names=['II'])
   #Filter
   print('Downloaded')
   signal = record.p_signal
   #Note: we don't expect nan because it has been sampled ... but? 
   fs = record.fs
   lowcut = 5.0
   highcut = 15.0
   
   filtered_ecg = butter_bandpass_filter(signal.reshape(-1), lowcut, highcut, fs, order=3)
   filtered_ecg = np.nan_to_num(filtered_ecg)
    #Detect peaks
   R, Q, S, T, P_w = octave.MTEO_qrst(filtered_ecg, 125, False, nout=5, verbose=True)

   R = np.asarray(R).astype(int)[:, 0]
   Q = np.asarray(Q).astype(int)[:, 0]
   S = np.asarray(S).astype(int)[:, 0]
   T = np.asarray(T).astype(int)[:, 0]
   use_rows = np.min([len(R), len(Q), len(S), len(T)])
   print('Peaks detected')
   #Get distances
   dist_vector = get_distances(Q,R,S,T, use_rows)

   #Save signal and dist_vector where?
   np.save('signals/died/'+file+'_signal.npy', signal)
   np.save('signals/died/'+file+'_dist_vector.npy', dist_vector)
   print('Saved')