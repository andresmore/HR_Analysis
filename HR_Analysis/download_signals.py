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
from tqdm import tqdm
from time import sleep
from multiprocessing import Pool,freeze_support


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


def get_distances(Q, R, S, T, use_rows, ecg):
    '''
    Receives qrst in vector form
    @return Distances vector rr qr rs st and T wave height
    '''
    print(use_rows)
    rr = np.diff(R[:use_rows+1, ])
    print(rr.shape)
    qr = R[:use_rows, ] - Q[:use_rows, ]
    print(qr.shape)
    rs = S[:use_rows, ] - R[:use_rows, ]
    print(rs.shape)
    st = T[:use_rows, ] - S[:use_rows, ]
    print(st.shape)
    t = ecg[T[:use_rows, ]]
    print(t.shape)
    resp = np.concatenate((rr, qr[1:], rs[1:], st[1:], t[1:]))
    return resp.reshape((-1, 5), order='F')

def download_row(row):

    try:
        # Must init in each thread oct2py
        octave.addpath('matlab')
        octave.eval('pkg load signal')
        row=row[1]
        # Download signal
        print(row.file[:3] + '/' + row.file[:7] + '/')
        # 'p000107-2121-11-30-20-03'
        dir_path = row.file[:3] + '/' + row.file[:7] + '/'
        record = wfdb.rdrecord(row.file, pb_dir='mimic3wdb/matched/' + dir_path, sampfrom=row.download_start_idx,
                               sampto=row.download_end_idx, channel_names=['II'])
        # Filter
        print('Downloaded')
        signal = record.p_signal
        # Note: we don't expect nan because it has been sampled ... but?
        fs = record.fs
        lowcut = 5.0
        highcut = 15.0

        filtered_ecg = butter_bandpass_filter(signal.reshape(-1), lowcut, highcut, fs, order=3)
        filtered_ecg = np.nan_to_num(filtered_ecg)
        print('Filtered, detecting peaks')
        # Detect peaks
        R, Q, S, T, P_w = octave.MTEO_qrst(filtered_ecg, 125, False, nout=5, verbose=True)

        R = pd.DataFrame(R, columns=["peak","sig_value","complex_id"]).astype({"peak":int,"sig_value":float,"complex_id":int})
        R['type']='R'
        Q = pd.DataFrame(Q, columns=["peak","sig_value","complex_id"]).astype({"peak":int,"sig_value":float,"complex_id":int})
        Q['type'] = 'Q'
        S = pd.DataFrame(S, columns=["peak","sig_value","complex_id"]).astype({"peak":int,"sig_value":float,"complex_id":int})
        S['type'] = 'S'
        T = pd.DataFrame(T, columns=["peak","sig_value","complex_id"]).astype({"peak":int,"sig_value":float,"complex_id":int})
        T['type'] = 'T'
        P_w= pd.DataFrame(P_w, columns=["peak","sig_value","complex_id"]).astype({"peak":int,"sig_value":float,"complex_id":int})
        P_w['type'] = 'P_w'


        common_complex_id= set(R.complex_id).intersection(set(Q.complex_id)).intersection(set(S.complex_id)).intersection(set(T.complex_id))

        R_filter = R[R['complex_id'].isin(common_complex_id)]
        Q_filter = Q[Q['complex_id'].isin(common_complex_id)]
        S_filter = S[S['complex_id'].isin(common_complex_id)]
        T_filter = T[T['complex_id'].isin(common_complex_id)]

        use_rows = np.min([len(R_filter.peak.values), len(Q_filter.peak.values), len(S_filter.peak.values), len(T_filter.peak.values)])
        print('Peaks detected')
        # Get distances
        dist_vector = get_distances(Q_filter.peak.values, R_filter.peak.values, S_filter.peak.values, T_filter.peak.values, use_rows, filtered_ecg)

        # Save signal and dist_vector where?
        np.savez_compressed('signals/died/' + row.file + '_signal.npz', signal)
        pd.concat([R, Q, S, T, P_w]).to_pickle('signals/died/' + row.file + '_peaks.pkl')
        #np.save('signals/died/' + row.file + '_peaks.npz', np.concatenate((Q.peak.values, R.peak.values, S.peak.values, T.peak.values, P_w.peak.values)).reshape((-1, 5), order='F'))
        np.save('signals/died/' + row.file + '_dist_vector.npz', dist_vector)
        print('Saved')
    except Exception as e:
        print(str(e))


if __name__ == '__main__':
    freeze_support()  # for Windows support


    # Read data
    dead_signals = pd.read_csv('died_data.csv', index_col=[0])
    L = 6

    p = Pool(L)
    with tqdm(total=len(dead_signals)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(download_row, dead_signals.iterrows()))):
            pbar.update()
    pbar.close()
    p.close()
    p.join()
    #for _, row in tqdm(dead_signals.iterrows()):

