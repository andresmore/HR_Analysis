#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 18:09:40 2019

@author: paula
"""
import wfdb
import numpy as np
import pandas as pd
import urllib.request

#get from mimic relevant stays
record_list= wfdb.get_record_list('mimic3wdb/matched')
patient_ids=list(map(lambda x: int(x[x.rfind('p')+1:x.rfind('/')]), record_list ))
print(len(record_list))

all_data=[]
for i, record in enumerate(record_list):
    files=list(filter( lambda x : x.startswith('p') and not x.endswith('n'),wfdb.get_record_list('mimic3wdb/matched/'+record)))
    subject_id = patient_ids[i]
    for file in files:
        #need start example name 'p000107-2121-11-30-20-03'
        start_dt = pd.to_datetime(file[8:])
        #need end
        #url https://physionet.org/physiobank/database/mimic3wdb/matched/
        contents = urllib.request.urlopen("https://physionet.org/physiobank/database/mimic3wdb/matched/"+record+file+".hea").readline()
        #get first line, split by space get 4th records
        length = contents.decode().split(" ")[3]
        #record2 = wfdb.rdrecord(files[0], pb_dir='mimic3wdb/matched/'+record_list[6], channel_names=['II'])
        data = [file, subject_id, start_dt, length]
        all_data.append(data)
    
all_data = np.asarray(all_data).reshape((len(all_data),4))
all_data = pd.DataFrame(all_data)