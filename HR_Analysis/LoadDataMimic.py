
# coding: utf-8

# # Integración de datos MIMIC- Waveform database
# ## 1) Verificar Prerequisitos
# ### Python
# SciDB-Py requires Python 2.6-2.7 or 3.3

# In[1]:


import sys
sys.version


import pprint

pp=pprint.PrettyPrinter(indent=4)


# ### NumPy
# tested with version 1.9 (1.13.1)

# In[5]:


import numpy as np
np.__version__


import pandas as pd
pd.__version__


# ### SciPy (optional)
# tested with versions 0.10-0.12. (0.19.0) Required only for importing/exporting SciDB arrays as SciPy sparse matrices.

# In[7]:


import scipy
scipy.__version__


# ## 4) Importar WFDB para conectarse a physionet

# In[8]:


import wfdb
wfdb.__version__


# In[9]:


record_list= wfdb.get_record_list('mimic3wdb/matched')


# In[10]:


len(record_list)


# In[11]:


record_list


# In[12]:


patient_ids=list(map(lambda x: int(x[x.rfind('p')+1:x.rfind('/')]), record_list ))


# In[13]:


files=list(filter( lambda x : x.startswith('p') and not x.endswith('n'),wfdb.get_record_list('mimic3wdb/matched/'+record_list[6])))


# In[14]:


files


# In[15]:


record_list[6]


# In[16]:


files[0]


# In[17]:


subject_id = patient_ids[6]
print(subject_id)


# In[18]:


record2 = wfdb.rdrecord(files[0], pb_dir='mimic3wdb/matched/'+record_list[6], channel_names=['II'])


# In[25]:


record2


# In[26]:


wfdb.plot_items(signal=record2.p_signal[0:10000], title="Example")


# In[27]:


record2.fs


# In[28]:


len(record2.p_signal)


# In[29]:


pp.pprint(record2.__dict__)


# In[30]:


import wfdb.processing as processing


# In[31]:


import matplotlib.pyplot as plt


# In[32]:


type(record2.p_signal.shape)


# Normaliza la señal y le quita los valores en null

# In[33]:

np.save('signal.np',record2.p_signal[-10000:,0])
xqrs=processing.XQRS(sig=record2.p_signal[-10000:,0], fs=record2.fs)


# In[34]:


xqrs.detect()


# In[35]:


xqrs.qrs_inds


# In[36]:


ts= pd.Series(record2.p_signal[-8000:,0])


# In[37]:


ts.plot(figsize=(18,8))


# In[38]:


wfdb.plot_items(signal=record2.p_signal[-8000:,0], ann_samp=[xqrs.qrs_inds] ,fs=record2.fs,figsize=(18,8))


# In[39]:


np.sum(np.isnan(record2.p_signal[-8000:,0]))


# In[40]:


import QRSDetectorOffline


# In[41]:


nyquist_freq = 0.5 * 125


# In[42]:


low = 0 / nyquist_freq


# In[43]:


high = 15 / nyquist_freq


# In[44]:


high


# In[45]:


from scipy.signal import butter, lfilter


# In[46]:


b, a = butter(1, [0.0, high], btype="band")


# In[60]:


qrs_detector = QRSDetectorOffline.QRSDetectorOffline(record2.p_signal[-8000:,0], verbose=True,
                                  log_data=False, plot_data=True, show_plot=True)


# In[57]:


xqrs


# In[ ]:


def peaks_hr(sig, peak_inds, fs, title, figsize=(20, 10), saveto=None):
    "Plot a signal with its peaks and heart rate"
    # Calculate heart rate
    hrs = processing.compute_hr(sig_len=sig.shape[0], qrs_inds=peak_inds, fs=fs)
    
    N = sig.shape[0]
    
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()
    
    ax_left.plot(sig, color='#3979f0', label='Signal')
    ax_left.plot(peak_inds, sig[peak_inds], 'rx', marker='x', color='#8b0000', label='Peak', markersize=12)
    ax_right.plot(np.arange(N), hrs, label='Heart rate', color='m', linewidth=2)

    ax_left.set_title(title)

    ax_left.set_xlabel('Time (ms)')
    ax_left.set_ylabel('ECG (mV)', color='#3979f0')
    ax_right.set_ylabel('Heart rate (bpm)', color='m')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax_left.tick_params('y', colors='#3979f0')
    ax_right.tick_params('y', colors='m')
    if saveto is not None:
        plt.savefig(saveto, dpi=600)
    plt.show()


# In[ ]:


peaks_hr(sig=record2.p_signal, peak_inds=peaks_ind, fs=record2.fs,
        title="GQRS peak detection on record")


# Cambiar los guiones "-" por raya al piso "_" porque por algun motivo SciDB tiene problemas con estos caracteres
# Si el arreglo sin valores nulos no queda vacio lo sube al SciDB

# In[ ]:


## Cargar datos de pacientes, admisiones y icu_stays
patients = pd.read_csv('mimic/PATIENTS.csv')
admissions = pd.read_csv('mimic/ADMISSIONS.csv')
icu_stays = pd.read_csv('mimic/ICUSTAYS.csv')


# In[ ]:


patients.head()


# In[ ]:


#obtener admisiones del paciente
admissions['ADMITTIME']=pd.to_datetime(admissions['ADMITTIME'])
admissions['DISCHTIME']=pd.to_datetime(admissions['DISCHTIME'])
admissions['DEATHTIME']=pd.to_datetime(admissions['DEATHTIME'])


# In[ ]:


#hacer match de las fechas
date = pd.to_datetime(files[1][8:])
admission = admissions[(admissions.SUBJECT_ID==int(subject_id)) & (admissions.ADMITTIME <= date) & (admissions.DISCHTIME >= date) ]
died = admission.DEATHTIME
print(pd.isna(died))


# In[ ]:


date


# In[ ]:


record_list


# In[ ]:


import tqdm


# In[ ]:


#save into a pandas dataframe hrs and deathtime? 
df = pd.DataFrame()
list_hrs = []
list_deathtime = []
list_admission = []
#for each subject
for i, record in enumerate(record_list):
    files = list(filter( lambda x : x.startswith('p') and not x.endswith('n'),wfdb.get_record_list('mimic3wdb/matched/'+record)))
    #for each record (file)
    patient = patient_ids[i]
    for file in files:
        date = pd.to_datetime(file[8:])
        admission = admissions[(admissions.SUBJECT_ID==int(patient)) & (admissions.ADMITTIME <= date) & (admissions.DISCHTIME >= date) ]
        died = admission.DEATHTIME
        record_signal = wfdb.rdrecord(file, pb_dir='mimic3wdb/matched/'+record, channel_names=['II'])
        #record_signal = np.nan_to_num(record_signal)
        peak_inds=processing.Xqrs_detect(sig=np.nan_to_num(record_signal.p_signal[:,0]), fs=record_signal.fs)
        hrs = processing.compute_hr(sig_len=record_signal.p_signal.shape[0], qrs_inds=peak_inds, fs=record_signal.fs)
        list_hrs.append(hrs)
        list_admission.append(admission)

