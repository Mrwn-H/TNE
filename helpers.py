import os.path
import pyxdf
import pandas as pd
import mne
from os import path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tkinter import Tk
from tkinter.filedialog import askdirectory
import glob, os
from autoreject import get_rejection_threshold
import promptlib
import autoreject
import copy
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import mne
import pandas as pd
import copy

from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
import helpers
from helpers import *
import importlib
import scipy
from scipy import signal
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from operator import itemgetter
from skfeature.function.similarity_based import fisher_score

'''
Constants
'''
TIME_OFFSET = -1  # in sec, the event window begins x seconds before the time stamp
DUR = 5 # in sec, duration of the event window in seconds
EVENTS = {
    'Baseline' : None,
    'MI' : 'MI trial marker',
    'PO' : 'Unity.Marker.TargetID'
}
EPOCH_VAR = {
    'TIME_OFFSET' : TIME_OFFSET,
    'DUR' : DUR,
    'EVENTS' : EVENTS



}
MONTAGE = 'DATA/montage_perfect.csv'
ICA_RECAP = 'DATA/ica_recap.csv'


'''
Preprocessing
'''

def select_stream(stream_data, selection='type',desired_type='EEG'):
    
    for stream in stream_data:
        stream_type = stream['info'][selection][0]
        
        if stream_type == desired_type and len(stream['time_series'])>0:
            return stream
        else:
            continue

def show_epoch(epochs,n_epochs=10):
    if epochs != None:
        print(epochs)
        print(epochs.events)
        epochs.plot(n_epochs=n_epochs,events=epochs.events, event_id=epochs.event_id)


        
def select_eegs(xdf_files):
    
    eeg_files = []

    for file in xdf_files:
        print('Processing {}'.format(file))
        data,header = pyxdf.load_xdf(file);
        eeg_file = select_stream(data);
        eeg_file['file'] = os.path.basename(os.path.normpath(file));
        eeg_files.append(eeg_file);
    
    return eeg_files

def raw_processing(raw,filtering,bad_chs,fmin=0.1,fmax=40):
    signal = raw.copy()
    
    if isinstance(bad_chs,str):
        bad_chs = bad_chs.split(',') + ['M1','M2','EOG']
    else:
        bad_chs = ['M1','M2','EOG']
    
    if bad_chs.count('CPZ') > 0:
        bad_chs.remove('CPZ')
        
    signal.info['bads'].extend(bad_chs)

    if "BP" in filtering:
        signal = signal.filter(fmin,fmax)

    if filtering == 'rawBPCAR':
        signal, ref_data = mne.set_eeg_reference(signal, ref_channels='average', copy=True)
    
    return signal
    
def to_mne(data_path, filtering, NB_CHANNELS = 64):
    stream,header = pyxdf.load_xdf(data_path)
    stream_eeg = select_stream(stream)

    if stream_eeg is None:
        print("No EEG for " + data_path)
        signal = None;
        eventss = None;
        event_idss = None;
        return signal, eventss, event_idss
    
    ch_names = []
    channels = stream_eeg['info']['desc'][0]['channels'][0]['channel']
    file = os.path.basename(os.path.normpath(data_path))

    for channel in channels:
        current_label = channel['label'][0]
        if (current_label != 'AUX3') & (current_label != 'TRIGGER'):
            ch_names.append(channel['label'][0])
        else:
            continue

    data = stream_eeg["time_series"].copy().T;  # data.shape = (nb_channels, nb_samples)
    data = data[0:64,:] # We do not keep data for AUX3 and TRIGGER
    data *= 1e-6  # convert from uV to V (mne uses V)
    #assert len(data) == NB_CHANNELS      
    sfreq = float(stream_eeg["info"]["nominal_srate"][0])
    ch_types = ['eeg'] * 64;
    info = mne.create_info(ch_names, sfreq, ch_types);
    # info = mne.create_info(NB_CHANNEfLS, sfreq)
    raw = mne.io.RawArray(data, info);

    ### Filtering and CAR
    subject = file.split('_')[0];
    bad_chs = get_bad_chs(subject);
    
    signal = raw_processing(raw,filtering,bad_chs);
    

    ### Epoch management
    
    
    event_keys = list(EVENTS.keys());
    event_type = [key for key in event_keys if key in file][0];
    print('Subject: {}, EVENT: {}'.format(subject,event_type))
    if event_type == 'PO':
        
        event_id = {
            'Left':1,
            'Right':2,
            'Third':3
        }
        stream_event = select_stream(stream,'name',EVENTS[event_type]);
        if stream_event is None:
            eventss = None;
            event_idss = None;
        else:
            target_nb = np.array(stream_event['time_series']).squeeze();  # array with target IDs
            t_origin = stream_eeg['time_stamps'][0];
            t_stamps = stream_event['time_stamps'] - t_origin;
            annots = mne.Annotations(onset=t_stamps, duration = 0., description=target_nb);
            raw_annotated = signal.copy().set_annotations(annots);
            signal = raw_annotated
            eventss, event_idss = mne.events_from_annotations(raw_annotated);
        '''
        reject = get_rejection_threshold(epochs)
        epochs = mne.Epochs(raw_annotated, eventss, event_id, tmin = -2, tmax = 7, baseline=(-2,0), reject=dict(eeg=reject), proj=False)
        '''
        
    elif event_type == 'MI':
        '''
        t_origin = stream_eeg['time_stamps'][0]
        stream_event = select_stream(stream,'name',EVENTS[event_type])
        time_stamps = stream_event['time_stamps'] - t_origin  # signal begins at 0s
        target_nb = np.array(stream_event['time_series']).squeeze()  # array with target IDs
        TIME_OFFSET = -1
        annots = mne.Annotations(onset=time_stamps, duration=0., description=target_nb)
        rawannot = rawBPCAR.copy().set_annotations(annots)


        #### Create events from annotations

        events, event_id = mne.events_from_annotations(rawannot)
        # print(event_id)
        # print(events)


        #### Create mne.Epochs with events and event_ID
        # also define the event window 

        epochs = mne.Epochs(rawBPCAR, events, event_id, tmin=TIME_OFFSET, tmax=DUR, baseline=(TIME_OFFSET,0), reject=dict(eeg=400e-6))
        '''
        
        stream_event = select_stream(stream,'name',EVENTS[event_type]);
        if stream_event is None:
            eventss = None;
            event_idss = None;
        else:
            target_nb = np.array(stream_event['time_series']).squeeze();  # array with target IDs
            t_origin = stream_eeg['time_stamps'][0];

            n_trial = int(len(stream_event['time_stamps'])/30);
            t_stamps = np.empty(0)
            for j in range(n_trial):
                ts = [stream_event['time_stamps'][30*j] - t_origin + 9* i for i in range(30)];
                t_stamps = np.concatenate((t_stamps, ts));
            '''
            ts_1 = [stream_event['time_stamps'][0] - t_origin +2 + 9* i for i in range(30)];
            ts_2 = [stream_event['time_stamps'][30] - t_origin +2 + 9* i for i in range(30)];
            t_stamps = np.concatenate((ts_1, ts_2));
            '''
            t_onset = t_stamps;
            
            annots = mne.Annotations(onset=t_onset, duration = 0., description=target_nb);  
            
            
            
            raw_annotated = signal.copy().set_annotations(annots);
            signal = raw_annotated
            #reject = get_rejection_threshold(signal)
            eventss, event_idss = mne.events_from_annotations(raw_annotated);
            
        '''
        
        epochs = mne.Epochs(raw_annotated, eventss, event_idss, tmin = -2, tmax = 7, baseline=(-2,0), reject=dict(eeg=reject), proj=False)
        '''
    else:
        eventss = None;
        event_idss = None;
        #epochs = None


    return signal, eventss, event_idss

def get_epochs(EEG_dict,EVENTS_dict,baseline=None,shift=0):
    ### Epoch management
    
    conditions = list(EEG_dict.keys())
    
    for condition in conditions:
 
        signal = EEG_dict[condition]['signal']
        eventss = EVENTS_dict[condition]['events']
        event_idss = EVENTS_dict[condition]['events_ids']
        
        if "Baseline" in condition:
            epochs = None
        else:
            tmin = -2
            tmax = 7
            epochs = mne.Epochs(signal, eventss, event_idss, tmin = tmin - 0.5, tmax = tmax + 0.5, baseline=baseline)
            reject = get_rejection_threshold(epochs)
            epochs = mne.Epochs(signal, eventss, event_idss, tmin = tmin - 0.5, tmax = tmax + 0.5, baseline=baseline,reject=reject)

        
        EEG_dict[condition]['epochs'] = epochs

    
    return EEG_dict

def read_file(path,filtering='rawBPCAR',mode='all'):
    
    initial_directory = os.getcwd()
    xdf_files = []

    for file in glob.glob(path+"\*.xdf"):
        if 'MIpre' in file or 'MIpost' in file:
            xdf_files.append(file)
        

    EEG_dict = {}
    EVENTS_dict = {}
    for file in xdf_files:
        mne_data,events,events_idss = to_mne(file,filtering)

        if mne_data is None:
            continue
        else:
            file_name = os.path.basename(file).split('.')[0]
            EEG_dict[file_name] = {}
            EEG_dict[file_name]['signal'] = mne_data
            #EEG_dict[file_name]['epochs'] = epochs
            EVENTS_dict[file_name] = {}
            EVENTS_dict[file_name]['events'] = events
            EVENTS_dict[file_name]['events_ids'] = events_idss
    
    EEG_dict_corrected = {}
    ICA_dict = {}
    EEG_dict_RAW = copy.deepcopy(EEG_dict)

    for condition in list(EEG_dict.keys()):
        ICA_dict[condition] = mne.preprocessing.read_ica(path+'/ICA_data/'+condition+'_ica')
        
    EEG_dict_corrected = apply_ica(EEG_dict,ICA_dict,mode)
    
    EEG_dict_corrected_CAR = copy.deepcopy(EEG_dict_corrected)
    EEG_dict_corrected_CAR = reject_off_center(EEG_dict_corrected_CAR)
    for condition in list(EEG_dict.keys()):
        signal = EEG_dict_corrected_CAR[condition]['signal']
        #signal.info['bads'].extend(['C6','C5'])
        signal, ref_data = mne.set_eeg_reference(signal, ref_channels='average', copy=True)
        EEG_dict_corrected_CAR[condition]['signal'] = signal


    
    #save_fif(EEG_dict_epoched)
    return EEG_dict_RAW,EEG_dict_corrected,EEG_dict_corrected_CAR,EVENTS_dict

def save_fif(EEG_dict):
    EEG_keys = list(EEG_dict.keys())
    subject_ID = EEG_keys[0].split('-')[1].split('_')[0]
    
    if int(subject_ID[1:3]) < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
    
    path = "Data/" + grp + "/" + subject_ID + "/Fif" 
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
    for condition in EEG_keys:
        EEG_dict[condition]['signal'].save(path + condition + '-signal.fif' ,overwrite=True)
        
        if EEG_dict[condition]['epochs'] != None:
            EEG_dict[condition]['epochs'].save(path + condition + '-epochs.fif',overwrite=True)
    
def save_bad_epochs(EEG_dict):
    EEG_keys = list(EEG_dict.keys())
    subject_ID = EEG_keys[0].split('-')[1].split('_')[0]
    
    if int(subject_ID[1:3]) < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
    
    path = "Data/" + grp + "/" + subject_ID + "/Bad_epochs/" 
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
    for condition in EEG_keys:
        epoch = EEG_dict[condition]['epochs'].copy()
        epoch.drop_bad()
        bad_epochs = epoch.plot_drop_log()
        
        bad_epochs.savefig(path+condition+'_bad_epochs.png')

def get_montage(csv_file, scale_factor=0.095):
    """
    Get the montage of the EEG data
    :param csv_file: csv file with the montage
    :param scale_factor: scale factor for the montage
    :return: montage
    """
    df_montage = pd.read_csv(csv_file)
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(df_montage['ch_name'], df_montage[['x', 'y', 'z']].values*scale_factor)), coord_frame='head')
    # montage.plot();
    return montage

def interpolate_bad_channels(raw, montage, bad_channels=[]):
    """
    Interpolate bad channels
    :param raw: raw data
    :param bad_channels: list of bad channels
    :return: raw data with interpolated bad channels
    """
    if bad_channels != []:
        raw.info['bads'].extend(bad_channels)
        raw.set_montage(montage)
        print("Interpolating channels: ", raw.info['bads'], "exclude M1, M2, Trigger")
        raw_interp = raw.copy().interpolate_bads(exclude=['M1', 'M2', 'Trigger'])
        raw_interp.plot(scalings=0.00013); 
        # print list of interpolated channels
        return raw_interp
    else:
        print("No bad channels to interpolate")
        return raw

def ica_analysis(signal,n_components=20,random_seed=0):
    ica = mne.preprocessing.ICA(n_components=20,random_state=0)
    signal.set_montage(get_montage(MONTAGE),on_missing='warn')
    ica.fit(signal.copy().filter(0.1,40))
    ica.plot_components()
    ica.plot_sources(signal.copy().filter(0.1,40))
    return ica

def get_bad_chs(subject):
    ch_recap = pd.read_csv("Data\ch_recap_ica.csv")
    bad_chs = ch_recap[ch_recap['group'] == subject]['interpolated_channels'].squeeze()
    if not isinstance(bad_chs,str):
        bad_chs = [];
    
    return bad_chs

def get_ica(EEG_dict):
    conditions = list(EEG_dict.keys())
    ICA_dict ={}
    for condition in conditions:
        ica = mne.preprocessing.ICA(n_components=25,random_state=0);
        signal = EEG_dict[condition]['signal'].copy();
        signal.set_montage(get_montage(MONTAGE),on_missing='warn');
        ica.fit(signal.copy());
        ICA_dict[condition] = ica.copy();

    return ICA_dict

def apply_ica(EEG_dict,ICA_dict,mode='all'):
    ica_recap = pd.read_csv(ICA_RECAP)
    EEG_keys = list(EEG_dict.copy().keys())

    EEG_dict_corrected = copy.deepcopy(EEG_dict.copy())
    for condition in EEG_keys:
        bad_comp = ica_recap[ica_recap['condition'] == condition]['eye_blink'].copy().values.squeeze().tolist().split(',')
        if mode == 'all':
            bad_comp = bad_comp + ica_recap[ica_recap['condition'] == condition]['other'].copy().values.squeeze().tolist().split(',')
        bad_comp = [x for x in bad_comp if str(x) != 'None']
        bad_comp = [int(x) for x in bad_comp]
        #bad_comp = [int(x) for x in bad_comp.split(',')];
        ica = ICA_dict[condition].copy()
        signal = EEG_dict[condition]['signal'].copy()
        eeg_corrected = ica.apply(signal.copy(),exclude=bad_comp)
        EEG_dict_corrected[condition]['signal'] = eeg_corrected.copy()

    return EEG_dict_corrected
    
def save_ica_imgs(ICA_dict,EEG_dict):
    conditions = list(ICA_dict.keys())
    subject_ID = conditions[0].split('-')[1].split('_')[0]
    
    if int(subject_ID[1:3]) < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
        
    path = "Data/" + grp + "/" + subject_ID + "/" + "ICA_data/"
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
        
    for condition in conditions:
        print('Subject ' + subject_ID + ': ' + condition)
        ica_comp = ICA_dict[condition].plot_components()
        ica_sources = ICA_dict[condition].plot_sources(EEG_dict[condition]['signal'].copy())
        ICA_dict[condition].save(path+condition+"_ica",overwrite=True)
        ica_comp[0].savefig(path+condition+'_ica_comp.png')
        ica_comp[1].savefig(path+condition+'_ica_comp_2.png')
        ica_sources.savefig(path+condition+'_ica_sources.png')
        
def process_all(folder_idxs,filtering="rawBP",mode='all'):
    for idx in folder_idxs:
        if idx < 11:
            path = 'Data/Group_Realistic_Arm/S'
            if idx < 10:
                path = path+'0'+str(idx)
            else:
                path = path+str(idx)
        else:
            path = 'Data/Group_Realistic_Arm_Tactile/S'+str(idx)
        print("Processing: " + path)
     
        EEG_dict,EVENTS_dict = read_file(path,False,filtering,mode);
        ICA_dict = get_ica(EEG_dict.copy())
        save_ica_imgs(ICA_dict,EEG_dict)

def get_subject(folder_idx,mode,filters=None,baseline=None):
    if folder_idx < 11:
        path = 'Data/Group_Realistic_Arm/S'
        if folder_idx < 10:
            path = path+'0'+str(folder_idx)
        else:
            path = path+str(folder_idx)
    else:
        path = 'Data/Group_Realistic_Arm_Tactile/S'+str(folder_idx)
    print("Processing: " + path)

    EEG_dict_RAW,EEG_dict_corrected,EEG_dict_corrected_CAR,EVENTS_dict = read_file(path,'rawBP',mode);
    
    EEG_filtered = {}

    if filters != None:

        conditions = list(EEG_dict_corrected.keys())
        MIPOST = [x for x in conditions if 'MIpost' in x][0]
        MIPRE = [x for x in conditions if 'MIpre' in x][0]
        for filter in list(filters.keys()):
            EEG_filtered[filter] = get_epochs(filter_dict(select_keys(EEG_dict_corrected,{MIPOST,MIPRE}),filters[filter]),EVENTS_dict,baseline)

    
    return EEG_dict_RAW,EEG_dict_corrected,EEG_dict_corrected_CAR,EEG_filtered,EVENTS_dict

def select_keys(origin_dict,keys):
    new_dict = {}
    for key in keys:
        new_dict[key] = origin_dict[key].copy()
    return new_dict

def reject_off_center(EEG_dict): 
    conditions = list(EEG_dict.keys())
    for cond in conditions:
        ch_names = EEG_dict[cond]['signal'].ch_names
        bad_ch_names = [ch for ch in ch_names if ch[0] != 'C']
        EEG_dict[cond]['signal'].info['bads'].extend(bad_ch_names)
    return EEG_dict

def reject_bad_chs(Epoch_dict,threshold=10):
    conditions = list(Epoch_dict.keys())
    print('Bad channels identification...')
    for cond in conditions:
        epochs = Epoch_dict[cond]['epochs'].copy()
        n_epochs = epochs.events.shape[0]
        epochs.drop_bad()
        ch_names = epochs.ch_names
        list1 = epochs.drop_log
        bad_chs = []
        for ch in ch_names:
            count = 0
            for entry in list1:
                count += entry.count(ch)
            percent = (count*100)/n_epochs
            if percent >= threshold:
                bad_chs.append(ch)
        Epoch_dict[cond]['epochs'].info['bads'].extend(bad_chs)
        print(f'Current condition: {cond}, removing: {bad_chs}')
    return Epoch_dict

'''
Time-Frequency Analysis
'''
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_evoked(EEG_dict,method='mean'):
    evoked_dict = dict()
    eeg_montage = mne.channels.read_custom_montage(MONTAGE)
    for condition in list(EEG_dict.keys()):
        epochs = EEG_dict[condition]['epochs']
        if epochs != None:
            event_type = [key for key in list(EVENTS_IDS.keys()) if key in condition][0];
            events = EVENTS_IDS[event_type]
            evoked = dict()
            for event in events:
                evoked_event = epochs[event].average(method=method)
                evoked_event.info.set_montage(eeg_montage)
                evoked[event] = evoked_event

            evoked_dict[condition] = evoked
    
    return evoked_dict

def get_ERDS_old(EEG_dict_epoched):
    
    freqs = np.arange(1, 40)  # frequencies from 2-35Hz
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    baseline = (-2, 0)  # baseline interval (in s)
    cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS

    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                buffer_size=None, out_type='mask')  # for cluster test
    tmin, tmax = -2, 7

    conditions = list(EEG_dict_epoched.keys())

    subject_ID = conditions[0].split('-')[1].split('_')[0]
    
    if int(subject_ID[1:3]) < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
            
    path = "Data/" + grp + "/" + subject_ID + "/" + "ERDS/"
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)

    for condition in conditions:

        if EEG_dict_epoched[condition]['epochs'] != None:
            epochs_MI = EEG_dict_epoched[condition]['epochs'].copy().pick_channels(['C1', 'C2', 'C3', 'CZ', 'C4', 'C5', 'C6'])
            tfr = tfr_multitaper(epochs_MI , freqs=freqs, n_cycles=freqs, use_fft=True,
                            return_itc=False, average=False, decim=2)
            tfr.crop(-2, 7).apply_baseline(baseline, mode="percent")
            df = tfr.to_data_frame(time_format = None, long_format = True)
            freq_bounds = {'_': 0,
                    'delta': 3,
                    'theta': 7,
                    'alpha': 13,
                    'beta': 30,
                    'gamma' : 140}
            df['band'] = pd.cut(df['freq'], list(freq_bounds.values()),
                            labels=list(freq_bounds)[1:])
            # Filter to retain only relevant frequency bands:
            freq_bands_of_interest = ['alpha', 'beta','theta','gamma']
            df = df[df.band.isin(freq_bands_of_interest)]
            df['band'] = df['band'].cat.remove_unused_categories()
            df['channel'] = df['channel'].cat.reorder_categories(('C1', 'C2', 'C3', 'C4', 'C5', 'C6'),
                                                            ordered=True)
            g = sns.FacetGrid(df, row='band', col='channel', margin_titles=True)
            g.map(sns.lineplot, 'time', 'value', 'condition', n_boot=10)
            axline_kw = dict(color='black', linestyle='dashed', linewidth=0.5, alpha=0.5)
            g.map(plt.axhline, y=0, **axline_kw)
            g.map(plt.axvline, x=0, **axline_kw)
            g.set(ylim=(-2, 6))
            g.set_axis_labels("Time (s)", "ERDS (%)")
            g.set_titles(col_template="{col_name}", row_template="{row_name}")
            g.add_legend(ncol=2, loc='lower center')
            g.fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
            g.savefig(path+condition)
        
def misc():
    EEG_dict_corrected = {}
    ICA_dict = {}
    EEG_dict_2 = EEG_dict

    for condition in list(EEG_dict.keys()):
        ICA_dict[condition] = mne.preprocessing.read_ica(path+'/ICA_data/'+condition+'_ica')
        
    EEG_dict,EEG_dict_corrected = apply_ica(EEG_dict,ICA_dict)
        
    for condition in list(EEG_dict.keys()):
        signal = EEG_dict_corrected[condition]['signal']
        signal, ref_data = mne.set_eeg_reference(signal, ref_channels='average', copy=True)
        EEG_dict_corrected[condition]['signal'] = signal


    EEG_dict_epoched = get_epochs(EEG_dict_corrected,EVENTS_dict)
    #save_fif(EEG_dict_epoched)

def filter_dict(EEG_dict,filter):

    EGG_dict_filt = copy.deepcopy(EEG_dict)

    for condition in list(EEG_dict.keys()):
        signal = EEG_dict[condition]['signal'].copy().filter(filter[0],filter[1])
        EGG_dict_filt[condition]['signal'] = signal

    return EGG_dict_filt



def eeg_square(EEG_dict):
    new_dict = copy.deepcopy(EEG_dict)

    for condition in list(new_dict.keys()):
        signal = new_dict[condition]['signal'].copy()
        new_eeg = mne.io.RawArray(np.square(signal.get_data()),signal.info)
        new_dict[condition]['signal'] = new_eeg
    return new_dict

def pfurt_erds(EEG_dict_filtered,EVENTS_dict):
    bands = list(EEG_dict_filtered.keys())
    new_dict = copy.deepcopy(EEG_dict_filtered)
    for band in bands:
        current_eeg_dict = new_dict[band]
        current_eeg_dict = eeg_square(current_eeg_dict)
        current_eeg_dict = get_epochs(current_eeg_dict,EVENTS_dict)
        new_dict[band] = current_eeg_dict
        
    return new_dict
    
def the_erds_maker(EEG_dict,EVENTS_dict,filters):
    conditions = list(EEG_dict.keys())
    bands = list(filters.keys())
    channels = ['C5','C3','CZ','C4','C6']
    MIPOST = [x for x in conditions if 'MIpost' in x][0]
    MIPRE = [x for x in conditions if 'MIpre' in x][0]
    plt.figure()
    fig , axs = plt.subplots(len(bands),len(channels))
    for band_idx,band in enumerate(bands):
        fmin = filters[band][0]
        fmax = filters[band][1]
        new_dict = copy.deepcopy(EEG_dict)
        for cond in list(new_dict.keys()):
            new_dict[cond]['signal'] = new_dict[cond]['signal'].copy().filter(fmin,fmax)
            signal = new_dict[cond]['signal'] 
            data = signal.get_data()
            time = signal.times.copy()
            for i in range(data.shape[0]):

                data[i] = filtered = lowess(data[i], time, is_sorted=True, frac=0.025, it=0)

            new_dict[cond]['signal'] = mne.io.RawArray(np.square(data),signal.info)
        new_eeg = get_epochs(new_dict,EVENTS_dict)
        for cond in list(new_eeg.keys()):
            epoch = new_eeg[cond]['epochs']
            epoch.load_data()
            epoch.subtract_evoked()
            epoch.resample(500)
            #new_eeg[cond]['epochs'] = epoch.apply_hilbert(envelope=True)
        evoked = get_evoked(new_eeg)
            
        colors = {'Left':'blue',
                'Right':'orange',
                'Third':'green'
                }
            
        for k,ch in enumerate(channels):
            axs[band_idx,k].set_title(ch)
            for j,trial in enumerate(list(evoked[MIPOST])):
                current_evoked = evoked[MIPOST][trial].copy()
                avg = current_evoked.data[current_evoked.ch_names.index(ch)]
                gfp = mne.baseline.rescale(avg, current_evoked.times, baseline=(-2, 0),mode='percent', copy=True);
                filtered = lowess(gfp, current_evoked.times, is_sorted=True, frac=0.025, it=0)

                axs[band_idx,k].plot(current_evoked.times,filtered[:,1],color=colors[trial],label=trial)
                axs[band_idx,k].grid(True)
            axs[band_idx,k].axvline(x = 0, color = 'black',linestyle="--", label = 'trial onset')
            axs[band_idx,k].legend()
        del epoch
        del evoked
        del current_evoked
    fig.tight_layout()
    fig.set_figwidth(30)
    fig.set_figheight(50)
    plt.show()
    return

def smooth(signal,time):
    print(signal.shape)
    print(time.shape)
    filtered = lowess(signal, time, is_sorted=True, frac=0.025, it=0)
    return filtered[:,1]

def t_test(signal_dict):
    sensors = list(signal_dict.keys())
    t_test_keys = ['Right_vs_Left','Right_vs_Third','Left_vs_Third']
    val_dict = {}
    for sensor in sensors:
        signals = signal_dict[sensor]
        conditions = list(signals.keys())
        val_dict[sensor] = {}
        for key in t_test_keys:
            current_conds = [x for x in conditions if x in key]
            current_dict = select_keys(signals,current_conds)
            signal_1, signal_2 = itemgetter(current_conds[0], current_conds[1])(current_dict)
            value = scipy.stats.ttest_ind(signal_1, signal_2, axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
            val_dict[sensor][key] = value[1]
            
    return val_dict

def t_test_2(eeg_dict):
    test_dict = {}
    conditions = list(eeg_dict.keys())
    channels = ['C5','C3','CZ','C4','C6']
    t_test_keys = ['Right_vs_Left','Right_vs_Third','Left_vs_Third']
    val_dict = {}
    for condition in conditions:
        test_dict[condition] = {}
        epochs = eeg_dict[condition]['epochs'].copy()
        for ch in channels:
            
            test_dict[condition][ch] = {}
            
            epoch_key = list(epochs.event_id.keys())
            for key in t_test_keys:
                
                current_conds = [x for x in epoch_key if x in key]
                signal_1 = epochs[current_conds[0]].get_data(picks=[ch])
                signal_2 = epochs[current_conds[1]].get_data(picks=[ch])
                n_times = signal_1.shape[-1]
                comp = np.zeros((n_times,))
                for n in range(n_times):
                    value = scipy.stats.ttest_ind(signal_1[:,:,n], signal_2[:,:,n], axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
                    comp[n] = value[1]/(2*n_times)
                test_dict[condition][ch][key] = comp
    return test_dict

def t_test_3(tfr,freqs_dict):
    freqs = list(freqs_dict.keys())
    channels = tfr.ch_names
    trials = list(tfr.event_id.keys())
    values_dict = {}
    t_test_keys = ['Right_vs_Left','Right_vs_Third','Left_vs_Third']
    for freq in freqs:
        values_dict[freq] = {}
        freq_mask = (tfr.freqs >= freqs_dict[freq][0]) & (tfr.freqs <= freqs_dict[freq][1])
        for ch in channels:
            tfr_dict = {}
            values_dict[freq][ch] = {}
            for trial in trials:
                current_tfr = tfr.copy().pick([ch])
                selection = current_tfr[trial].data[:,:,freq_mask,:]
                avg = np.mean(selection,axis=2)
                tfr_dict[trial] = avg
            for key in t_test_keys:
                
                current_conds = [x for x in trials if x in key]
                signal_1 = tfr_dict[current_conds[0]]
                signal_2 = tfr_dict[current_conds[1]]
                n_times = signal_1.shape[-1]
                comp = np.zeros((n_times,))
                for n in range(n_times):
                    value = scipy.stats.ttest_ind(signal_1[:,:,n], signal_2[:,:,n], axis=0, equal_var=True, nan_policy='propagate', permutations=None, random_state=None, alternative='two-sided', trim=0)
                    comp[n] = value[1]/(2*n_times)
                values_dict[freq][ch][key] = comp
    return values_dict

def apply_baseline(epoch,times,baseline):
    gfp = mne.baseline.rescale(epoch, times, baseline=baseline,mode='percent', copy=True)
    return gfp

def get_power(signal):
    power = np.mean(np.square(np.absolute(signal)))
    return power

def compute_f_score(inputs,labels):
    # Compute the number of features and number of samples
    n_samples = len(labels)
    n_features = inputs.shape[1]

    # Compute the number of classes
    classes = np.unique(labels)
    n_classes = len(classes)

    # Compute feature mean
    feature_mean = np.nanmean(inputs, axis=0)

    # Compute mean and std for features class-wise
    class_means = np.zeros((n_classes, num_features))
    class_var = np.zeros((n_classes, num_features))

    for i,c in enumerate(classes):
        class_means[i] = np.nanmean(inputs[labels == c], axis=0)
        class_var[i] = np.nanvar(inputs[labels == c], axis=0)
    
    f_scores = np.zeroes(n_features,1)

    for n in n_features:
        f_scores[n] = ((class_means[0]))
def fisher_score_matrix(inputs, labels):
    """
    Computes the Fisher score matrix given the inputs and labels.

    Parameters:
    inputs (numpy.ndarray): An array of input values.
    labels (numpy.ndarray): An array of labels.

    Returns:
    fisher_score (numpy.ndarray): The Fisher score matrix.
    """
    # Compute the number of features and number of samples
    num_features = inputs.shape[1]
    num_samples = inputs.shape[0]

    # Compute the mean and standard deviation of the inputs
    input_mean = np.nanmean(inputs, axis=0)
    input_std = np.nanstd(inputs, axis=0)

    # Standardize the inputs
    inputs = (inputs - input_mean) / input_std

    # Compute the class means
    classes = np.unique(labels)
    class_means = np.zeros((3, num_features))

    for c in classes:
        class_means[c-1] = np.nanmean(inputs[labels == c], axis=0)

    # Compute feature mean
    feature_mean = np.nanmean(inputs, axis=0)

    # Compute the within-class scatter matrix
    within_class_scatter = np.zeros((num_features, num_features))
    for c in classes:
        class_inputs = inputs[labels == c]
        class_diff = class_inputs - class_means[c-1]
        within_class_scatter += np.dot(class_diff.T, class_diff)

    # Compute the between-class scatter matrix
    between_class_scatter = np.zeros((num_features, num_features))
    for c in classes:
        between_class_scatter += (np.outer(class_means[c-1] - input_mean, class_means[c-1] - input_mean) * len(inputs[labels == c]))

    # Compute the Fisher score matrix
    fisher_score = np.dot(np.linalg.inv(within_class_scatter), between_class_scatter)

    return fisher_score

def fisher_analysis(EEG_dict,EVENTS_dict,mode='triple'):
    
    freqs = np.arange(4,41,2)

    conditions = list(EEG_dict.keys())
    save_path = get_dict_path(EEG_dict)
    save_path = save_path + '/Fisher'
    isExist = os.path.exists(save_path)
    if not isExist:
        os.makedirs(save_path)
    f_score_dict = {}
    bandpower_dict = {}
    for cond in conditions:
        bandpower_dict[cond] = {}
        epochs = EEG_dict[cond]['epochs'].copy()
        epochs.load_data()
        epochs.crop(2,7)
        epochs.resample(500)
        picks = [ch for ch in epochs.ch_names if ch not in epochs.info['bads']]
        fft = mne.time_frequency.psd_array_multitaper(epochs.get_data(picks=picks), epochs.info['sfreq'], fmin=0.0, fmax=40, bandwidth=2)
        bandpower = fft[0]
        current_freqs = fft[1]
        IDs = np.where(np.in1d(current_freqs,freqs))
        print(np.squeeze(bandpower[:,:,IDs]).shape)
        bandpower = np.squeeze(bandpower[:,:,IDs])
        bandpower_dict[cond]['power'] = bandpower

        x = np.squeeze(bandpower)
        X = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        print(X.shape)
        y = epochs.events[:,2].copy()
        print(y.shape)

        inputs = X
        labels = y    
        # Compute the number of features and number of samples
        n_samples = len(labels)
        n_features = inputs.shape[1]

        if mode == 'duo':
            labels[labels==2] = 1
        
        bandpower_dict[cond]['labels'] = labels
        # Compute the number of classes
        classes = np.unique(labels)
        n_classes = len(classes)

        # Compute feature mean
        feature_mean = np.nanmean(inputs, axis=0)

        # Compute mean and std for features class-wise
        class_means = np.zeros((n_classes, n_features))
        class_var = np.zeros((n_classes, n_features))

        for i,c in enumerate(classes):
            class_means[i] = np.nanmean(inputs[labels == c], axis=0)
            class_var[i] = np.nanvar(inputs[labels == c], axis=0)
            
        f_scores = np.zeros((n_features,))
        new_mean = class_means-feature_mean

        for n in range(n_features):
            f_scores[n] = np.sum(np.square(new_mean[:,n]))/np.sum(class_var[:,n])
        f_scores = f_scores.reshape(x.shape[1],x.shape[2])

        
        g = sns.heatmap(f_scores,cmap = "turbo",xticklabels=freqs, yticklabels=picks)
        plt.title('Fisher score for ' + cond, fontsize = 20) # title with fontsize 20
        plt.xlabel('Freqs [Hz]', fontsize = 15) # x-axis label with fontsize 15
        plt.ylabel('Channels', fontsize = 15) # y-axis label with fontsize 15

        plt.show()
        figure = g.get_figure()    
        figure.savefig(save_path+'/'+cond+'_fisher')
        f_score_dict[cond] = f_scores
    return bandpower_dict,f_score_dict

def quick_ERDS(epoch_dict,picked_chs = ['C3','C1','CZ','C2','C4']):
    freqs = np.arange(1, 40)  # frequencies from 2-35Hz
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    baseline = (-2, 0.)  # baseline interval (in s)
    tmin = -2.
    tmax = 7.
    colors = {'Left':'lightskyblue',
            'Right':'sandybrown',
            'Third':'springgreen'
            }
    for cond in list(epoch_dict.keys()):
        
        epochs = epoch_dict[cond]['epochs'].copy()
        epochs.load_data()
        epochs.pick(picked_chs)
        epochs.resample(500)
        
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=freqs, use_fft=True,
                            return_itc=False, average=False, decim=2)
        tfr.crop(tmin, tmax).apply_baseline(baseline, mode="percent")
        
        channels = tfr.ch_names
        trials = list(tfr.event_id.keys())
        freq_mask = (tfr.freqs >= 8) & (tfr.freqs <= 12)
        fig,axs = plt.subplots(1,len(picked_chs))
        
        for k,ch in enumerate(channels):
            for trial in trials:
                current_tfr = tfr.copy().pick([ch])
                selection = current_tfr[trial].data[:,:,freq_mask,:]
                avg = np.nanmean(np.mean(selection,axis=2),axis=0)
                axs[k].plot(tfr.times,avg[0],color=colors[trial],label = trial)
            axs[k].grid(True)
            axs[k].set_ylim([-2, 6])
            axs[k].set_xlim([-2, 7])
            axs[k].axvline(x = baseline[1], color = 'black',linestyle="--", label = 'trial onset')
            axs[k].axhline(y = 0., color = 'black',linestyle="--")
            axs[k].legend()
            axs[k].set_title(ch)
        fig.tight_layout()
        fig.set_figwidth(len(picked_chs)*10)
        fig.suptitle(cond, fontsize=10)
        fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.08)
        plt.show()
        del epochs

def get_dict_path(EEG_dict):
    EEG_keys = list(EEG_dict.keys())
    subject_ID = EEG_keys[0].split('-')[1].split('_')[0]
    
    if int(subject_ID[1:3]) < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
    
    path = "Data/" + grp + "/" + subject_ID 

    return path

def get_subject_path(subject_ID):
    
    if subject_ID < 11:
        grp = "Group_Realistic_Arm"
    else:
        grp = "Group_Realistic_Arm_Tactile"
    
    if subject_ID >= 10:
        subject_ID = "S" + str(subject_ID)
    else:
        subject_ID = "S0" + str(subject_ID)
    
    path = "Data/" + grp + "/" + subject_ID

    return path