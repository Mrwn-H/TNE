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
            eventss, event_idss = mne.events_from_annotations(raw_annotated);
            epochs = mne.Epochs(raw_annotated, eventss, event_id, tmin = -2, tmax = 7, baseline=(-2,0), reject=dict(eeg=400e-6), proj=False, reject_by_annotation=None);
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
                ts = [stream_event['time_stamps'][30*(j-1)] - t_origin +2 + 9* i for i in range(30)];
                t_stamps = np.concatenate((t_stamps, ts));
            '''
            ts_1 = [stream_event['time_stamps'][0] - t_origin +2 + 9* i for i in range(30)];
            ts_2 = [stream_event['time_stamps'][30] - t_origin +2 + 9* i for i in range(30)];
            t_stamps = np.concatenate((ts_1, ts_2));
            '''
            t_onset = t_stamps;
            annots = mne.Annotations(onset=t_onset, duration = 9, description=target_nb);  
            raw_annotated = signal.copy().set_annotations(annots);
            #reject = get_rejection_threshold(signal)
            eventss, event_idss = mne.events_from_annotations(raw_annotated);
            epochs = mne.Epochs(raw_annotated, eventss, event_idss, tmin = -2, tmax = 7, baseline=(-2,0), reject=dict(eeg=400e-6), proj=False, reject_by_annotation=None)
        '''
        
        epochs = mne.Epochs(raw_annotated, eventss, event_idss, tmin = -2, tmax = 7, baseline=(-2,0), reject=dict(eeg=reject), proj=False)
        '''
    else:
        eventss = None;
        event_idss = None;
        #epochs = None


    return signal, eventss, event_idss

def get_epochs(EEG_dict,EVENTS_dict):
    ### Epoch management
    
    conditions = list(EEG_dict.keys())
    
    for condition in conditions:
 
        signal = EEG_dict[condition]['signal']
        eventss = EVENTS_dict[condition]['events']
        event_idss = EVENTS_dict[condition]['events_ids']

        if "Baseline" in condition:
            epochs = None
        else:
            epochs = mne.Epochs(signal, eventss, event_idss, tmin = -2, tmax = 7, baseline=(-2,0), preload=True, reject=dict(eeg=400e-6), proj=False, reject_by_annotation=None)

        
        EEG_dict[condition]['epochs'] = epochs

    
    return EEG_dict

def read_file(path,filtering='rawBPCAR'):
    
    initial_directory = os.getcwd()
    xdf_files = []

    for file in glob.glob(path+"\*.xdf"):
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
    
    return EEG_dict,EVENTS_dict

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
    ch_recap = pd.read_csv("Data\ch_recap.csv")
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

def apply_ica(EEG_dict,ICA_dict):
    ica_recap = pd.read_csv(ICA_RECAP)
    EEG_keys = list(EEG_dict.copy().keys())

    EEG_dict_2 = EEG_dict.copy()
    EEG_dict_corrected = EEG_dict.copy()
    for condition in EEG_keys:
        bad_comp = ica_recap[ica_recap['condition'] == condition]['bad_components'].copy().squeeze()
        if bad_comp != "None":
            bad_comp = [int(x) for x in bad_comp.split(',')];
            ica = ICA_dict[condition].copy()
            signal = EEG_dict[condition]['signal'].copy()
            eeg_corrected = ica.apply(signal.copy(),exclude=bad_comp)
            EEG_dict_corrected[condition]['signal'] = eeg_corrected.copy()

    return EEG_dict_2,EEG_dict_corrected.copy()
    
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
        
def process_all(folder_idxs,filtering="rawBP"):
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
     
        EEG_dict,EVENTS_dict = read_file(path,filtering);
        ICA_dict = get_ica(EEG_dict.copy())
        save_ica_imgs(ICA_dict,EEG_dict)

def get_subject(folder_idx):
    if folder_idx < 11:
        path = 'Data/Group_Realistic_Arm/S'
        if folder_idx < 10:
            path = path+'0'+str(folder_idx)
        else:
            path = path+str(folder_idx)
    else:
        path = 'Data/Group_Realistic_Arm_Tactile/S'+str(folder_idx)
    print("Processing: " + path)

    EEG_dict,EVENTS_dict = read_file(path,'rawBP');
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
    return EEG_dict_2,EEG_dict_epoched


'''
Time-Frequency Analysis
'''
def load_subject(folder_idxs):
    for idx in folder_idxs:
        if idx < 11:
            path = 'Data/Group_Realistic_Arm/S'
            if idx < 10:
                path = path+'0'+str(idx)
            else:
                path = path+str(idx)
        else:
            path = 'Data/Group_Realistic_Arm_Tactile/S'+str(idx)
        print("Loading FIF: " + path)
        fif_files = []

        for file in glob.glob(path+"\FIF\*"):
            fif_files.append(file)

        EEG_dict = {}
        EVENTS_dict = {}

