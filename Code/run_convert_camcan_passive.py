#Purpose:
    #A template script to run all subjects BIDS conversion in independent jobs, via
    #submit_jobs.py with (by prepending s= 1 etc. to this script):
    #
    # python /imaging/camcan/sandbox/ek03/fixCC700/cc700_MEG/scripts/createBIDS_MNEBIDS/Cam-CAN-to-BIDS/Code/submitJobs.py /imaging/camcan/sandbox/ek03/fixCC700/cc700_MEG/scripts/createBIDS_MNEBIDS/Cam-CAN-to-BIDS/Code/run_convert_camcan_singleSubject_smt.py /imaging/ek03/tmp/tmpJobs 645 1 s

#SETUP
#experiments = ('smt')
experiments = ('passive')
#experiments = ('rest')
exp = experiments


import pathlib
#import tqdm
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids
import os
#from mne_bids import write_meg_calibration, write_meg_crosstalk

#SETUP
mne.set_log_level(verbose=False)

input_dir = pathlib.Path('/imaging/camcan/sandbox/ek03/fixCC700/cc700_MEG/scripts/createBIDS_MNEBIDS/BIDS_symlinks')
output_dir = pathlib.Path(f'/imaging/camcan/sandbox/ek03/fixCC700/cc700_MEG/scripts/createBIDS_MNEBIDS/BIDS_MNE-BIDS_task-{exp}')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
participants = sorted([p.parts[-1] for p in input_dir.glob('sub-CC*')])

restart_from = 'sub-CC620026'
#restart_from = None
if restart_from is not None:
    idx = participants.index(restart_from)
    participants = participants[idx:]
    
date_sound_card_change = datetime(month=12, day=8, year=2011,
                                  tzinfo=timezone.utc)

event_name_to_id_mapping = {'audiovis/300Hz': 1,
                            'audiovis/600Hz': 2,
                            'audiovis/1200Hz': 3,
                            'catch/0': 4,
                            'catch/1': 5,
                            'audio/300Hz': 6,
                            'audio/600Hz': 7,
                            'audio/1200Hz': 8,
                            'vis/checker': 9,
                            'button': 99}

stim_chs = ('STI001', 'STI002', 'STI003', 'STI004')


#START
t_start = datetime.now()

for participant in participants:
    
    minDur_longEvents = 0.2
    minDur_buttonEvents = 0 #event.py default; 0.1 isnt a good alllround solution unfortunately
    button_triggerValue = 4000
    
    raw_fname = input_dir / participant / f'ses-{exp}' / 'meg' / f'{participant}_ses-{exp}_task-{exp}_meg.fif'
        
    #Skip missing subject/session
    if not raw_fname.exists():
        print(f'\nSkipping sub-{participant[4:]} ses-{exp} as non-existent\n')
        continue
        
    print(f'\nConverting sub-{participant[4:]} ses-{exp}\n')
    
    raw = mne.io.read_raw_fif(raw_fname)
    
    # Work around an acquisition bug in STI101: construct a stimulus
    # channel ourselves from STI001:STI004.
    stim_data = (raw
                 .copy()
                 .load_data()
                 .pick_channels(stim_chs)
                 .get_data())
    stim_data /= 5  # Signal is always +5V
    
    # First channel codes for bit 1, second for bit 2, etc.
    for stim_ch_idx, stim_ch in enumerate(stim_chs):
        # First we spot spurious triggers that last too long
        long_events = mne.find_events(raw, stim_channel=stim_ch,
                                      min_duration= minDur_longEvents)
        # Find all events
        this_events = mne.find_events(raw, stim_channel=stim_ch,
                                      min_duration=0.002)
        # Remove the spurious events
        this_events = this_events[~np.isin(this_events[:, 0], long_events[:, 0])]
    
        # Reconstruct a clean stim channel
        stim_data[stim_ch_idx, :] = 0
        stim_data[stim_ch_idx, this_events[:, 0] - raw.first_samp] = 1
        stim_data[stim_ch_idx, :] *= 2**stim_ch_idx
    
    # Create a virtual channel which is the sum of the individual
    # channels.
    stim_data = stim_data.sum(axis=0, keepdims=True)
    info = mne.create_info(['STI_VIRTUAL'], raw.info['sfreq'],
                           ['stim'])
    stim_raw = mne.io.RawArray(stim_data, info,
                               first_samp=raw.first_samp)
    
    events = mne.find_events(stim_raw, stim_channel='STI_VIRTUAL')
    
    print(Counter(events[:, 2]))  # uncomment for debugging
    
    if exp == 'smt':  # add button presses for the active task
        button_events = mne.find_events(raw, stim_channel='STI101', verbose=True,min_duration = minDur_buttonEvents)
        button_events = button_events[button_events[:, 2] > button_triggerValue]
        button_events[:, 2] = 99
    
        events = np.concatenate([events, button_events], axis=0)
        events = events[np.argsort(events[:, 0])]
    
        print(Counter(events[:, 2]))  # uncomment for debugging
    
    del stim_data, stim_raw, info
    
    if exp != 'rest':
        before_sound_card = (date_sound_card_change >=
                             raw.info['meas_date'])
    
        scdelay = 13 if before_sound_card else 26
    
        for event in events:
            # Apply delays in stimuli following file get_trial_info.m
            # in the Cam-CAN release file
            if event[2] in [6, 7, 8, 774, 775, 776]:
                assert exp == 'passive'
                delay = scdelay  # audio delay
            elif event[2] in [9, 777]:
                assert exp == 'passive'
                delay = 34  # visual delay
            elif event[2] in [1, 2, 3]:
                assert exp == 'smt'
                # take mean between audio and vis
                delay = (scdelay + 34) // 2
            elif event[2] in [4, 5, 99]:
                delay = 0  # catch have no delay
            else:
                raise ValueError('Trigger not found')
    
            event[0] += delay
    
    #raw.copy().pick_types(meg=False, stim=True).plot(events=events) #for debug
    #raw.copy().pick_types(meg=False, stim=True).plot()
    anonDict = {
    "daysback": 35240,
    "keep_his": False}
    
    # Now actually convert to BIDS.
    bids_path = BIDSPath(subject=participant[4:], session=exp, task=exp, datatype='meg',
                         root=output_dir)
    write_raw_bids(raw, bids_path=bids_path,
                   events_data=events,
                   event_id=event_name_to_id_mapping,
                   anonymize=anonDict,
                   overwrite=True,
                   verbose=False)

    del bids_path, raw_fname, raw, events
    
    print('Finished conversion.')
    t_end = datetime.now()
    
    print(f'Process took {t_end - t_start}.')
    