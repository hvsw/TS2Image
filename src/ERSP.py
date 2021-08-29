import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from mne.viz.utils import center_cmap
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test

import Logger

__all__ = ["ERSP"]

class ERSP:
    def __init__(self, file_path: str, valid_cue_descriptions: list, cue_map, debug: bool = True):
        self.file_path = file_path
        self.valid_cue_descriptions = valid_cue_descriptions
        self.debug = debug
        self.cue_map = cue_map
    
    def _save_image(self, image_folder, image_file_name, image):
        all_cmaps = {
            'Perceptually Uniform Sequential' : ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Sequential' : ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 
                            'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        }

        # TODO: Testar outras cores
        default = 'viridis'
        cmaps_selected = [default]

        # set min and max ERDS values in plot
        # vmin, vmax = -1, 1  
        
        # zero maps to white
        # cmap = center_cmap(plt.cm.RdBu, vmin, vmax)
        # cmaps_selected = [cmap]
        
        for cmap_type, cmaps in all_cmaps.items():
            for cmap in cmaps:
                if cmap in cmaps_selected:
                    final_image_folder = f'{image_folder}/{cmap_type}/{cmap}'
                    if not os.path.exists(final_image_folder):
                        os.makedirs(final_image_folder)
        
                    image_path = f'{final_image_folder}/{image_file_name}.png'
                    # TODO: Add mask here https://mne.tools/stable/generated/mne.time_frequency.AverageTFR.html#mne.time_frequency.AverageTFR.plot
                    plt.imsave(image_path, image, cmap=cmap)

    def generate_images(self, output_folder: str, generate_intermediate_images: bool = False, desired_channels: list = [], merge_channels=False):
        raw = mne.io.read_raw_gdf(self.file_path, preload=True)
        raw.filter(l_freq=1, h_freq=40)

        # TODO: Filter by event_id = {"769":769}
        # Map descriptions (keys) to integer event codes (values). Only the descriptions present will be mapped, others will be ignored.
        event_ids = {"769":1, "770":2}  # map event IDs to tasks
        # event_ids = LABELS_DICTIONARY
        # event_ids = None
        
        # https://mne.tools/stable/generated/mne.events_from_annotations.html?highlight=events_from_annotations#mne.events_from_annotations
        # Get events and event_id from an Annotations object.
        # This function will assign an integer Event ID to each unique element of raw.annotations.description, 
        # and will return the mapping of descriptions to integer Event IDs along with the derived Event array.
        events, event_id = mne.events_from_annotations(raw, event_ids)
        # events.shape = (271, 3) = (n_events, 3) = the events array is [start_sample, 0?, new int from event_id converted by the lib]
        
        channels = desired_channels
        picks = mne.pick_channels(raw.info["ch_names"], channels)
        
        ##################### EPOCH DATA #####################
        # TODO: What should be tmin, tmax? I think we're cropping it wrong...
        # Start and end time of the epochs in seconds, relative to the time-locked event. Defaults to -0.2 and 0.5, respectively.
        tmin, tmax = 0, 1.25  # define epochs around events (in s)
        
        # WARNING: This prevents exceptions to be thrown if we have overlapping events and it can also drop data!!!
        # Check epocs.drop_log or something
        # Error thrown: 'Event time samples were not unique. Consider setting the `event_repeated` parameter."'
        event_repeated = None

        # TODO: time_padding to reduce border effects?
        time_padding = 0.5
        
        # https://mne.discourse.group/t/what-is-tmin-and-tmax-in-epochs/2920/1
        # “Epochs” are equal-duration chunks of the continuous raw signal. 
        # Epochs are created relative to a series of “events” (an event is a sample number plus an event ID integer encoding what kind of event it was)
        epochs = mne.Epochs(raw, events, event_ids, tmin - time_padding, tmax + time_padding,
                            picks=picks, baseline=None, preload=True, event_repeated=event_repeated)

        # Compute ERDS maps ###########################################################
        # Frequencies from 2-35Hz
        freqs = np.arange(2, 36, 1)  
        
        # TODO: What this n_cycles actually mean???
        # The number of cycles globally or for each frequency. The time-window length is thus T = n_cycles / freq.
        n_cycles = freqs # use constant t/f resolution

        # Run time-frequency decomposition overall epochs
        # Time-Frequency Representation (TFR)
        return_inter_trial_coherence = False
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                             use_fft=True, return_itc=return_inter_trial_coherence, average=False, decim=2)
        # TODO: Should we crop this? :think:
        # tfr.crop(tmin, tmax)
        
        # https://mne.tools/stable/generated/mne.time_frequency.EpochsTFR.html#mne.time_frequency.EpochsTFR.apply_baseline
        # The time interval to apply rescaling / baseline correction. 
        # If None do not apply it. 
        # If baseline is (a, b) the interval is between “a (s)” and “b (s)”. 
        # If a is None the beginning of the data is used and if b is None then b is set to the end of the interval. 
        # If baseline is equal to (None, None) all the time interval is used.
        baseline = [-1, 0]  # baseline interval (in s)
        
        # Perform baseline correction by: 
        # subtracting the mean of baseline values followed by dividing by the mean of baseline values (‘percent’)
        mode = "percent"
        tfr.apply_baseline(baseline, mode=mode) # tfr.event_id = {'10': 10, '11': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

        raw_file_name = self.file_path.split('/')[-1]
        # TODO: So at the end of the day we're justing averaging all runs for an specific event type? See tfr_ev.average()
        # This will produce less images, maybe not enough to train the CNN?

        for key_event_description, value_event_id_int in event_ids.items():
            # select desired epochs for visualization
            tfr_ev = tfr[key_event_description]
            cue_human_readable = key_event_description
            images_folder = f'{output_folder}/ERSP/{cue_human_readable}'

            # if merge_channels:
            #     images_folder += '/merged_channels'
            
            # if generate_intermediate_images: 
            #     images_folder += '/individual_channels'

            if not os.path.exists(images_folder):
                os.makedirs(images_folder)
            
            # https://mne.tools/stable/generated/mne.time_frequency.EpochsTFR.html#mne.time_frequency.EpochsTFR.average
            # Average the data across epochs.
            # Reduce A annotations into 1 - it'll also reduce the array dimensionality.
            # tfr_ev.data.shape (60, 3, 34, 282)
            avg = tfr_ev.average() # avg.data.shape = (3, 34, 282) - ndarray, shape (n_channels, n_freqs, n_times)

            if merge_channels:
                n_timestamps_index = 2
                n_timestamps = avg.data.shape[n_timestamps_index]
                channel_data = avg.data.reshape(-1, n_timestamps)
                image_file_name = f'{raw_file_name}-Ch-{tfr.ch_names}'
                self._save_image(image_folder=images_folder, image_file_name=image_file_name, image=channel_data)

            if generate_intermediate_images:
                for channel_index, channel_name in enumerate(tfr_ev.ch_names):
                    channel_data = avg.data[channel_index] # avg.data.shape = (34, 282) - ndarray, shape (n_freqs, n_times)
                    image_file_name = f'{raw_file_name}-Ch-{channel_name}'
                    self._save_image(image_folder=images_folder, image_file_name=image_file_name, image=channel_data)
        
            # fig.savefig(f'Event-{key_event_id}.png')