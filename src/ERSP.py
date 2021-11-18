import mne

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
# from mne.viz.utils import center_cmap
# from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

# import Logger

# Python 3.8.2

__all__ = ["ERSP"]

class ERSP:
    def __init__(self, file_path: str, debug: bool = True):
        self.file_path = file_path
        self.debug = debug
    
    def _save_image(self, image_folder, image_file_name, image):
        all_cmaps = {
            'Perceptually Uniform Sequential' : ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Sequential' : ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 
                            'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        }

        # TODO: Test other color maps
        default = 'viridis'
        cmaps_selected = [default]

        # Tests
        custom_cmap = center_cmap(plt.cm.RdBu, -1, 1)  # zero maps to white
        final_image_folder = f'{image_folder}'
        if not os.path.exists(final_image_folder):
            os.makedirs(final_image_folder)

        image_path = f'{final_image_folder}/{image_file_name}.png'
        plt.imsave(image_path, image, cmap=custom_cmap)
        return

        # cmaps_selected = [cmap]

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
                    plt.imsave(image_path, image, cmap=custom_cmap)

    # Generate sequential ids for events_descriptions_list
    # { x: 1, y: 2, z: 3, ... }
    def _generate_events_dictionary(self, events_descriptions_list: list):
        event_ids = {}
        if events_descriptions_list is None:
            return event_ids
        
        index = 1
        for desired_event in events_descriptions_list:
            event_ids[desired_event] = index
            index += 1

        return event_ids

    # Start and end time of the epochs in seconds, relative to the time-locked event. Defaults to -0.2 and 0.5, respectively.
    # t_start and t_end refer to the epoch time window around the event. 
    # Ex.: t_start = -0.5, t_end = 1.5, t_event = 10, would create an epoch starting at 10-0.5 and end at 11.5
    def generate_images(self, output_folder: str, desired_channels: list, desired_events: list, t_start, t_end, generate_intermediate_images: bool = False, merge_channels=False):
        raw = mne.io.read_raw_gdf(self.file_path, preload=True)
        raw.filter(l_freq=1, h_freq=40)

        # https://mne.tools/stable/generated/mne.events_from_annotations.html?highlight=events_from_annotations#mne.events_from_annotations
        # Map descriptions (keys) to integer event codes (values). Only the descriptions present will be mapped, others will be ignored.
        event_id = self._generate_events_dictionary(events_descriptions_list=desired_events)

        # https://mne.tools/stable/generated/mne.events_from_annotations.html?highlight=events_from_annotations#mne.events_from_annotations
        # Get events and event_id from an Annotations object.
        # This function will assign an integer Event ID to each unique element of raw.annotations.description, 
        # and will return the mapping of descriptions to integer Event IDs along with the derived Event list.
        events, generated_event_ids = mne.events_from_annotations(raw, event_id=event_id)
        # events.shape = (271, 3) = (n_events, 3) = the events array is [start_sample, 0?, new int from event_id converted by the lib]
        
        channels = desired_channels
        picks = mne.pick_channels(raw.info["ch_names"], channels)
        
        ##################### EPOCH DATA #####################
        # WARNING: This prevents exceptions to be thrown if we have overlapping events and it can also drop data!!!
        # Check epocs.drop_log or something
        # Error thrown: 'Event time samples were not unique. Consider setting the `event_repeated` parameter."'
        event_repeated = None

        # https://mne.discourse.group/t/what-is-tmin-and-tmax-in-epochs/2920/1
        # “Epochs” are equal-duration chunks of the continuous raw signal. 
        # Epochs are created relative to a series of “events” (an event is a sample number plus an event ID integer encoding what kind of event it was)
        # TODO: Test if we can replace event_id by desired_events. It seems it already supports a list of ids, so no need to convert list to dict...
        epochs = mne.Epochs(raw, events, event_id, t_start, t_end,
                            picks=picks, baseline=None, preload=True, event_repeated=event_repeated)

        # Compute ERDS maps ###########################################################
        # Frequencies from 2-35Hz
        freqs = np.arange(1, 40, 1)  
        
        # TODO: What this n_cycles actually mean???
        # The number of cycles globally or for each frequency. The time-window length is thus T = n_cycles / freq.
        n_cycles = freqs # use constant t/f resolution

        # Return inter-trial coherence (ITC) as well as averaged (or single-trial) power.
        return_inter_trial_coherence = False

        # Time-Frequency Representation (TFR)
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                             use_fft=True, return_itc=return_inter_trial_coherence, average=False, decim=2)
        
        # TODO: Should we crop this? :think:
        # https://mne.tools/stable/generated/mne.time_frequency.EpochsTFR.html#mne.time_frequency.EpochsTFR.crop
        tfr.crop(t_start, t_end)
        
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

        for key_event_description, value_event_id_int in event_id.items():
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