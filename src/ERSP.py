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

# TODO: REFACTOR! This was heavily inspired in https://mne.tools/stable/auto_examples/time_frequency/plot_time_frequency_erds.html#id6

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

        default = 'viridis'
        cmaps_selected = [default]
        for cmap_type, cmaps in all_cmaps.items():
            for cmap in cmaps:
                if cmap in cmaps_selected:
                    final_image_folder = f'{image_folder}/{cmap_type}/{cmap}'
                    if not os.path.exists(final_image_folder):
                        os.makedirs(final_image_folder)
        
                    image_path = f'{final_image_folder}/{image_file_name}.png'
                    plt.imsave(image_path, image, cmap=cmap)

    def generate_images(self, output_folder: str, generate_intermediate_images: bool = False, generate_difference_images: bool = False):
        raw = mne.io.read_raw_gdf(self.file_path)
        # TODO: Filter by event_id = {"769":769}
        # This function will assign an integer Event ID to each unique element of raw.annotations.description, 
        # and will return the mapping of descriptions to integer Event IDs along with the derived Event array.
        event_ids = {"769":1, "770":2}  # map event IDs to tasks
        # event_ids = LABELS_DICTIONARY
        # event_ids = None
        
        events, event_id = mne.events_from_annotations(raw, event_ids)
        # events.shape = (271, 3) (events, 3) = the array in the second dimension is [start, ?, new int event_id converted by the lib]

        picks = mne.pick_channels(raw.info["ch_names"], ['EEG:C3', 'EEG:Cz', 'EEG:C4'])

        # epoch data ##################################################################
        tmin, tmax = 0, 1.25  # define epochs around events (in s)
        # tmin, tmax = 0, 0  # define epochs around events (in s)
        
        # WARNING: This prevents exceptions to be thrown if we have overlapping events and it can also drop data!!!
        # Check epocs.drop_log or something
        # Error thrown: 'Event time samples were not unique. Consider setting the `event_repeated` parameter."'
        event_repeated = None
        epochs = mne.Epochs(raw, events, event_ids, tmin - 0.5, tmax + 0.5,
                            picks=picks, baseline=None, preload=True, event_repeated=event_repeated)

        # compute ERDS maps ###########################################################
        # TODO: What this actually means? Are we filtering the signal? Is it related to the ERSP frequency resolution?
        # frequencies from 2-35Hz
        freqs = np.arange(2, 36, 1)  
        n_cycles = freqs  # use constant t/f resolution
        vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
        baseline = [-1, 0]  # baseline interval (in s)
        # cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
        # kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
        #               buffer_size=None, out_type='mask')  # for cluster test

        # Run TF decomposition overall epochs
        tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                            use_fft=True, return_itc=False, average=False,
                            decim=2)
        tfr.crop(tmin, tmax)
        tfr.apply_baseline(baseline, mode="percent") # tfr.event_id = {'10': 10, '11': 11, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

        raw_file_name = self.file_path.split('/')[-1]
        epoch_index = 0
        # TODO: So at the end of the day we're justing averaging all runs for an specific event type? See tfr_ev.average()
        for key_event_id, v in event_ids.items():
            # select desired epochs for visualization
            tfr_ev = tfr[key_event_id]

            # https://mne.tools/stable/generated/mne.time_frequency.EpochsTFR.html#mne.time_frequency.EpochsTFR.average
            # Reduce 60 annotations into 1. It'll reduce the array dimensionality
            avg = tfr_ev.average() # avg.data.shape = (3, 34, 627) = (channels?, frequencies, samples?)
            
            timestamps_dimension_index = 2
            n_timestamps = avg.data.shape[timestamps_dimension_index]
            stacked_image = avg.data.reshape(-1, n_timestamps)
            
            # plot TFR (ERDS map with masking)
            # avg.plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
            #         axes=ax, colorbar=False, show=False, mask=mask,
            #         mask_style="mask")
            cue_human_readable = key_event_id
            image_file_name = f'{raw_file_name}-Ep-{epoch_index}'
            image_folder = f'{output_folder}/ERSP/{cue_human_readable}'
            if not os.path.exists(image_folder):
                os.makedirs(image_folder)
            self._save_image(image_folder=image_folder, image_file_name=image_file_name, image=stacked_image)
            epoch_index += 1