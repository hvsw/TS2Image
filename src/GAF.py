# TODO: Update MNE from 0.17.0 to latest
import mne
from pyts.image import GramianAngularField
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
# import Logger
import cv2

# Python 3.8.2

__all__ = ["GAF"]

def log(msg:str, log_file: str = None):
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    txt = f'{now}: {msg}'
    if log_file is None:
        print(txt)
    else:
        f = open(log_file, 'w')
        f.write(txt)
        f.close

class GAF:
    def __init__(self, file_path: str, valid_events_descriptions: list, cue_map, debug: bool = True):
        # File path to 
        self.file_path = file_path

        # Event type we want to generate images for
        self.valid_events_descriptions = valid_events_descriptions

        # Variable to print or not some information along execution
        self.debug = debug

        # This is a dictionary [number:text], we use the values to get the class name and use as the output folder
        self.cue_map = cue_map
        
    def __time_from_annotation(self, annotation):
        # Get data to slice the time series data
        start_time = annotation['onset']
        duration = annotation['duration']
        end_time = start_time + duration
        return start_time, end_time

    # Get event description. It's the number, not the string.
    def __description_from_annotation(self, annotation):
        return int(annotation['description'])
    
    def __should_process_annotation(self, annotation):
        description_int = self.__description_from_annotation(annotation)
        is_valid = description_int in self.valid_events_descriptions
        return is_valid

    def __save_image(self, image_folder, image_file_name, image):
        #  Color maps
        all_cmaps = {
            'Perceptually Uniform Sequential' : ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
            'Sequential' : ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 
                            'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        }
        default = 'viridis'
        cmaps_selected = [default]
        
        # Resize image if needed
        desired_image_size = 256
        desired_image = image
        if image.shape[0] < desired_image_size:
            desired_image = cv2.resize(desired_image, dsize=(desired_image_size, desired_image_size), interpolation=cv2.INTER_AREA)
        
        for cmap_type, cmaps in all_cmaps.items():
            for cmap in cmaps:
                if cmap in cmaps_selected:
                    final_image_folder = f'{image_folder}/{cmap_type}/{cmap}'
                    if not os.path.exists(final_image_folder):
                        os.makedirs(final_image_folder)
        
                    image_path = f'{final_image_folder}/{image_file_name}.png'
                    log("Saving image to " + image_path)
                    # plt.imsave(image_path, desired_image)
                    plt.imsave(image_path, desired_image, cmap=cmap)
                    
                    # image_path = f'{image_path}.csv'
                    # log("Saving image to " + image_path)
                    # np.savetxt(image_path, image, delimiter=",")

    # TODO: Is there a way to declare `gaf` type to be `pyts.image.GramianAngularField`?
    def __generate_image(self, gaf, output_folder: str, cue_human_readable: str, 
                        cue_samples: pd.DataFrame, n_timestamps: int, image_file_name: str, 
                        generate_intermediate_images: bool = False, merge_channels: bool = True):
        # Generate Garmian Angular Field
        image_folder = f'{output_folder}/GAF/{gaf.method}/{cue_human_readable}'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        cue_samples_gaf = gaf.fit_transform(cue_samples)
        
        if merge_channels:
            # Reshape to reduce/remove channels dimension, making it a taller matrix i.e. stacking channels images vertically
            # https://github.com/johannfaouzi/pyts/issues/95#issuecomment-809177142
            merged_channels_image = cue_samples_gaf.reshape(-1, n_timestamps)
            self.__save_image(image_folder, image_file_name, merged_channels_image)
        
        # TODO: generate_intermediate_images is coming as False but we're passing True in __generate_image_from_annotation
        if generate_intermediate_images:
            channel_index = 0
            for img in cue_samples_gaf:
                intermediate_image_file_name = f"{image_file_name}-Ch-{channel_index}"
                self.__save_image(image_folder, intermediate_image_file_name, img)
                channel_index += 1

    def __generate_image_from_annotation(self, gasf, gadf, output_folder: str, 
                                        cue_human_readable: str, cue_samples: pd.DataFrame, 
                                        n_timestamps: int, image_file_name: str, 
                                        generate_intermediate_images: bool = False, generate_difference_images: bool = False, merge_channels: bool = True):
        log(f'Generating summation image ({image_file_name})...')
        self.__generate_image(gaf=gasf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                             cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name,
                             generate_intermediate_images=generate_intermediate_images, merge_channels=merge_channels)

        if generate_difference_images:
            log(f'Generating difference image ({image_file_name})...')
            self.__generate_image(gaf=gadf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                                 cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name,
                                 generate_intermediate_images=generate_intermediate_images, merge_channels=merge_channels)

    def generate_images(self, output_folder: str, generate_intermediate_images: bool = False, generate_difference_images: bool = False, desired_channels: list = [], merge_channels: bool=True):
        gasf = GramianAngularField(image_size=32, method='summation')
        gadf = GramianAngularField(image_size=32, method='difference')
        
        raw_file_name = self.file_path.split('/')[-1]

        # Read data
        raw = mne.io.read_raw_gdf(self.file_path, preload=True)
        
        # Filter channels
        if len(desired_channels) > 0:
            # TODO: Improve channels filter!
            # Use cases:
            # - File with multiple channels with same name "EEG": /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf
            # -  How can we address this?
            # - Files with different prefixes "EEG:", "EEG-", etc: /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf
            # -  This is a detail the user may need to address
            raw = raw.pick_channels(desired_channels)

        raw.filter(l_freq=1, h_freq=40)

        # Get annotations and iterate over
        annotations = raw.annotations
        annotation_index = 0
        for ann in annotations:
            # This is used only for the output folder as the event class
            description_int = self.__description_from_annotation(ann)

            # TODO: This can crash if the annotation is not in the `event_description_dictionary`
            cue_human_readable = self.cue_map[description_int]
            
            # Internal validation if the annotation should be processed
            if not self.__should_process_annotation(ann):
                log(f'Ignoring annotation ({annotation_index}): {cue_human_readable}. See __should_process_annotation to understand why.')
                continue

            log(f'Processing annotation ({annotation_index}): {cue_human_readable}...')

            # Extract samples for that event/cue
            start_time, end_time = self.__time_from_annotation(ann)
            cue_samples = raw.copy().crop(tmin=start_time, tmax=end_time).to_data_frame().drop(columns='time')
            
            n_timestamps, n_samples = cue_samples.shape # (314, 6)
            
            # Setup GAFs
            # The image size can only be as big as there are samples, and the minimum size required by the ML model is 32.
            # TODO: What if we get a signal that has less than 32 samples?
            gasf.image_size = gadf.image_size = min(32, n_timestamps)
            
            # Prepare data as required by GAF lib
            cue_samples = cue_samples.transpose() # GramianAngularField.fit_transform() expects (n_samples, n_features): cue_samples.shape = (6, 314)
            # TODO: remove this multiplication?
            cue_samples = cue_samples * 1000 # convert to volts?
            
            # Mount image path
            image_file_name = f'size_{gasf.image_size}-{raw_file_name}-Ann-{annotation_index}'

            self.__generate_image_from_annotation(gasf=gasf, gadf=gadf, 
                                                 output_folder=output_folder, 
                                                 cue_human_readable=cue_human_readable, cue_samples=cue_samples, n_timestamps=n_timestamps, 
                                                 image_file_name=image_file_name, 
                                                 generate_intermediate_images=generate_intermediate_images, 
                                                 generate_difference_images=generate_difference_images, merge_channels=merge_channels)
            
            # # TODO: Test multiprocess only here and make the files to run in main process
            # p = Process(target=self.__generate_image_from_annotation, args=(ann, gasf, gadf, output_folder, cue_human_readable, cue_samples, n_timestamps, image_file_name))
            # processes.append(p)
            # p.start()
            
            annotation_index += 1