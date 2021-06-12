from pyts.image import GramianAngularField
import os
import numpy as np
import mne
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import Logger

__all__ = ["GDF"]

def log(msg:str, log_file: str = None):
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    txt = f'{now}: {msg}'
    if log_file is None:
        print(txt)
    else:
        f = open(log_file, 'w')
        f.write(txt)
        f.close

class GDF:

    def __init__(self, file_path: str, valid_cue_descriptions: list, cue_map, debug: bool = True):
        # File path to 
        self.file_path = file_path

        # Event type we want to generate images for
        self.valid_cue_descriptions = valid_cue_descriptions

        # Variable to print or not some information along execution
        self.debug = debug

        # This is a dictionary [number:text], we use the values to get the class name and use as the output folder
        self.cue_map = cue_map
        
    def _time_from_annotation(self, annotation):
        # Get data to slice the time series data
        start_time = annotation['onset']
        duration = annotation['duration']
        end_time = start_time + duration
        return start_time, end_time

    # Get event description. It's the number, not the string.
    def _description_from_annotation(self, annotation):
        return int(annotation['description'])
    
    def _check(self, annotation):
        # Ignore annotations that are not cues
        description_int = self._description_from_annotation(annotation)
        is_valid = description_int in self.valid_cue_descriptions
        return is_valid

    def _generate_image_name(self, annotation_index: int, channel_index: int = None):
        raw_file_name = self.file_path.split('/')[-1]
        file_name_components = [raw_file_name, 'Ann', annotation_index]
        if not channel_index is None:
            file_name_components.extend(['Ch', channel_index])

        file_name_components.join('-')
        image_file_name = f'{image_file_name}.png'
        
        return image_file_name

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
                    log("Saving image to " + image_path)
                    plt.imsave(image_path, image, cmap=cmap)

    # TODO: Is there a way to declare `gaf` type to `pyts.image.GramianAngularField`?
    def _generate_image(self, gaf, output_folder: str, cue_human_readable: str, 
                        cue_samples: pd.DataFrame, n_timestamps: int, image_file_name: str, 
                        generate_intermediate_images: bool = False):
        # Generate Garmian Angular Field
        # Summation
        image_folder = f'{output_folder}/GAF/{gaf.method}/{cue_human_readable}'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        cue_samples_gaf = gaf.fit_transform(cue_samples)
        
        # Reshape to reduce 1 dimension, making it a taller matrix aka stacking vertically all images
        # https://github.com/johannfaouzi/pyts/issues/95#issuecomment-809177142
        stacked_image = cue_samples_gaf.reshape(-1, n_timestamps)
        
        self._save_image(image_folder, image_file_name, stacked_image)
        
        # TODO: generate_intermediate_images is coming as False but we're passing True in _generate_image_from_annotation
        if generate_intermediate_images:
            channel_index = 0
            for img in cue_samples_gaf:
                intermediate_image_file_name = f"{image_file_name}-Ch-{channel_index}"
                self._save_image(image_folder, intermediate_image_file_name, img)
                channel_index += 1

    def _generate_image_from_annotation(self, annotation, gasf, gadf, output_folder: str, 
                                        cue_human_readable: str, cue_samples: pd.DataFrame, 
                                        n_timestamps: int, image_file_name: str, 
                                        generate_intermediate_images: bool = False, generate_difference_images: bool = False):
        log(f'Generating summation image ({image_file_name})...')
        self._generate_image(gaf=gasf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                             cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name,
                             generate_intermediate_images=generate_intermediate_images)

        if generate_difference_images:
            log(f'Generating difference image ({image_file_name})...')
            self._generate_image(gaf=gadf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                                 cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name,
                                 generate_intermediate_images=generate_intermediate_images)

    def generate_images(self, output_folder: str, generate_intermediate_images: bool = False, generate_difference_images: bool = False, desired_channels: list = []):
        gasf = GramianAngularField(image_size=1, method='summation')
        gadf = GramianAngularField(image_size=1, method='difference')
        
        raw_file_name = self.file_path.split('/')[-1]

        # Read data
        raw = mne.io.read_raw_gdf(self.file_path)
        
        # Filter channels
        if len(desired_channels) > 0:
            raw = raw.pick_channels(desired_channels)

        # Get annotations and iterate over
        annotations = raw.annotations
        annotation_index = 0
        for ann in annotations:
            # This is used only for the output folder as the event class
            description_int = self._description_from_annotation(ann)
            cue_human_readable = self.cue_map[description_int]
            
            # Internal validation if the annotation should be processed
            if not self._check(ann):
                log(f'Ignoring annotation ({annotation_index}): {cue_human_readable}...')
                continue

            log(f'Processing annotation ({annotation_index}): {cue_human_readable}...')

            # Extract samples for that event/cue
            start_time, end_time = self._time_from_annotation(ann)
            cue_samples = raw.copy().crop(tmin=start_time, tmax=end_time).to_data_frame().drop(columns='time')
            
            n_timestamps, n_samples = cue_samples.shape # (314, 6)
            
            # Setup GAFs
            gasf.image_size = gadf.image_size = n_timestamps # the image size can only be as big as there are samples
            
            # Prepare data as required by GAF lib
            cue_samples = cue_samples.transpose() # GramianAngularField.fit_transform() expects (n_samples, n_features): cue_samples.shape = (6, 314)
            cue_samples = cue_samples * 1000 # convert to volts?
            
            # Mount image path
            image_file_name = f'{raw_file_name}-Ann-{annotation_index}'

            self._generate_image_from_annotation(annotation=ann, gasf=gasf, gadf=gadf, 
                                                 output_folder=output_folder, 
                                                 cue_human_readable=cue_human_readable, cue_samples=cue_samples, n_timestamps=n_timestamps, 
                                                 image_file_name=image_file_name, 
                                                 generate_intermediate_images=generate_intermediate_images, 
                                                 generate_difference_images=generate_difference_images)
            
            # # TODO: Test multiprocess only here and make the files to run in main process
            # p = Process(target=self._generate_image_from_annotation, args=(ann, gasf, gadf, output_folder, cue_human_readable, cue_samples, n_timestamps, image_file_name))
            # processes.append(p)
            # p.start()
            
            annotation_index += 1