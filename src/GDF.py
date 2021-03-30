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
        self.file_path = file_path
        self.valid_cue_descriptions = valid_cue_descriptions
        self.debug = debug
        self.cue_map = cue_map
        
    def _time_from_annotation(self, annotation):
        # Get data to slice the time series data
        start_time = annotation['onset']
        duration = annotation['duration']
        end_time = start_time + duration
        return start_time, end_time

    def _description_from_annotation(self, annotation):
        return int(annotation['description'])
    
    def _check(self, annotation):
        # Ignore annotations that are not cues
        description_int = self._description_from_annotation(annotation)
        is_valid = description_int in self.valid_cue_descriptions
        return is_valid

    def _generate_image_name(annotation_index: int, channel_index: int = None):
        raw_file_name = self.file_path.split('/')[-1]
        file_name_components = [raw_file_name, 'Ann', annotation_index]
        if not channel_index is None:
            file_name_components.extend(['Ch', channel_index])

        file_name_components.join('-')
        image_file_name = f'{image_file_name}.png'
        
        return image_file_name

    # TODO: Is there a way to declare `gaf` type to `pyts.image.GramianAngularField`?
    def _generate_image(self, gaf, output_folder: str, cue_human_readable: str, cue_samples: pd.DataFrame, n_timestamps: int, image_file_name: str, generate_intermediate_images: bool):
        # Generate Garmian Angular Field
        # Summation
        image_folder = f'{output_folder}/{gaf.method}/{cue_human_readable}'
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        cue_samples_gaf = gaf.fit_transform(cue_samples)
        
        # Reshape to reduce 1 dimension, making it a taller matrix aka stacking vertically all images
        # https://github.com/johannfaouzi/pyts/issues/95#issuecomment-809177142
        stacked_image = cue_samples_gaf.reshape(-1, n_timestamps) 
        
        image_path = f'{image_folder}/{image_file_name}'
        plt.imsave(image_path, stacked_image)

        if generate_intermediate_images:
            channel_index = 0
            for img in cue_samples_gaf:
                plt.imsave(f'{image_path}/Ch-{channel_index}-{image_file_name}', img)
                channel_index += 1

    def _generate_image_from_annotation(self, annotation, gasf, gadf, output_folder: str, cue_human_readable: str, cue_samples: pd.DataFrame, n_timestamps: int, image_file_name: str, generate_intermediate_images: bool, generate_difference_images: bool):
        log(f'Generating summation image ({image_file_name})...')
        self._generate_image(gaf=gasf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                                cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name, 
                                generate_intermediate_images=generate_intermediate_images)

        if generate_difference_images:
            log(f'Generating difference image ({image_file_name})...')
            self._generate_image(gaf=gadf, output_folder=output_folder, cue_human_readable=cue_human_readable, 
                                    cue_samples=cue_samples, n_timestamps=n_timestamps, image_file_name=image_file_name, 
                                    generate_intermediate_images=generate_intermediate_images)

    def generate_images(self, output_folder: str, generate_difference_images: bool = False, generate_intermediate_images: bool = False):
        gasf = GramianAngularField(image_size=1, method='summation')
        gadf = GramianAngularField(image_size=1, method='difference')
        
        raw_file_name = self.file_path.split('/')[-1]

        raw = mne.io.read_raw_gdf(self.file_path)
        annotations = raw.annotations
        annotation_index = 0
        processes = []
        for ann in annotations:
            description_int = self._description_from_annotation(ann)
            cue_human_readable = self.cue_map[description_int]
            
            if not self._check(ann):
                log(f'Ignoring annotation ({annotation_index}): {cue_human_readable}...')
                continue
            
            log(f'Processing annotation ({annotation_index}): {cue_human_readable}...')
            start_time, end_time = self._time_from_annotation(ann)
            cue_samples = raw.copy().crop(tmin=start_time, tmax=end_time).to_data_frame().drop(columns='time')
            
            n_timestamps, n_samples = cue_samples.shape # (314, 6)
            
            # Setup GAFs
            gasf.image_size = gadf.image_size = n_timestamps # the image size can only be as big as there are samples
            
            # Prepare data as required by GAF lib
            cue_samples = cue_samples.transpose() # GramianAngularField.fit_transform() needs (n_samples, n_features): cue_samples.shape = (6, 314)
            cue_samples = cue_samples * 1000 # convert to volts?
            
            # Mount image path
            image_file_name = f'{raw_file_name}-Ann-{annotation_index}.png'

            self._generate_image_from_annotation(ann, gasf, gadf, output_folder, cue_human_readable, cue_samples, n_timestamps, image_file_name, generate_intermediate_images, generate_difference_images)
            
            # # TODO: Test multiprocess only here and make the files for run in main process
            # p = Process(target=self._generate_image_from_annotation, args=(ann, gasf, gadf, output_folder, cue_human_readable, cue_samples, n_timestamps, image_file_name, generate_intermediate_images, generate_difference_images))
            # processes.append(p)
            # p.start()
            
            annotation_index += 1