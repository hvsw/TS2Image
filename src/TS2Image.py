from multiprocessing import Process
from GAF import GAF
from Logger import log
from ERSP import ERSP
import os

# Python 3.8.2

__all__ = ["TS2Image"]

# Use public method `generate_images` to process time series files into images
class TS2Image:
    def __init__(self, input_folder: str, output_folder: str):
        self.input_folder = input_folder
        self.output_folder = output_folder

    ## HELPERS
    # File managment helpers
    def __list_filtered_files(self, input_folder:str):
        from os import listdir
        from os.path import isfile, join
        files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
        gdf_files = sorted(filter(self.__custom_filter, files), reverse=False)
        return gdf_files

    # Set filter files
    def __custom_filter(self, file:str):
        # Note: Change your filter here
        should_include_file = self.__bci_competition_b(file)
        log(f"should_include_file: {file}: {should_include_file}")
        return should_include_file
    
    # http://www.bbci.de/competition/iv/desc_2b.pdf
    # 3 training, 2 evaluation
    def __bci_competition_b(self, file):
        is_gdf_file = file.endswith('.gdf')
        
        # we only want training
        is_training_file = file.split('.')[0].endswith('T')

        # Smiley feedback sessions have different timing scheme
        is_smiley_feedback_session = file.endswith("03T.gdf") or file.endswith("04E.gdf") or file.endswith("05E.gdf")

        return is_gdf_file and is_training_file and file.startswith("B") and not is_smiley_feedback_session
    # This function defines the channels you want to use for each file
    def __desired_channels_for_file(self, file_name: str):
        channels = ["C3", "C4", "Cz"]
        # BCI dataset
        if file_name.startswith("A"):
            return ['EEG-C3', 'EEG-C4', 'EEG-Cz']
        else:
            return ['EEG:C3', 'EEG:C4', 'EEG:Cz']

    # Helper function to generate the images. Serve kind as a facade.
    def __generate_images(self, files_dir: str, file_name: str, output_folder: str, method: str, events_descriptions_to_process: list, t_start, duration, events_dictionary: dict):
        # Set desired channels
        desired_channels = self.__desired_channels_for_file(file_name)

        file_full_path = f'{files_dir}/{file_name}'
        log(f'Started {file_full_path}...')
        log(f"Working on {method.upper()}")
        if method.upper() == "GAF":
            gaf = GAF(file_path=file_full_path, valid_events_descriptions=events_descriptions_to_process, cue_map=events_dictionary)
            gaf.generate_images(output_folder=output_folder, t_start=t_start, duration=duration, generate_intermediate_images=True, generate_difference_images=False, desired_channels=desired_channels, merge_channels=False)
        else:
            ersp = ERSP(file_path=file_full_path)
            ersp.generate_images(output_folder=output_folder, desired_events=events_descriptions_to_process, t_start=t_start, t_end=duration, generate_intermediate_images=True, desired_channels=desired_channels, merge_channels=False)

        log(f'Finished {file_full_path}!')

    # Set the method you want to use. Accepted values: GAF, ERSP
    # events_dictionary: a dictionary specifying the ALL the data set events' identifier and description. This is optional only needed by GAF class to get the description to create folders to save the generate images. This is not used by ERSP class.
    def generate_images(self, method: str, valid_events_descriptions: list, events_dictionary: dict, t_start, duration):
        # if not method == "GAF":
        #     raise Exception(f'Method {method} not supported yet')
        
        base_dir = os.getcwd()
        # Set the directory containing the files you want to process
        input_folder = base_dir + "/datasets"

        # Set the root output folder
        output_folder = base_dir + '/output'

        files = self.__list_filtered_files(self.input_folder)

        log('!!! START !!!')
        log(f"Method: {method}")
        log(f'Processing files inside: {self.input_folder}')
        log(f'Files: {files}')

        for file in files:
            self.__generate_images(self.input_folder, file, self.output_folder, method=method, t_start=t_start, duration=duration, events_descriptions_to_process=valid_events_descriptions, events_dictionary=events_dictionary)

        log(f'Generated images available at: {self.output_folder}')
        log('!!! FINISH !!!')
