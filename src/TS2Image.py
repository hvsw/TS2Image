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
        files_filter_out = ['A09T.gdf']
        # 2021-06-17 17:56:58.753: Started /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf...
        # Extracting EDF parameters from /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf...
        # GDF file detected
        # /Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/edf/edf.py:1044: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
        # etmode = np.fromstring(etmode, UINT8).tolist()[0]
        # Setting channel info structure...
        # Creating raw.info structure...
        # /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py:132: RuntimeWarning: Channel names are not unique, found duplicates for: {'EEG'}. Applying running numbers for duplicates.
        # raw = mne.io.read_raw_gdf(self.file_path)
        # Traceback (most recent call last):
        # File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 94, in <module>
        #     generate_images(files_dir, gdf_file, output_folder, "GAF")
        # File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 36, in generate_images
        #     gdf.generate_images(output_folder=output_folder, generate_intermediate_images=True, generate_difference_images=False, desired_channels=desired_channels, merge_channels=False)
        # File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py", line 136, in generate_images
        #     raw = raw.pick_channels(desired_channels)
        # File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/channels/channels.py", line 795, in pick_channels
        #     return self._pick_drop_channels(picks)
        # File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/channels/channels.py", line 919, in _pick_drop_channels
        #     pick_info(self.info, idx, copy=False)
        # File "<decorator-gen-8>", line 24, in pick_info
        # File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/pick.py", line 533, in pick_info
        #     raise ValueError('No channels match the selection.')
        # ValueError: No channels match the selection.

        files = ['B0101T.gdf']
        # should_include = (len(files) > 0 and file in files) or len(files) == 0

        is_gdf_file = file.endswith('.gdf')
        is_training_file = file.split('.')[0].endswith('T')

        return is_gdf_file and is_training_file # and file.startswith("B") and (file not in files_filter_out) and file in files

    # This function defines the channels you want to use for each file
    def __desired_channels_for_file(self, file_name: str):
        channels = ["C3", "C4", "Cz"]
        if file_name.startswith("A"):
            return ['EEG-C3', 'EEG-C4', 'EEG-Cz']
        else:
            return ['EEG:C3', 'EEG:C4', 'EEG:Cz']

    # Helper function to generate the images. Serve kind as a facade.
    def __generate_images(self, files_dir: str, gdf_file: str, output_folder: str, method: str, events_descriptions_to_process: list, events_dictionary: dict):
        # Set desired channels
        desired_channels = self.__desired_channels_for_file(gdf_file)

        gdf_file_full_path = f'{files_dir}/{gdf_file}'
        log(f'Started {gdf_file_full_path}...')
        if method == "GAF":
            gaf = GAF(file_path=gdf_file_full_path, valid_events_descriptions=events_descriptions_to_process, cue_map=events_dictionary)
            gaf.generate_images(output_folder=output_folder, generate_intermediate_images=True, generate_difference_images=False, desired_channels=desired_channels, merge_channels=False)
        else:
            ersp = ERSP(file_path=gdf_file_full_path)
            ersp.generate_images(output_folder=output_folder, desired_events=events_descriptions_to_process, generate_intermediate_images=True, desired_channels=desired_channels, merge_channels=True)

        log(f'Finished {gdf_file_full_path}!')

    # Set the method you want to use. Accepted values: GAF, ERSP
    def generate_images(self, method: str, valid_events_descriptions: list, events_dictionary: dict):
        base_dir = os.getcwd()
        # Set the directory containing the files you want to process
        input_folder = base_dir + "/datasets"

        # Set the root output folder
        output_folder = base_dir + '/output'

        gdf_files = self.__list_filtered_files(input_folder)

        log('!!! START !!!')
        log(f'Processing files in {input_folder}')
        log(f'Images will be here {output_folder}')
        log(f'Files: {gdf_files}')

        for gdf_file in gdf_files:
            self.__generate_images(input_folder, gdf_file, output_folder, method=method, events_descriptions_to_process=valid_events_descriptions, events_dictionary=events_dictionary)

        log('!!! FINISH !!!')
