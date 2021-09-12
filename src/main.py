from multiprocessing import Process
from GAF import GAF
from Logger import log
from ERSP import ERSP
import os

# B0101T.gdf
# f = 250Hz
# duration = 1.25s
# n_samples = 1.25 * 150 = 313

## HELPERS
# File managment helpers
def list_filtered_files(input_folder:str):
    from os import listdir
    from os.path import isfile, join
    files = [f for f in listdir(input_folder) if isfile(join(input_folder, f))]
    gdf_files = sorted(filter(custom_filter, files), reverse=False)
    return gdf_files

# Set filter files
def custom_filter(file:str):
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
def desired_channels_for_file(file_name: str):
    channels = ["C3", "C4", "Cz"]
    if file_name.startswith("A"):
        return ['EEG-C3', 'EEG-C4', 'EEG-Cz']
    else:
        return ['EEG:C3', 'EEG:C4', 'EEG:Cz']

# Set LABELS_DICTIONARY according to your dataset. These are the events in BCI IV competition dataset.
DESCRIPTION_EYES_OPEN = 276
DESCRIPTION_EYES_CLOSED = 277
DESCRIPTION_START_TRIAL = 768
DESCRIPTION_CUE_LEFT = 769
DESCRIPTION_CUE_RIGHT = 770
DESCRIPTION_BCI_FEEDBACK = 781
DESCRIPTION_CUE_UNKNOWN = 783
DESCRIPTION_REJECTED_TRIAL = 1023
DESCRIPTION_EYE_MOVEMENT_HORIZONTAL = 1077
DESCRIPTION_EYE_MOVEMENT_VERTICAL = 1078
DESCRIPTION_EYE_ROTATION = 1079
DESCRIPTION_EYE_BLINK = 1081
DESCRIPTION_START_NEW_RUN = 32766
LABELS_DICTIONARY = {
    DESCRIPTION_EYES_OPEN:'Idling EEG (eyes open)',
    DESCRIPTION_EYES_CLOSED:'Idling EEG (eyes closed)',
    DESCRIPTION_START_TRIAL:'Start of a trial',
    DESCRIPTION_CUE_LEFT:'Cue onset left (class 1)',
    DESCRIPTION_CUE_RIGHT:'Cue onset right (class 2)',
    DESCRIPTION_BCI_FEEDBACK:'BCI feedback (continuous)',
    DESCRIPTION_CUE_UNKNOWN:'Cue unknown',
    DESCRIPTION_REJECTED_TRIAL:'Rejected trial',
    DESCRIPTION_EYE_MOVEMENT_HORIZONTAL:'Horizontal eye movement',
    DESCRIPTION_EYE_MOVEMENT_VERTICAL:'Vertical eye movement',
    DESCRIPTION_EYE_ROTATION:'Eye rotation',
    DESCRIPTION_EYE_BLINK:'Eye blinks',
    DESCRIPTION_START_NEW_RUN:'Start of a new run'
}

# Helper function to generate the images. Serve kind as a facade.
def generate_images(files_dir: str, gdf_file: str, output_folder: str, method: str = "GAF"):
    # Set the events you want to export
    valid_annotation_descriptions = [DESCRIPTION_CUE_LEFT, DESCRIPTION_CUE_RIGHT]

    # Set desired channels
    desired_channels = desired_channels_for_file(gdf_file)

    gdf_file_full_path = f'{files_dir}/{gdf_file}'
    log(f'Started {gdf_file_full_path}...')
    if method == "GAF":
        gaf = GAF(file_path=gdf_file_full_path, valid_cue_descriptions=valid_annotation_descriptions, cue_map=LABELS_DICTIONARY)
        gaf.generate_images(output_folder=output_folder, generate_intermediate_images=True, generate_difference_images=False, desired_channels=desired_channels, merge_channels=False)
    else:
        ersp = ERSP(file_path=gdf_file_full_path, valid_cue_descriptions=valid_annotation_descriptions, cue_map=LABELS_DICTIONARY)
        ersp.generate_images(output_folder=output_folder, generate_intermediate_images=True, desired_channels=desired_channels, merge_channels=True)
    # number_processed_files += 1
    log(f'Finished {gdf_file_full_path}!')

base_dir = os.getcwd()
# Set the directory containing the files you want to process
input_folder = base_dir + "/datasets"

# Set the root output folder
output_folder = base_dir + '/output'

## -------------------------- DON'T TOUCH ---------------------------

gdf_files = list_filtered_files(input_folder)

log('!!! START !!!')
log(f'Processing files in {input_folder}')
log(f'Files: {gdf_files}')

# Set the method you want to use. Accepted values: GAF, ERSP
method = "ERSP"
for gdf_file in gdf_files:
    generate_images(input_folder, gdf_file, output_folder, method=method)

log('!!! FINISH !!!')

# TODO: SOME WEIRD OUTPUTS when processing BCI Competition IV 2a and 2b datasets:
# TODO: How to fix channels with same name, ignore file?
# Extracting EDF parameters from /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf...
# GDF file detected
# /Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/edf/edf.py:1044: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
#   etmode = np.fromstring(etmode, UINT8).tolist()[0]
# Setting channel info structure...
# Creating raw.info structure...
# /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py:132: RuntimeWarning: Channel names are not unique, found duplicates for: {'EEG'}. Applying running numbers for duplicates.
#   raw = mne.io.read_raw_gdf(self.file_path)
# Traceback (most recent call last):
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 94, in <module>
#     generate_images(files_dir, gdf_file, output_folder, "GAF")
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 36, in generate_images
#     gdf.generate_images(output_folder=output_folder, generate_intermediate_images=True, generate_difference_images=False, desired_channels=desired_channels, merge_channels=False)
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py", line 136, in generate_images
#     raw = raw.pick_channels(desired_channels)
#   File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/channels/channels.py", line 795, in pick_channels
#     return self._pick_drop_channels(picks)
#   File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/channels/channels.py", line 919, in _pick_drop_channels
#     pick_info(self.info, idx, copy=False)
#   File "<decorator-gen-8>", line 24, in pick_info
#   File "/Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/pick.py", line 533, in pick_info
#     raise ValueError('No channels match the selection.')
# ValueError: No channels match the selection.

# TODO: Is this a problem?
# 2021-06-12 19:26:58.272: Started /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/B0101T.gdf...
# Extracting EDF parameters from /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/B0101T.gdf...
# GDF file detected
# /Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/edf/edf.py:1044: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
#   etmode = np.fromstring(etmode, UINT8).tolist()[0]
# Setting channel info structure...
# /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py:132: RuntimeWarning: Highpass cutoff frequency 100.0 is greater than lowpass cutoff frequency 0.5, setting values to 0 and Nyquist.
#   raw = mne.io.read_raw_gdf(self.file_path)
# Creating raw.info structure...

# TODO: How we got this 1072 key from?
# 2021-06-12 19:27:55.370: Started /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf...
# Extracting EDF parameters from /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/datasets/A09T.gdf...
# GDF file detected
# /Users/henrique/Library/Python/3.8/lib/python/site-packages/mne/io/edf/edf.py:1044: DeprecationWarning: The binary mode of fromstring is deprecated, as it behaves surprisingly on unicode inputs. Use frombuffer instead
#   etmode = np.fromstring(etmode, UINT8).tolist()[0]
# Setting channel info structure...
# Creating raw.info structure...
# /Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py:132: RuntimeWarning: Channel names are not unique, found duplicates for: {'EEG'}. Applying running numbers for duplicates.
#   raw = mne.io.read_raw_gdf(self.file_path)
# 2021-06-12 19:27:55.386: Ignoring annotation (0): Start of a new run...
# 2021-06-12 19:27:55.386: Ignoring annotation (0): Idling EEG (eyes open)...
# 2021-06-12 19:27:55.386: Ignoring annotation (0): Start of a new run...
# 2021-06-12 19:27:55.386: Ignoring annotation (0): Idling EEG (eyes closed)...
# 2021-06-12 19:27:55.386: Ignoring annotation (0): Start of a new run...
# Traceback (most recent call last):
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 90, in <module>
#     generate_images(files_dir, gdf_file, output_folder, "GAF")
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/main.py", line 33, in generate_images
#     gdf.generate_images(output_folder=output_folder, generate_intermediate_images=False, generate_difference_images=False)
#   File "/Users/henrique/Documents/UFRGS/TCC Local/TS2Image/src/GDF.py", line 144, in generate_images
#     cue_human_readable = self.cue_map[description_int]
# KeyError: 1072