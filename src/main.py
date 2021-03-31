from multiprocessing import Process
from GDF import GDF
from Logger import log
import os

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

def get_all_files_from_dir(mypath):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

def is_gdf_file(file:str):
    return file.endswith('.gdf')

def is_training_file(file:str):
    return file.split('.')[0].endswith('T')

def my_filter(file:str):
    files = ['B0101T.gdf']
    return is_gdf_file(file) and is_training_file(file) and file in files

number_processed_files = 0
def generate_images(files_dir: str, gdf_file: str, output_folder: str):
    gdf_file_full_path = f'{files_dir}/{gdf_file}'
    log(f'Started {gdf_file_full_path}...')
    gdf = GDF(file_path=gdf_file_full_path, valid_cue_descriptions=[DESCRIPTION_CUE_LEFT, DESCRIPTION_CUE_RIGHT], cue_map=LABELS_DICTIONARY)
    gdf.generate_images(output_folder=output_folder)
    # number_processed_files += 1
    log(f'Finished {gdf_file_full_path}!')


base_dir = os.getcwd()
files_dir = base_dir + "/datasets"
output_folder = base_dir + '/output'

files = get_all_files_from_dir(files_dir)
gdf_files = sorted(filter(my_filter, files), reverse=True)

max_number_files = None
proccesses = []

log('!!! START !!!')
log(f'Processing files in {files_dir}')
log(f'Max number of files: {max_number_files}')
for gdf_file in gdf_files:
    p = Process(target=generate_images, args=(files_dir, gdf_file, output_folder))
    proccesses.append(p)
    p.start()
    
for p in proccesses:
    if p.is_alive:
        p.join()

log('!!! FINISH !!!')
log(f'Processed files: {number_processed_files}')