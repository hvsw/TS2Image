import os
current_working_directory = os.getcwd()

import argparse

# TODO: Add other arguments to parser
parser = argparse.ArgumentParser(description='Parse TS2Image parameters.')
parser.add_argument('method', metavar='Method', type=str, nargs='?',
                    help='Image generation method: GAF or ERSP')
args = parser.parse_args()
method = args.method or "GAF"

# Python 3.8.2
##########################################################################
######################## Variables to change #############################
##########################################################################
# NOTE: Set the directory containing the files you want to process
input_folder = current_working_directory + "/datasets"
# input_folder = "/Users/henrique/Downloads/Supernumerary_BCI_EEG_VR/001_hands"

# NOTE: Set the root output folder
output_folder = current_working_directory + '/output'

# NOTE: Create according to your dataset. 
# These are the events from some files in BCI IV competition dataset.
DESCRIPTION_EYES_OPEN = "276"
DESCRIPTION_EYES_CLOSED = "277"
DESCRIPTION_START_TRIAL = "768"
DESCRIPTION_CUE_LEFT = "769"
DESCRIPTION_CUE_RIGHT = "770"
DESCRIPTION_SOMETHING = "771" # TODO: What is this description? It's from A01T.gdf
DESCRIPTION_SOMETHING2 = "772" # TODO: What is this description? It's from A01T.gdf
# DESCRIPTION_SOMETHING = 773 # TODO: What is this description? It's from A01T.gdf
# DESCRIPTION_SOMETHING = 774 # TODO: What is this description? It's from A01T.gdf
DESCRIPTION_BCI_FEEDBACK = "781"
DESCRIPTION_CUE_UNKNOWN = "783"
DESCRIPTION_REJECTED_TRIAL = "1023"
DESCRIPTION_UNKNOWN_GROUP = "1072"
DESCRIPTION_EYE_MOVEMENT_HORIZONTAL = "1077"
DESCRIPTION_EYE_MOVEMENT_VERTICAL = "1078"
DESCRIPTION_EYE_ROTATION = "1079"
DESCRIPTION_EYE_BLINK = "1081"
DESCRIPTION_START_NEW_RUN = "32766"
BCI_competition_dataset_events_dictionary = {
    DESCRIPTION_EYES_OPEN:'Idling EEG (eyes open)',
    DESCRIPTION_EYES_CLOSED:'Idling EEG (eyes closed)',
    DESCRIPTION_START_TRIAL:'Start of a trial',
    DESCRIPTION_CUE_LEFT:'Cue onset left (class 1)',
    DESCRIPTION_CUE_RIGHT:'Cue onset right (class 2)',
    DESCRIPTION_BCI_FEEDBACK:'BCI feedback (continuous)',
    DESCRIPTION_CUE_UNKNOWN:'Cue unknown',
    DESCRIPTION_SOMETHING:'DESCRIPTION_SOMETHING', 
    DESCRIPTION_SOMETHING2:'DESCRIPTION_SOMETHING 2', 
    DESCRIPTION_REJECTED_TRIAL:'Rejected trial',
    DESCRIPTION_UNKNOWN_GROUP:'Unkown Group',
    DESCRIPTION_EYE_MOVEMENT_HORIZONTAL:'Horizontal eye movement',
    DESCRIPTION_EYE_MOVEMENT_VERTICAL:'Vertical eye movement',
    DESCRIPTION_EYE_ROTATION:'Eye rotation',
    DESCRIPTION_EYE_BLINK:'Eye blinks',
    DESCRIPTION_START_NEW_RUN:'Start of a new run'
}

# NOTE: Set the events you want to export from your dataset
valid_events_descriptions = [DESCRIPTION_CUE_LEFT, DESCRIPTION_CUE_RIGHT]

# NOTE: Start time window padding. Negative values are accepted.
t_start = 0

# NOTE: End time window padding
duration = 4 # imagery duration is 4 seconds

##########################################################################
from TS2Image import TS2Image
ts2i = TS2Image(input_folder=input_folder, output_folder=output_folder)
ts2i.generate_images(method="GAF", valid_events_descriptions=valid_events_descriptions, events_dictionary=BCI_competition_dataset_events_dictionary, t_start=t_start, duration=duration)
##########################################################################
from Logger import log
log('End main')
##########################################################################