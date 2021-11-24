# TS2Image 🌠
A tool to read EEG data from GDF files and export as images using [Gramian Angular Field](https://aaai.org/ocs/index.php/WS/AAAIW15/paper/viewFile/10179/10251) or [ERSP](https://sccn.ucsd.edu/~scott/pdf/ERSP93.pdf).

## Setup

### Dependencies
To install all dependencies run the following in your terminal:
```
pip -r install requirements.txt
```

### Configuration
1. Set the directory containing the files you want to process:
```
input_folder = current_working_directory + "/datasets"
```

2. Set the root output folder:
```
output_folder = current_working_directory + '/images-output'
```

3. Create according to your dataset. For example, these are the events from some files in BCI IV competition dataset:
```
DESCRIPTION_EYES_OPEN = "276"
DESCRIPTION_EYES_CLOSED = "277"
DESCRIPTION_START_TRIAL = "768"
DESCRIPTION_CUE_LEFT = "769"
DESCRIPTION_CUE_RIGHT = "770"
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
    DESCRIPTION_REJECTED_TRIAL:'Rejected trial',
    DESCRIPTION_UNKNOWN_GROUP:'Unkown Group',
    DESCRIPTION_EYE_MOVEMENT_HORIZONTAL:'Horizontal eye movement',
    DESCRIPTION_EYE_MOVEMENT_VERTICAL:'Vertical eye movement',
    DESCRIPTION_EYE_ROTATION:'Eye rotation',
    DESCRIPTION_EYE_BLINK:'Eye blinks',
    DESCRIPTION_START_NEW_RUN:'Start of a new run'
}
```

4. List of events from `events_dictionary` you want to export from your dataset:
```
valid_events_descriptions = [DESCRIPTION_CUE_LEFT, DESCRIPTION_CUE_RIGHT]
```

5. Start time window padding, in seconds. Negative values are accepted:
```
t_start = 0
```

6. Time window length, in seconds:
```
duration = 4
```

# References
[Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks](https://aaai.org/ocs/index.php/WS/AAAIW15/paper/viewFile/10179/10251)
