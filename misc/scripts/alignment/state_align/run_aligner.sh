#!/bin/bash -e

if test "$#" -ne 1; then
    echo "Usage: ./run_aligner.sh config.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Config file doesn't exist"
    exit 1
else
    source $1
fi

#############################################################
##### Create training labels for merlin with HTK tools ######
#############################################################

### convert text to festival utts and full-contextual labels

### Step 1: create label files from text ###
echo "Step 1: creating label files from text..."
./prepare_labels_from_txt.sh config.cfg

status_step1=$?
if [ $status_step1 -eq 1 ]; then
    echo "Step 1 not successful !!"
    exit 1
fi

### tools required

if [[ ! -d "${HTKDIR}" ]]; then
    echo "Please configure path to HTK tools in config.cfg !!"
    exit 1
fi

### Step 2: do forced alignment using HVite 
echo "Step 2: forced-alignment using HTK tools..."

sed -i s#'HTKDIR =.*'#'HTKDIR = "'$HTKDIR'"'# forced_alignment.py
sed -i s#'work_dir =.*'#'work_dir = "'$WorkDir'"'# forced_alignment.py

python forced_alignment.py

echo "You should have your labels ready in: label_state_align !!"

