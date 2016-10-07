#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./text_to_speech.sh"
    exit 1
fi

global_config_file=conf/global_settings.cfg

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

### Step 1: create label files from text ###
echo "Step 1: creating label files from text..."
./scripts/prepare_labels_from_txt.sh $global_config_file

### Step 2: synthesize speech   ###
echo "Step 2: synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_dur_synth_${Voice}.conf
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_synth_${Voice}.conf

### Step 3: delete intermediate synth files ###
echo "Step 3: deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"

