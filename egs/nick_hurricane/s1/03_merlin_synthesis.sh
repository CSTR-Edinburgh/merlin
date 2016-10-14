#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./merlin_synthesis.sh"
    exit 1
fi

PrintUsage () {
    echo "Please first run either demo voice or full voice!!"
    echo "To run:"
    echo "Step 1: ./setup.sh nick_hurricane_demo"
    echo "                  (or)                " 
    echo "        ./setup.sh nick_hurricane_full"
    echo "                                      "
    echo "Step 2: ./run_merlin.sh               "
}

global_config_file=conf/global_settings.cfg

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    PrintUsage
    exit 1
else
    source $global_config_file
fi

### define few variables here
testDir=experiments/${Voice}/test_synthesis

txt_dir=${testDir}/txt
txt_file=${testDir}/utts.data

if [[ ! -d "${testDir}" ]]; then
    PrintUsage
    exit 1
fi

if [[ ! -d "${txt_dir}" ]] && [[ ! -f "${txt_file}" ]]; then
    echo "Please give input: either 1 or 2"
    echo "1. ${txt_dir}  -- a text directory containing text files"
    echo "2. ${txt_file} -- a single text file with each sentence in a new line in festival format"
    exit 1
fi

### Step 1: create label files from text ###
echo "Step 1: creating label files from text..."
./scripts/prepare_labels_from_txt.sh $global_config_file

status_step1=$?
if [ $status_step1 -eq 1 ]; then
    echo "Step 1 not successful !!"
    exit 1
fi

### Step 2: synthesize speech   ###
echo "Step 2: synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_dur_synth_${Voice}.conf
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_synth_${Voice}.conf

### Step 3: delete intermediate synth files ###
echo "Step 3: deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"

