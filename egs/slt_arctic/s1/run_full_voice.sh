#!/bin/bash

usage ()
{
    echo "Usage: ./run_full_voice.sh voice"
    echo "  options:"
    echo "  voice  Specify voice to use. Example:"
    echo "      female voice: slt"
    echo "      male voice: bdl"
    echo "      Default: slt. If no voice specified then 'slt' will be used."
    exit 1
}

if [ "$#" -eq 0 ] || [ "$1" == "slt" ]
then
  voice=slt_arctic_full
elif [ "$1" == "bdl" ]
then
  voice=bdl_arctic_full
else
  usage
fi

### Step 1: setup directories and the training data files ###
echo "Step 1: setting up experiments directory and the training data files. Selected voice:$voice"
global_config_file=conf/global_settings.cfg
./scripts/setup.sh $voice
./scripts/prepare_config_files.sh $global_config_file
./scripts/prepare_config_files_for_synthesis.sh $global_config_file

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

### Step 2: train duration model ###
echo "Step 2: training duration model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/duration_${Voice}.conf

### Step 3: train acoustic model ###
echo "Step 3: training acoustic model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/acoustic_${Voice}.conf

### Step 4: synthesize speech   ###
echo "Step 4: synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_dur_synth_${Voice}.conf
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py conf/test_synth_${Voice}.conf

### Step 5: delete intermediate synth files ###
echo "Step 5: deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh conf/global_settings.cfg

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"
echo "All successfull!! Your full voice is ready :)"

