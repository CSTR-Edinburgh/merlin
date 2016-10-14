#!/bin/bash -e

if test "$#" -ne 0; then
    echo "Usage: ./run_merlin.sh"
    exit 1
fi

### Step 1: prepare config files for acoustic, duration models and for test synthesis ###
echo "Step 1: preparing config files for acoustic, duration models and for test synthesis..."
global_config_file=conf/global_settings.cfg

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

./scripts/prepare_config_files.sh $global_config_file
./scripts/prepare_config_files_for_synthesis.sh $global_config_file

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
echo "All successfull!! Your demo voice is ready :)"

