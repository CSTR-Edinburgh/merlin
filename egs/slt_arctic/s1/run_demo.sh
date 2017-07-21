#!/bin/bash -e

if test "$#" -ne 0; then
    echo "Usage: ./run_demo.sh"
    exit 1
fi

### Step 1: setup directories and the training data files ###
./01_setup.sh slt_arctic_demo

### Step 2: prepare config files for acoustic, duration models and for synthesis ###
./02_prepare_conf_files.sh conf/global_settings.cfg

### Step 3: train duration model ###
./03_train_duration_model.sh conf/duration_slt_arctic_demo.conf

### Step 4: train acoustic model ###
./04_train_acoustic_model.sh conf/acoustic_slt_arctic_demo.conf 

### Step 5: synthesize speech ###
./05_run_merlin.sh conf/test_dur_synth_slt_arctic_demo.conf conf/test_synth_slt_arctic_demo.conf 


