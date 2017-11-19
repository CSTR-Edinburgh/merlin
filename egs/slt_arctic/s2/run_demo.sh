#!/bin/bash -e

if test "$#" -ne 0; then
    echo "Usage: ./run_demo.sh"
    exit 1
fi

### Step 1: setup directories and the training data files ###
#./01_setup.sh slt_arctic_demo_magphase

### Step 2: prepare config files for acoustic, duration models and for synthesis ###
#./02_prepare_conf_files.sh conf/global_settings.cfg


### Step 3: Convert constant-frame-rate to variable-frame-rate (workaround) ###
#./03_convert_labels_to_var_frame_rate.sh conf/acous_train_slt_arctic_demo_magphase.conf


### Step 4: train duration model ###
#./04_train_duration_model.sh conf/dur_train_slt_arctic_demo_magphase.conf

### Step 5: train acoustic model ###
#./05_train_acoustic_model.sh conf/acous_train_slt_arctic_demo_magphase.conf


### Step 6: synthesize speech ###
./06_synthesise_waveforms.sh conf/dur_synth_slt_arctic_demo_magphase.conf conf/acous_synth_slt_arctic_demo_magphase.conf


