#!/bin/bash

train_tts=true
run_tts=true
voice_name="mandarin_voice"

# train tts system
if [ "$train_tts" = true ]; then
    # step 1: run setup and check data
    ./01_setup.sh $voice_name

    # step 2: prepare labels
    ./02_prepare_labels.sh database/labels database/prompt-lab

    if [ ! -d 'database/feats' ]; then
        # step 3: extract acoustic features
        ./03_prepare_acoustic_features.sh database/wav database/feats
    else
        echo "---Step3 database/feats dir exists! skip this step!----"
    fi

    # step 4: prepare config files for training and testing
    ./04_prepare_conf_files.sh conf/global_settings.cfg

    # step 5: train duration model
    ./05_train_duration_model.sh conf/duration_${voice_name}.conf
    
    # step 6: train acoustic model
    ./06_train_acoustic_model.sh conf/acoustic_${voice_name}.conf

fi

# run tts
if [ "$run_tts" = true ]; then

    # step 7: run text to speech
   ./07_run_merlin.sh conf/test_dur_synth_${voice_name}.conf conf/test_synth_${voice_name}.conf

fi

echo "done...!"
