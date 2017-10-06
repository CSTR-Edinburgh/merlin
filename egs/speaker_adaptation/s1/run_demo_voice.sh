#!/bin/bash

train_tts=true
run_tts=true
do_adaptation=true

if test "$#" -ne 1; then
    voice_name="vctk_avm"
else
    voice_name=$1
    setup_data=false # assuming wav and txt in database
fi


# train tts system
if [ "$train_tts" = true ]; then
    # step 1: run setup
    ./01_setup.sh $voice_name

    # step 2: prepare labels
    ./02_prepare_labels.sh database/wav database/txt database/labels

    # step 3: extract acoustic features
    ./03_prepare_acoustic_features.sh database/wav database/feats

    # step 4: prepare config files for training and testing
    ./04_prepare_conf_files.sh conf/global_settings.cfg

    # step 5: train duration model
    ./05_train_duration_model.sh conf/duration_${voice_name}.conf
    
    # step 6: train acoustic model
    ./06_train_acoustic_model.sh conf/acoustic_${voice_name}.conf

fi

# run tts
if [ "$run_tts" = true ]; then

    basename --suffix=.txt -- experiments/${voice_name}/test_synthesis/txt/* > experiments/${voice_name}/test_synthesis/test_id_list.scp

    # step 7: run text to speech
   ./07_run_merlin.sh experiments/${voice_name}/test_synthesis/txt conf/test_dur_synth_${voice_name}.conf conf/test_synth_${voice_name}.conf

fi

# start adaptation

if [ "$do_adaptation" = true ]; then
    # step 8: run setup
    ./08_setup_adapt.sh adapt_p234 experiments/vctk_avm/duration_model/nnets_model/feed_forward_6_tanh.model experiments/vctk_avm/acoustic_model/nnets_model/feed_forward_6_tanh.model fine_tune
    
    # step 9: prepare labels
    ./09_prepare_labels_adapt.sh database_p234/wav database_p234/txt database_p234/labels

    # step 10: extract acoustic features
    ./10_prepare_acoustic_features.sh database_p234/wav database_p234/feats/

    # step 11: prepare config files for training and testing
    ./11_prepare_conf_files_adapt.sh conf/global_settings_adapt.cfg

    # step 12: adapt acoustic model
    ./12_adapt_duration_model.sh conf/duration_adapt_p234_fine_tune.conf

    # step 13: adapt duration model
    ./13_adapt_acoustic_model.sh conf/duration_adapt_p234_fine_tune.conf

fi
echo "done...!"
