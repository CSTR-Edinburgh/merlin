#!/bin/bash

setup_data=true
train_vc=true
run_vc=true

if test "$#" -ne 2; then
    src_speaker="SF1"
    tgt_speaker="TM1"
else
    src_speaker=$1
    tgt_speaker=$2
fi

# setup directory structure and download data
if [ "$setup_data" = true ]; then
    # download vcc2016 training data
    wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/vcc2016_training.zip

    # unzip files
    unzip -q vcc2016_training.zip

    # delete zip file
    rm -rf vcc2016_training.zip
fi

# train voice conversion system for any source-target pair
if [ "$train_vc" = true ]; then
    # step 1: run setup
    ./01_setup_vcc2016.sh $src_speaker $tgt_speaker

    # step 2: extract acoustic features
    ./02_prepare_acoustic_features.sh vcc2016_training/$src_speaker database/$src_speaker/feats
    ./02_prepare_acoustic_features.sh vcc2016_training/$tgt_speaker database/$tgt_speaker/feats

    # step 3: align source features with target (create parallel data)
    ./03_align_src_with_target.sh database/$src_speaker/feats database/$tgt_speaker/feats database/${src_speaker}_aligned_with_${tgt_speaker}/feats

    # step 4: prepare config files for training and testing
    ./04_prepare_conf_files.sh conf/global_settings.cfg

    # step 5: train voice-conversion model
    # prepare acoustic features for source
    ./05_train_acoustic_model.sh conf/acoustic_${src_speaker}.conf

    # create a symbolic link for source features in target-voice directory
    ./scripts/create_symbolic_link.sh

    # train an acoustic model with mapping from source to target
    ./05_train_acoustic_model.sh conf/acoustic_${src_speaker}2${tgt_speaker}.conf
fi

# run voice conversion
if [ "$run_vc" = true ]; then

    mkdir -p experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}
    mkdir -p experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${tgt_speaker}
    
    if [ "$setup_data" = true ]; then
        # download evaluation data
        wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/evaluation_all.zip
        
        # unzip files
        unzip -q evaluation_all.zip

        # copy source audio files
        cp evaluation_all/${src_speaker}/* experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/

        # delete zip file
        rm -rf evaluation_all.zip
    fi

    basename --suffix=.wav -- experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/* > experiments/${src_speaker}2${tgt_speaker}/test_synthesis/test_id_list.scp

    # step 6: run voice conversion from source to target 
    ./06_run_merlin_vc.sh experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker} conf/test_synth_${src_speaker}.conf conf/test_synth_${src_speaker}2${tgt_speaker}.conf
fi

echo "done...!"
