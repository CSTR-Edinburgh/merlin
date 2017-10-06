#!/bin/bash

setup_data=true
train_vc=true
run_vc=true

if test "$#" -ne 2; then
    src_speaker="bdl"
    tgt_speaker="slt"
else
    src_speaker=$1
    tgt_speaker=$2
    setup_data=false
fi

# setup directory structure and download data
if [ "$setup_data" = true ]; then
    # download demo data (300 utterances)
    wget http://104.131.174.95/downloads/voice_conversion/bdl_arctic.zip
    wget http://104.131.174.95/downloads/voice_conversion/slt_arctic.zip

    # unzip files
    unzip -q bdl_arctic.zip
    unzip -q slt_arctic.zip

    mkdir -p database
    mkdir -p database/bdl
    mkdir -p database/slt

    # copy data
    mv bdl_arctic/wav database/bdl/wav
    mv slt_arctic/wav database/slt/wav

    rm -rf bdl_arctic slt_arctic
    rm -rf bdl_arctic.zip slt_arctic.zip
fi

# train voice conversion system for any source-target pair
if [ "$train_vc" = true ]; then
    # step 1: run setup
    ./01_setup.sh $src_speaker $tgt_speaker

    # step 2: extract acoustic features
    ./02_prepare_acoustic_features.sh database/$src_speaker/wav/ database/$src_speaker/feats
    ./02_prepare_acoustic_features.sh database/$tgt_speaker/wav/ database/$tgt_speaker/feats

    # step 3: align source features with target (create parallel data)
    ./03_align_src_with_target.sh database/$src_speaker/feats/ database/$tgt_speaker/feats/ database/${src_speaker}_aligned_with_${tgt_speaker}/feats

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
    
    # let's copy some files for testing
    if [ "$setup_data" = true ]; then
        cp database/${src_speaker}/wav/arctic_a0001.wav experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/test_01.wav
        cp database/${src_speaker}/wav/arctic_a0002.wav experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/test_02.wav
        cp database/${src_speaker}/wav/arctic_a0003.wav experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/test_03.wav
    fi

    basename -s .wav experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker}/*.wav > experiments/${src_speaker}2${tgt_speaker}/test_synthesis/test_id_list.scp

    # step 6: run voice conversion from source to target 
    ./06_run_merlin_vc.sh experiments/${src_speaker}2${tgt_speaker}/test_synthesis/${src_speaker} conf/test_synth_${src_speaker}.conf conf/test_synth_${src_speaker}2${tgt_speaker}.conf
fi

echo "done...!"
