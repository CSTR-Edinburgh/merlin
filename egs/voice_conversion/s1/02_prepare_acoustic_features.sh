#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "$0 <path_to_wav_dir> <path_to_feat_dir>"
    echo ""
    echo "default path to wav dir(Input): database/${Source}/wav"
    echo "default path to feat dir(Output): database/${Source}/feats"
    echo "################################"
    exit 1
fi

wav_dir=$1
feat_dir=$2

if [ ! "$(ls -A ${wav_dir})" ]; then
    echo "Please place your audio files in: ${wav_dir}"
    exit 1
fi

####################################
##### prepare vocoder features #####
####################################

prepare_feats=true

if [ "$prepare_feats" = true ]; then
    echo "Step 2:" 
    echo "Prepare acoustic features using "${Vocoder}" vocoder..."
    python ${MerlinDir}/misc/scripts/vocoder/${Vocoder,,}/extract_features_for_merlin.py ${MerlinDir} ${wav_dir} ${feat_dir} $SamplingFreq 
fi

