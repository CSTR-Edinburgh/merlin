#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 3; then
    echo "################################"
    echo "Usage:"
    echo "$0 <path_to_src_feat_dir> <path_to_tgt_feat_dir> <path_to_src_align_dir>"
    echo ""
    echo "default path to source feat dir(Input): database/${Source}/feats"
    echo "default path to target feat dir(Input): database/${Target}/feats"
    echo "default path to aligned feat dir(Output): database/${Source}_aligned_with_${Target}/feats"
    echo "################################"
    exit 1
fi

src_feat_dir=$1
tgt_feat_dir=$2
src_aligned_feat_dir=$3

src_mgc_dir=$src_feat_dir/mgc
tgt_mgc_dir=$tgt_feat_dir/mgc

if [ ! "$(ls -A ${src_mgc_dir})" ] || [ ! "$(ls -A ${tgt_mgc_dir})" ]; then
    echo "Please run 02_prepare_acoustic_features.sh script for both ${Source} and ${Target}"
    exit 1
fi

# get bap dimension based on sampling rate and vocoder
function get_bap_dimension(){
    if [ "$Vocoder" == "STRAIGHT" ]
    then
        echo 25
    elif [ "$Vocoder" == "WORLD" ]
    then
        if [ "$SamplingFreq" == "16000" ]
        then
            echo 1
        elif [ "$SamplingFreq" == "48000" ]
        then
            echo 5
        fi
    else
        echo "This vocoder ($Vocoder) is not supported as of now...please configure yourself!!"
    fi
}

#######################################
##### align src features with tgt #####
#######################################

align_feats=true
copy=true

if [ "$align_feats" = true ]; then
    echo "Step 3:" 
    echo "Align source acoustic features with target acoustic features..."
    bap_dim=$(get_bap_dimension)
    python ${MerlinDir}/misc/scripts/voice_conversion/dtw_aligner_festvox.py ${MerlinDir}/tools ${src_feat_dir} ${tgt_feat_dir} ${src_aligned_feat_dir} ${bap_dim}
fi

if [ "$copy" = true ]; then
    echo "Copying features to acoustic data directory..."
    src_acoustic_data_dir=experiments/${Source}/acoustic_model/data
    tgt_acoustic_data_dir=experiments/${Voice}/acoustic_model/data
    cp -r ${src_aligned_feat_dir}/* $src_acoustic_data_dir
    cp -r ${tgt_feat_dir}/* $tgt_acoustic_data_dir
    
    echo "preparing file list..."
    basename --suffix=.mgc -- ${src_aligned_feat_dir}/mgc/* > file_id_list.scp
    cp file_id_list.scp $src_acoustic_data_dir
    mv file_id_list.scp $tgt_acoustic_data_dir
    echo "done...!"
fi

