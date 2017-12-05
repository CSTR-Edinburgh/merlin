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

src_mag_dir=$src_feat_dir
tgt_mag_dir=$tgt_feat_dir

if [ ! "$(ls -A ${src_mag_dir})" ] || [ ! "$(ls -A ${tgt_mag_dir})" ]; then
    echo "Please run 03_prepare_acoustic_features.sh script for both ${Source} and ${Target}"
    exit 1
fi

#######################################
##### align src features with tgt #####
#######################################

align_feats=true
copy=true

if [ "$align_feats" = true ]; then
    echo "Step 3:" 
    echo "Align source acoustic features with target acoustic features..."
    python ${MerlinDir}/misc/scripts/voice_conversion/dtw_aligner_magphase.py ${MerlinDir}/tools ${src_feat_dir} ${tgt_feat_dir} ${src_aligned_feat_dir}
fi

if [ "$copy" = true ]; then
    echo "Copying features to acoustic data directory..."
    src_acoustic_data_dir=experiments/${Source}/acoustic_model/data
    tgt_acoustic_data_dir=experiments/${Voice}/acoustic_model/data
    cp -r ${src_aligned_feat_dir}/* ${src_acoustic_data_dir}/feats
    cp -r ${tgt_feat_dir}/* ${tgt_acoustic_data_dir}/feats
    
    echo "preparing file list..."
    basename -s .mag ${src_aligned_feat_dir}/*.mag > file_id_list.scp
    cp file_id_list.scp $src_acoustic_data_dir
    mv file_id_list.scp $tgt_acoustic_data_dir
    echo "done...!"
fi

