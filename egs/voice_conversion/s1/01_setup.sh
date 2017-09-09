#!/bin/bash

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "$0 <src_speaker> <tgt_speaker>"
    echo ""
    echo "Give a source speaker name eg., bdl"
    echo "Give a target speaker name eg., slt"
    echo "################################"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments
data_dir=${current_working_dir}/database

src_speaker=$1
tgt_speaker=$2

voice_name=$12$2

src_voice_dir=${experiments_dir}/${src_speaker}
tgt_voice_dir=${experiments_dir}/${voice_name}

src_acoustic_dir=${src_voice_dir}/acoustic_model
tgt_acoustic_dir=${tgt_voice_dir}/acoustic_model

synthesis_dir=${tgt_voice_dir}/test_synthesis

mkdir -p ${data_dir}
mkdir -p ${data_dir}/$src_speaker
mkdir -p ${data_dir}/$tgt_speaker

mkdir -p ${experiments_dir}
mkdir -p ${src_voice_dir}
mkdir -p ${tgt_voice_dir}
mkdir -p ${src_acoustic_dir}
mkdir -p ${tgt_acoustic_dir}
mkdir -p ${src_acoustic_dir}/data
mkdir -p ${tgt_acoustic_dir}/data
mkdir -p ${tgt_acoustic_dir}/inter_module
mkdir -p ${synthesis_dir}

# create an empty question file
touch ${merlin_dir}/misc/questions/questions-vc.hed

global_config_file=conf/global_settings.cfg

### default settings ###
echo "######################################" > $global_config_file
echo "############# PATHS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "MerlinDir=${merlin_dir}" >>  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# PARAMS #################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Source=${src_speaker}" >> $global_config_file
echo "Target=${tgt_speaker}" >> $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "" >> $global_config_file

echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file
echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "######### No. of files ###############" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Train=250" >> $global_config_file 
echo "Valid=25" >> $global_config_file 
echo "Test=25" >> $global_config_file 
echo "" >> $global_config_file

echo "Step 1:"
echo "Merlin default voice-conversion settings configured in \"$global_config_file\""
echo "Modify these params as per your data..."
echo "eg., sampling frequency, no. of train files etc.,"
echo "setup done...!"

