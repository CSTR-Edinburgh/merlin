#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./01_setup.sh <voice_name>"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments
data_dir=${current_working_dir}/database

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}

mkdir -p ${data_dir}
mkdir -p ${data_dir}/wav
mkdir -p ${data_dir}/txt

global_config_file=conf/global_settings.cfg

### default settings ###
echo "MerlinDir=${merlin_dir}" >  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=phone_align" >> $global_config_file
echo "QuestionFile=questions-radio_dnn_416.hed" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file

echo "FileIDList=file_id_list_full.scp" >> $global_config_file
echo "Train=50" >> $global_config_file 
echo "Valid=5" >> $global_config_file 
echo "Test=5" >> $global_config_file 

echo "Merlin default voice settings configured in $global_config_file"
echo "setup done...!"

