#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./setup.sh <voice_directory_name>"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}

echo "downloading data....."
wget http://104.131.174.95/slt_arctic_demo.zip
echo "unzipping files......"
unzip -q slt_arctic_demo.zip
mv slt_arctic_demo/merlin_baseline_practice/duration_data/ ${duration_dir}/data
mv slt_arctic_demo/merlin_baseline_practice/acoustic_data/ ${acoustic_dir}/data
echo "data is ready!"

global_config_file=conf/global_settings.cfg

### default settings ###
echo "MerlinDir=${merlin_dir}" >  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "QuestionFile=questions-radio_dnn_416.hed" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file
echo "FileIDList=file_id_list_demo.scp" >> $global_config_file
echo "Train=50" >> $global_config_file 
echo "Valid=5" >> $global_config_file 
echo "Test=5" >> $global_config_file 

echo "Merlin default voice settings configured in $global_config_file"
echo "setup done...!"

