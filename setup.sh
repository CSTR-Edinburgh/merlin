#!/bin/sh

if test "$#" -ne 1; then
    echo "Usage: ./setup.sh <voice_directory_name>"
    exit 1
fi

if [ ! -f ./run_lstm.py ]; then
    echo "Merlin directory not found!"
fi

merlin_dir=$(pwd)
experiments_dir=${merlin_dir}/experiments

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}

global_config_file=configuration/merlin_voice_settings.cfg

### default settings ###
echo "Merlin=${merlin_dir}" >  $global_config_file
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

