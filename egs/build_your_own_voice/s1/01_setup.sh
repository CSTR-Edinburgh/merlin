#!/bin/bash

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./01_setup.sh <voice_name>"
    echo ""
    echo "Give a voice name eg., slt_arctic"
    echo "################################"
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

mkdir -p ${data_dir}
mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}
mkdir -p ${synthesis_dir}
mkdir -p ${acoustic_dir}/data
mkdir -p ${duration_dir}/data
mkdir -p ${synthesis_dir}/txt

### create some test files ###
echo "Hello world." > ${synthesis_dir}/txt/test_001.txt
echo "Hi, this is a demo voice from Merlin." > ${synthesis_dir}/txt/test_002.txt
echo "Hope you guys enjoy free open-source voices from Merlin." > ${synthesis_dir}/txt/test_003.txt
printf "test_001\ntest_002\ntest_003" > ${synthesis_dir}/test_id_list.scp

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

echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "QuestionFile=questions-radio_dnn_416.hed" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file
echo "SilencePhone='sil'" >> $global_config_file
echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "######### No. of files ###############" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Train=50" >> $global_config_file 
echo "Valid=5" >> $global_config_file 
echo "Test=5" >> $global_config_file 
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# TOOLS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "ESTDIR=${merlin_dir}/tools/speech_tools" >> $global_config_file
echo "FESTDIR=${merlin_dir}/tools/festival" >> $global_config_file
echo "FESTVOXDIR=${merlin_dir}/tools/festvox" >> $global_config_file
echo "" >> $global_config_file
echo "HTKDIR=${merlin_dir}/tools/bin/htk" >> $global_config_file
echo "" >> $global_config_file

echo "Step 1:"
echo "Merlin default voice settings configured in \"$global_config_file\""
echo "Modify these params as per your data..."
echo "eg., sampling frequency, no. of train files etc.,"
echo "setup done...!"

