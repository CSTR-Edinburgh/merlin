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

setup_data=true

# For demo purpose we use the data of 10 speakers only!
# Build an average voice model (AVM) with 9 speakers
avg_voice="p225 p226 p227 p228 p229 p230 p231 p232 p233"

# setup directory structure and copy the data
if [ "$setup_data" = true ]; then
    # copy the data
    mkdir -p database
    mkdir -p database/wav
    mkdir -p database/txt    
    
    for spkid in $avg_voice; do
        echo "copying the speaker $spkid data to database"
        cp VCTK-Corpus/wav48/$spkid/*.wav database/wav
        cp VCTK-Corpus/txt/$spkid/*.txt database/txt
    done
    
    # create
    for spkid in $adapt_voice; do
        mkdir -p database_$spkid
        mkdir -p database_$spkid/wav
        mkdir -p database_$spkid/txt
        echo "copying the speaker $spkid data to database_$spkid"
        cp VCTK-Corpus/wav48/$spkid/*.wav database_$spkid/wav
        cp VCTK-Corpus/txt/$spkid/*.txt database_$spkid/txt
    done
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
echo "SamplingFreq=48000" >> $global_config_file
echo "SilencePhone='sil'" >> $global_config_file
echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "######### No. of files ###############" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Train=3258" >> $global_config_file 
echo "Valid=50" >> $global_config_file 
echo "Test=50" >> $global_config_file 
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# TOOLS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

#echo "ESTDIR=${merlin_dir}/tools/speech_tools" >> $global_config_file
#echo "FESTDIR=${merlin_dir}/tools/festival" >> $global_config_file
#echo "FESTVOXDIR=${merlin_dir}/tools/festvox" >> $global_config_file
echo "ESTDIR=/l/SRC/speech_tools/bin" >> $global_config_file
echo "FESTDIR=/l/SRC/festival_2_4/festival" >> $global_config_file
echo "FESTVOXDIR=/l/SRC/festvox/" >> $global_config_file
echo "" >> $global_config_file
#echo "HTKDIR=${merlin_dir}/tools/bin/htk" >> $global_config_file
echo "HTKDIR=/l/SRC/htk-3.5/bin" >> $global_config_file
echo "" >> $global_config_file

echo "Step 1:"
echo "Merlin default voice settings configured in \"$global_config_file\""
echo "Modify these params as per your data..."
echo "eg., sampling frequency, no. of train files etc.,"
echo "setup done...!"

