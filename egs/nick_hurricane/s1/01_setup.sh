#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage:"
    echo "To run on short data:"
    echo "./setup.sh nick_hurricane_demo"
    echo "                  (or)                " 
    echo "To run on full data:"
    echo "./setup.sh nick_hurricane_full"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}

if [ "$voice_name" == "nick_hurricane_demo" ]
then
    data_dir=nick_hurricane_demo_data
elif [ "$voice_name" == "nick_hurricane_full" ]
then
    data_dir=nick_hurricane_full_data
else
    echo "The data for voice name ($voice_name) is not available...please use nick_hurricane_demo or nick_hurricane_full !!"
    exit 1
fi

if [[ ! -f ${data_dir}.zip ]]; then
    echo "downloading data....."
    rm -f ${data_dir}.zip
    data_url=http://104.131.174.95/${data_dir}.zip
    if hash curl 2>/dev/null; then
        curl -O $data_url
    elif hash wget 2>/dev/null; then
        wget $data_url
    else
        echo "please download the data from $data_url"
        exit 1
    fi
    do_unzip=true
fi
if [[ ! -d ${data_dir} ]] || [[ -n "$do_unzip" ]]; then
    echo "unzipping files......"
    rm -fr ${data_dir}
    rm -fr ${duration_dir}/data
    rm -fr ${acoustic_dir}/data
    unzip -q ${data_dir}.zip
    mv ${data_dir}/merlin_baseline_practice/duration_data/ ${duration_dir}/data
    mv ${data_dir}/merlin_baseline_practice/acoustic_data/ ${acoustic_dir}/data
    mv ${data_dir}/merlin_baseline_practice/test_data/ ${synthesis_dir}
fi
echo "data is ready!"

global_config_file=conf/global_settings.cfg

### default settings ###
echo "MerlinDir=${merlin_dir}" >  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "QuestionFile=questions-combilex_dnn_481.hed" >> $global_config_file
echo "Vocoder=STRAIGHT" >> $global_config_file
echo "SamplingFreq=48000" >> $global_config_file

if [ "$voice_name" == "nick_hurricane_demo" ]
then
    echo "FileIDList=file_id_list_demo.scp" >> $global_config_file
    echo "Train=50" >> $global_config_file 
    echo "Valid=5" >> $global_config_file 
    echo "Test=5" >> $global_config_file 
elif [ "$voice_name" == "nick_hurricane_full" ]
then
    echo "FileIDList=file_id_list_full.scp" >> $global_config_file
    echo "Train=2400" >> $global_config_file 
    echo "Valid=70" >> $global_config_file 
    echo "Test=72" >> $global_config_file 
else
    echo "The data for voice name ($voice_name) is not available...please use nick_hurricane_demo or nick_hurricane_full !!"
    exit 1
fi

echo "Merlin default voice settings configured in $global_config_file"
echo "setup done...!"

