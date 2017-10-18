#!/bin/bash

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./01_setup.sh merlin_benchmark"
    echo "################################"
    exit 1
fi

### Step 1: setup directories and the training data files ###
echo "Step 1:"

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

data_dir="blizzard2017-merlin-benchmark-data"

if [[ ! -f ${data_dir}.zip ]]; then
    echo "downloading data....."
    rm -f ${data_dir}.zip
    data_url=http://datashare.is.ed.ac.uk/bitstream/handle/10283/2909/blizzard2017-merlin-benchmark-data.zip
    if hash wget 2>/dev/null; then
        wget $data_url
    elif hash curl 2>/dev/null; then
        curl -L -O $data_url
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
    mkdir -p ${duration_dir}/data
    mkdir -p ${acoustic_dir}/data
    cp -r ${data_dir}/labels/unilex/label_state_align ${duration_dir}/data/label_state_align
    cp -r ${data_dir}/labels/unilex/label_state_align ${acoustic_dir}/data/label_state_align
    cp -r ${data_dir}/feats/world/* ${acoustic_dir}/data
    cp -r ${data_dir}/file_id_list.scp ${duration_dir}/data/
    cp -r ${data_dir}/file_id_list.scp ${acoustic_dir}/data/
fi
echo "data is ready!"

global_config_file=conf/global_settings.cfg

### default settings ###
echo "MerlinDir=${merlin_dir}" >  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "QuestionFile=questions-unilex_dnn_600.hed" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=48000" >> $global_config_file

echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "Train=6866" >> $global_config_file 
echo "Valid=134" >> $global_config_file 
echo "Test=253" >> $global_config_file 

echo "Merlin default voice settings configured in $global_config_file"
echo "setup done...!"

