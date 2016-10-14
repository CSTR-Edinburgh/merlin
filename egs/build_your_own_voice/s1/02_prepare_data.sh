#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./02_prepare_data.sh"
    exit 1
fi

PrintUsage () {
    echo "Please first run setup!!"
    echo "To run:"
    echo "Step 1: ./01_setup.sh give_your_voice_name"
}

global_config_file=conf/global_settings.cfg

if [ ! -f  $global_config_file ]; then
    echo "Global config file doesn't exist"
    PrintUsage
    exit 1
else
    source $global_config_file
fi

### define few variables here
data_dir=${current_working_dir}/database

wav_dir=${data_dir}/wav
txt_dir=${data_dir}/txt
txt_file=${data_dir}/utts.data

if [[ ! -d "${data_dir}" ]]; then
    PrintUsage
    exit 1
fi

if [[ ! -d "${txt_dir}" ]] && [[ ! -f "${txt_file}" ]]; then
    echo "Please give input: either 1 or 2"
    echo "1. ${txt_dir}  -- a text directory containing text files"
    echo "2. ${txt_file} -- a single text file with each sentence in a new line in festival format"
    exit 1
fi


####################################
##### prepare vocoder features #####
####################################

echo "Follow the scripts in:"
echo "https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/vocoder"

####################################
######## prepare labels ############
####################################

echo "Follow the scripts in:"
echo "https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/alignment"

