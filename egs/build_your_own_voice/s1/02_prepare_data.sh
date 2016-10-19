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
PrintUsageOne () {
    echo ""
    echo "Please place your audio files in: ${wav_dir}"
    echo ""
}
PrintUsageTwo () {
    echo ""
    echo "Please give input: either 1 or 2"
    echo "1. ${txt_dir}  -- a text directory containing text files"
    echo "2. ${txt_file} -- a single text file with each sentence in a new line in festival format"
    echo ""
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
#data_dir=${WorkDir}/database
data_dir=database

wav_dir=${data_dir}/wav
txt_dir=${data_dir}/txt
txt_file=${data_dir}/utts.data

if [ ! "$(ls -A ${wav_dir})" ]; then
    PrintUsageOne
    PrintUsageTwo
    exit 1
fi

if [[ ! -d "${txt_dir}" ]] && [[ ! -f "${txt_file}" ]]; then
    PrintUsageTwo
    exit 1
fi


####################################
##### prepare vocoder features #####
####################################

echo ""
echo "Step 1: Prepare vocoder features"
echo "To prepare vocoder features, follow the scripts in:"
echo "https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/vocoder"
echo ""

####################################
######## prepare labels ############
####################################

echo ""
echo "Step 2: Prepare labels"
echo "To prepare labels, follow the scripts in:"
echo "https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/alignment"
echo ""

################################################
######## automatic data preparation ############
################################################

echo ""
echo "Scripts for automatic data preparation are still under development."
echo "If you want to contribute, this is the choice, please let us know!! :)"
echo ""
