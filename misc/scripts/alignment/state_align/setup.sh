#!/bin/bash

if test "$#" -ne 0; then
    echo "Usage: ./setup.sh"
    exit 1
fi

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $(dirname $current_working_dir))))

### Download data -- slt cmuarctic.data

audio_data=slt_wav

audio_data_url=http://104.131.174.95/${audio_data}.zip
text_data_url=http://festvox.org/cmu_arctic/cmuarctic.data

if [[ ! -f ${audio_data}.zip ]]; then
    echo "downloading data....."
    rm -f ${audio_data}.zip
    if hash curl 2>/dev/null; then
        curl -O $audio_data_url
        curl -O $text_data_url
    elif hash wget 2>/dev/null; then
        wget $audio_data_url
        wget $text_data_url
    else
        echo "please download the audio data from $audio_data_url"
        echo "please download the text data from $text_data_url"
        exit 1
    fi
    do_unzip=true
fi

if [[ ! -d ${audio_data} ]] || [[ -n "$do_unzip" ]]; then
    echo "unzipping files....."
    unzip -q ${audio_data}.zip
fi

### default settings ###
config_file=config.cfg

echo "######################################" > $config_file
echo "############# PATHS ##################" >> $config_file
echo "######################################" >> $config_file
echo "" >> $config_file

echo "MerlinDir=${merlin_dir}" >>  $config_file
echo "frontend=${merlin_dir}/misc/scripts/frontend" >> $config_file
echo "WorkDir=${current_working_dir}" >>  $config_file
echo "" >> $config_file

echo "######################################" >> $config_file
echo "############# TOOLS ##################" >> $config_file
echo "######################################" >> $config_file
echo "" >> $config_file

echo "FESTDIR=${merlin_dir}/tools/festival" >> $config_file
echo "HTKDIR=${merlin_dir}/tools/htk" >> $config_file
echo "" >> $config_file

echo "Path to Merlin and other tools configured in $config_file"
echo "setup done...!"
