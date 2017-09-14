#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 3; then
    echo "################################"
    echo "Usage:"
    echo "./02_prepare_labels.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir>"
    echo ""
    echo "default path to wav dir(Input): database/wav"
    echo "default path to txt dir(Input): database/txt"
    echo "default path to lab dir(Output): database/labels"
    echo "################################"
    exit 1
fi

wav_dir=$1
inp_txt=$2
lab_dir=$3

if [ ! "$(ls -A ${wav_dir})" ]; then
    echo "Please place your audio files in: ${wav_dir}"
    exit 1
fi

if [ ! "$(ls -A ${inp_txt})" ]; then
    echo "Please give text as input in format: either 1 or 2"
    echo "1. database/txt       -- a text directory containing text files"
    echo "2. database/utts.data -- a single text file with each sentence in a new line in festival format"
    exit 1
fi

####################################
########## Prepare labels ##########
####################################

prepare_labels=true
copy=true

if [ "$prepare_labels" = true ]; then
    echo "Step 2: "
    echo "Preparing labels..."

    if [ "$Labels" == "state_align" ]
    then
        ./scripts/run_state_aligner.sh $wav_dir $inp_txt $lab_dir $global_config_file 
    elif [ "$Labels" == "phone_align" ]
    then
        ./scripts/run_phone_aligner.sh $wav_dir $inp_txt $lab_dir $global_config_file 
    else
        echo "These labels ($Labels) are not supported as of now...please use state_align or phone_align!!"
    fi
fi

if [ "$copy" = true ]; then
    echo "Copying labels to duration and acoustic data directories..."
    
    duration_data_dir=experiments/${Voice}/duration_model/data
    acoustic_data_dir=experiments/${Voice}/acoustic_model/data
    
    cp -r $lab_dir/label_$Labels $duration_data_dir 
    cp -r $lab_dir/label_$Labels $acoustic_data_dir
    
    ls $lab_dir/label_$Labels > $duration_data_dir/$FileIDList
    ls $lab_dir/label_$Labels > $acoustic_data_dir/$FileIDList
    
    sed -i 's/\.lab//g' $duration_data_dir/$FileIDList
    sed -i 's/\.lab//g' $acoustic_data_dir/$FileIDList
    
    echo "done...!"
fi
