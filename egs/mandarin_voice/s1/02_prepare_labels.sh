#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "./02_prepare_labels.sh <path_to_labels_dir>"
    echo ""
    echo "default path to lab dir(Output): database/labels"
    echo "default path to lab dir(Output): database/prompt-lab"
    echo "################################"
    exit 1
fi

lab_dir=$1
prompt_lab_dir=$2

####################################
########## Copy labels ##########
####################################

copy=true

if [ "$copy" = true ]; then
    echo "Step 2: "
    echo "Copying labels to duration and acoustic data directories..."
    
    duration_data_dir=experiments/${Voice}/duration_model/data
    acoustic_data_dir=experiments/${Voice}/acoustic_model/data
    synthesis_data_dir=experiments/${Voice}/test_synthesis
    
    cp -r $lab_dir/label_$Labels $duration_data_dir 
    cp -r $lab_dir/label_$Labels $acoustic_data_dir

    cp -r $prompt_lab_dir $synthesis_data_dir
    
    ls $lab_dir/label_$Labels > $duration_data_dir/$FileIDList
    ls $lab_dir/label_$Labels > $acoustic_data_dir/$FileIDList
    ls $prompt_lab_dir > $synthesis_data_dir/test_id_list.scp
    
    sed -i 's/\.lab//g' $duration_data_dir/$FileIDList
    sed -i 's/\.lab//g' $acoustic_data_dir/$FileIDList
    sed -i 's/\.lab//g' $synthesis_data_dir/test_id_list.scp
    
    echo "done...!"
fi
