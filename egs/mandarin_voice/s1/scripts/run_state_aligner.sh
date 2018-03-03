#!/bin/bash

if test "$#" -ne 4; then
    echo "Usage: ./run_state_aligner.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir> <path_to_global_conf_file>"
    exit 1
fi

### Arguments
wav_dir=$1
inp_txt=$2
lab_dir=$3
global_config_file=$4

### Use paths from global config file
source $global_config_file

### force-alignment scripts
aligner=${MerlinDir}/misc/scripts/alignment/state_align

# initializations
train=true

####################################
######## prepare labels ############
####################################

### do prepare full-contextual labels without timestamps
echo "preparing full-contextual labels using Festival frontend..."
bash ${WorkDir}/scripts/prepare_labels_from_txt.sh $inp_txt $lab_dir $global_config_file $train

status_prev_step=$?
if [ $status_prev_step -eq 1 ]; then
    echo "Preparation of full-contextual labels unsuccessful!!"
    echo "Please check scripts/prepare_labels_from_txt.sh"
    exit 1
fi

### tools required
if [[ ! -d "${HTKDIR}" ]]; then
    echo "Please configure path to HTK tools in $global_config_file !!"
    exit 1
fi

### do forced alignment using HVite 
echo "forced-alignment using HTK tools..."

sed -i s#'HTKDIR =.*'#'HTKDIR = "'$HTKDIR'"'# $aligner/forced_alignment.py
sed -i s#'work_dir =.*'#'work_dir = "'$WorkDir/$lab_dir'"'# $aligner/forced_alignment.py
sed -i s#'wav_dir =.*'#'wav_dir = "'$WorkDir/$wav_dir'"'# $aligner/forced_alignment.py

python $aligner/forced_alignment.py

state_labels=$lab_dir/label_state_align

if [ ! "$(ls -A ${state_labels})" ]; then
    echo "Force-alignment unsucessful!! Please check $aligner/forced_alignment.py"
else
    echo "You should have your labels ready in: $state_labels !!"
fi

