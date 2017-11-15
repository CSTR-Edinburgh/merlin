#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./03_convert_labels_to_var_frame_rate.sh <path_to_acoustic_training_conf_file>"
    echo ""
    echo "Default path to acoustic training conf file: conf/acous_train_${Voice}.conf"
    echo "################################"
    exit 1
fi

acoustic_conf_file=$1

### Step 3: Converting constant frame rate to variable frame rate labels (workaround) ###
echo "Step 3:"
echo "Converting constant frame rate to variable frame rate labels..."
python ./scripts/convert_label_state_align_to_variable_frame_rate.py $acoustic_conf_file