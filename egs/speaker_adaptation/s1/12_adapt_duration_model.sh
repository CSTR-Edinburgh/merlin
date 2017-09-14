#!/bin/bash -e

global_config_file=conf/global_settings_adapt.cfg
source $global_config_file

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./12_adapt_duration_model.sh <path_to_duration_conf_file>"
    echo ""
    echo "Default path to duration conf file: conf/duration_${Voice}.conf"
    echo "################################"
    exit 1
fi

duration_conf_file=$1

### Step 5: train duration model ###
echo "Step 5:"
echo "training duration model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $duration_conf_file


