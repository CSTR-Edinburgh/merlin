#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "$0 <path_to_acoustic_conf_file>"
    echo ""
    echo "Default path to acoustic source conf file: conf/acoustic_${Source}.conf"
    echo "Default path to acoustic voice conf file: conf/acoustic_${Voice}.conf"
    echo "################################"
    exit 1
fi

acoustic_conf_file=$1

### Step 5: train acoustic model ###
echo "Step 5:"
echo "training acoustic model..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $acoustic_conf_file


