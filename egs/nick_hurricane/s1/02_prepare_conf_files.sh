#!/bin/bash -e

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./02_prepare_conf_files.sh <path_to_global_conf_file>"
    echo ""
    echo "default path to global conf file: conf/global_settings.cfg"
    echo "Config files will be prepared based on settings in global conf file"
    echo "################################"
    exit 1
fi

global_config_file=$1


### Step 2: prepare config files for acoustic, duration models and for synthesis ###
echo "Step 2:"

echo "preparing config files for acoustic, duration models..."
./scripts/prepare_config_files.sh $global_config_file

echo "preparing config files for synthesis..."
./scripts/prepare_config_files_for_synthesis.sh $global_config_file

