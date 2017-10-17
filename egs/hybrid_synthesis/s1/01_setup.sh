#!/bin/bash

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "$0 <path_to_global_conf_file>"
    echo ""
    echo "default path to global conf file: conf/global_settings.cfg"
    echo "Modify the params in conf file as per your data..."
    echo "################################"
    exit 1
fi

#pass arguments
global_config_file=$1

if [ ! -f $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

experiments_dir=${WORKDIR}/experiments
voice_dir=${experiments_dir}/${Voice}
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${DATADIR}
mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${synthesis_dir}
mkdir -p ${synthesis_dir}/txt

### create some test files ###
echo "Hello world." > ${synthesis_dir}/txt/test_001.txt
echo "Hi, this is a demo voice from Merlin." > ${synthesis_dir}/txt/test_002.txt
echo "Hope you guys enjoy free open-source voices from Merlin." > ${synthesis_dir}/txt/test_003.txt
printf "test_001\ntest_002\ntest_003" > ${synthesis_dir}/test_id_list.scp

echo "Step 1:"
echo "Merlin default voice settings configured in ${global_config_file}"
echo "setup done...!"

