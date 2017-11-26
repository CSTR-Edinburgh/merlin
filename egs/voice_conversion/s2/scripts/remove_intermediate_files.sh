#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/remove_intermediate_files.sh conf/global_settings.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $1
fi

###################################################
######## remove intermediate synth files ##########
###################################################

current_working_dir=$(pwd)

synthesis_dir=${WorkDir}/experiments/${Voice}/test_synthesis
gen_wav_dir=${synthesis_dir}/${Target}

shopt -s extglob

if [ -d "$gen_wav_dir" ]; then
    cd ${gen_wav_dir}
    rm -f weight
    rm -f *.!(wav)
fi

cd ${current_working_dir}
