#!/bin/bash

if test "$#" -ne 3; then
    echo "bash scripts/test_nan.sh <input_dir> <file_list> <ext: .cmp/.lab/.lf0/.mgc/.bap>"
    exit 1
fi

global_config_file="conf/global_settings.cfg"
if [ ! -f $global_config_file ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $global_config_file
fi

x2x=${MerlinDir}/tools/bin/SPTK-3.9/x2x

ext=$3

IFS=''
while read sentence 
do 
    nlines=`$x2x +fa $1/$sentence$ext | grep "nan" | wc -l` 
    z=0
    if test $nlines -gt $z
    then
        echo $sentence
        echo $nlines
    fi
done < $2
