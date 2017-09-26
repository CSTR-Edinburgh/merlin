#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage: "
    echo "./05_run_hybrid_voice.sh <path_to_text_dir> <path_to_wav_dir>"
    echo ""
    echo "default path to text dir: experiments/${Voice}/test_synthesis/txt"
    echo "default path to text dir: experiments/${Voice}/test_synthesis/wav"
    echo "################################"
    exit 1
fi

### arguments
txt_dir=$1
wav_dir=$2

### tools required
if [ ! -d "${FESTDIR}" ]; then
    echo "Please configure festival path in $global_config_file !!"
    exit 1
fi

### define few variables here
script_dir=${MerlinDir}/misc/scripts/hybrid_voice

out_dir=$(dirname $txt_dir)

file_id_scp=test_id_list.scp
scheme_file=new_test_sentences.scm

### generate a scheme file 
python ${script_dir}/genScmFile.py \
                            ${txt_dir} \
                            ${wav_dir} \
                            ${out_dir}/$scheme_file \
                            ${out_dir}/$file_id_scp 

### generate wav
echo "generating wav from scheme file"
${FESTDIR}/bin/festival -b ${out_dir}/$scheme_file


