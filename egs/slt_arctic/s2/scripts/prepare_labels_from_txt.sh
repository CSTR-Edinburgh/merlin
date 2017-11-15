#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/prepare_labels_from_txt.sh conf/global_settings.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $1
fi

### tools required
FESTDIR=${MerlinDir}/tools/festival
if [ ! -d "${FESTDIR}" ]; then
    echo "Please configure festival path in scripts/prepare_labels_from_txt.sh !!"
    exit 1
fi

### define few variables here
frontend=${MerlinDir}/misc/scripts/frontend
testDir=experiments/${Voice}/test_synthesis

txt_dir=${testDir}/txt
txt_file=${testDir}/utts.data

### create a scheme file with options from: txt directory or utts.data file

if [ -d "${txt_dir}" ]; then
    if [ ! "$(ls -A ${txt_dir})" ]; then
        echo "Please place your new test sentences (files) in: ${txt_dir} !!"
        exit 1
    else
        in_txt=${txt_dir}
    fi
elif [ -f "${txt_file}" ]; then
    in_txt=${txt_file}
else
    echo "Please give input: either 1 or 2"
    echo "1. ${txt_dir}  -- a text directory containing text files"
    echo "2. ${txt_file} -- a single text file with each sentence in a new line in festival format"
    exit 1
fi

python ${frontend}/utils/genScmFile.py \
                            ${in_txt} \
                            ${testDir}/prompt-utt \
                            ${testDir}/new_test_sentences.scm \
                            ${testDir}/test_id_list.scp 

### generate utt from scheme file
echo "generating utts from scheme file"
${FESTDIR}/bin/festival -b ${testDir}/new_test_sentences.scm 

### convert festival utt to lab
echo "converting festival utts to labels..."
${frontend}/festival_utt_to_lab/make_labels \
                            ${testDir}/prompt-lab \
                            ${testDir}/prompt-utt \
                            ${FESTDIR}/examples/dumpfeats \
                            ${frontend}/festival_utt_to_lab

### normalize lab for merlin with options: state_align or phone_align
echo "normalizing label files for merlin..."
python ${frontend}/utils/normalize_lab_for_merlin.py \
                            ${testDir}/prompt-lab/full \
                            ${testDir}/prompt-lab \
                            ${Labels} \
                            ${testDir}/test_id_list.scp

### remove any un-necessary files
rm -rf ${testDir}/prompt-lab/{full,mono,tmp}

echo "Labels are ready in: ${testDir}/prompt-lab !!"
