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

### define variables here
frontend=${MerlinDir}/misc/scripts/frontend
testDir=experiments/${Voice}/test_synthesis

### create a list of files from text directory
ls ${testDir}/txt/ > ${testDir}/test_id_list.scp 
sed -i 's/\.txt//g' ${testDir}/test_id_list.scp

### create a scheme file from text directory
echo "creating a scheme file from text directory"
python ${frontend}/utils/genScmFile.py \
                            ${testDir}/txt/ \
                            ${testDir}/new_test_sentences.scm \
                            ${testDir}/prompt-utt 

### generate utt from scheme file
echo "generating utts from scheme file"
${FESTDIR}/bin/festival -b ${testDir}/new_test_sentences.scm 

### convert festival utt to lab
echo "converting festival utts to labels..."
${frontend}/festival_utt_to_lab/make_labels \
                            ${testDir}/prompt-lab/ \
                            ${testDir}/prompt-utt/ \
                            ${FESTDIR}/examples/dumpfeats \
                            ${frontend}/festival_utt_to_lab/

### normalize lab for merlin with options: state_align or phone_align
echo "normalizing label files for merlin..."
python ${frontend}/utils/normalize_lab_for_merlin.py \
                            ${testDir}/prompt-lab/full/ \
                            ${testDir}/prompt-lab/ \
                            ${Labels} \
                            ${testDir}/test_id_list.scp

### remove any un-necessary files
rm -rf ${testDir}/prompt-lab/{full,mono,tmp}

echo "Labels are ready in: experiments/${Voice}/test_synthesis/prompt-lab !!"
