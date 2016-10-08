#!/bin/bash -e

if test "$#" -ne 1; then
    echo "Usage: ./run_aligner.sh config.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Config file doesn't exist"
    exit 1
else
    source $1
fi

#################################################################
##### Create training labels for merlin with festvox tools ######
#################################################################

### tools required

if [[ ! -d "${ESTDIR}" ]] || [[ ! -d "${FESTDIR}" ]] || [[ ! -d "${FESTVOXDIR}" ]]; then
    echo "Please configure paths to speech_tools, festival and festvox in config.cfg !!"
    exit 1
fi

### do forced alignment using ehmm in clustergen setup

mkdir cmu_us_slt_arctic
cd cmu_us_slt_arctic

$FESTVOXDIR/src/clustergen/setup_cg cmu us slt_arctic 

cp ../cmuarctic.data etc/txt.done.data
cp ../slt_wav/*.wav wav/

./bin/do_build build_prompts 
./bin/do_build label
./bin/do_build build_utts

cd ../

### convert festival utts to lab

cat cmuarctic.data | cut -d " " -f 2 > file_id_list.scp

echo "converting festival utts to labels..."
${frontend}/festival_utt_to_lab/make_labels \
                        full-context-labels \
                        cmu_us_slt_arctic/festival/utts \
                        ${FESTDIR}/examples/dumpfeats \
                        ${frontend}/festival_utt_to_lab 

echo "normalizing label files for merlin..."
python ${frontend}/utils/normalize_lab_for_merlin.py \
                        full-context-labels/full \
                        label_phone_align \
                        phone_align \
                        file_id_list.scp 

echo "You should have your labels ready in: label_phone_align !!"

