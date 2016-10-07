#!/bin/bash -e

if test "$#" -ne 0; then
    echo "Usage: ./run_aligner.sh"
    exit 1
fi

#################################################################
##### Create training labels for merlin with festvox tools ######
#################################################################

### Download data -- slt cmuarctic.data

audio_data=slt_wav

audio_data_url=http://104.131.174.95/${audio_data}.zip
text_data_url=http://festvox.org/cmu_arctic/cmuarctic.data

if [[ ! -f ${audio_data}.zip ]]; then
    echo "downloading data....."
    rm -f ${audio_data}.zip
    if hash curl 2>/dev/null; then
        curl -O $audio_data_url
        curl -O $text_data_url
    elif hash wget 2>/dev/null; then
        wget $audio_data_url
        wget $text_data_url
    else
        echo "please download the audio data from $audio_data_url"
        echo "please download the text data from $text_data_url"
        exit 1
    fi
    do_unzip=true
fi

if [[ ! -d ${audio_data} ]] || [[ -n "$do_unzip" ]]; then
    unzip -q ${audio_data}.zip
fi

### do forced alignment using ehmm in clustergen

FESTVOXDIR=${MerlinDir}/tools/festvox

mkdir cmu_us_slt_arctic
cd cmu_us_slt_arctic

$FESTVOXDIR/src/clustergen/setup_cg cmu us slt_arctic 

cp ../cmuarctic.data etc/txt.done.data
cp ../slt_wav/*.wav wav/

./bin/do_build build_prompts 
./bin/do_build label 

cd ../

### convert festival utts to lab

frontend=${MerlinDir}/misc/scripts/frontend 
FESTDIR=${MerlinDir}/tools/festival

head cmuarctic.data | cut -d " " -f 2 > file_id_list.scp

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

