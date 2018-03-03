#!/bin/bash -e

if test "$#" -ne 4; then
    echo "Usage: ./run_phone_aligner.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir> <path_to_global_conf_file>"
    exit 1
fi

### Arguments
wav_dir=$1
inp_txt=$2
lab_dir=$3
global_config_file=$4

### Use paths from global config file
source $global_config_file

### frontend scripts
frontend=${MerlinDir}/misc/scripts/frontend

#################################################################
##### Create training labels for merlin with festvox tools ######
#################################################################

### tools required

if [[ ! -d "${ESTDIR}" ]] || [[ ! -d "${FESTDIR}" ]] || [[ ! -d "${FESTVOXDIR}" ]]; then
    echo "Please configure paths to speech_tools, festival and festvox in config.cfg !!"
    exit 1
fi

### do forced alignment using ehmm in clustergen setup
mkdir -p $lab_dir
cd $lab_dir
mkdir cmu_us_${Voice}
cd cmu_us_${Voice}

$FESTVOXDIR/src/clustergen/setup_cg cmu us ${Voice} 

txt_file=${WorkDir}/${inp_txt}
txt_dir=${WorkDir}/${inp_txt}

if [ -f "${txt_file}" ]; then
    cp ${txt_file} etc/txt.done.data
elif [ -d "${txt_dir}" ]; then
    python ${frontend}/utils/prepare_txt_done_data_file.py ${txt_dir} etc/txt.done.data
else
    echo "Please check ${inp_txt} !!"
    exit 1
fi

cp $WorkDir/$wav_dir/*.wav wav/

./bin/do_build build_prompts 
./bin/do_build label
./bin/do_build build_utts

cd ../

### convert festival utts to lab

cat cmu_us_${Voice}/etc/txt.done.data | cut -d " " -f 2 > file_id_list.scp

echo "converting festival utts to labels..."
${frontend}/festival_utt_to_lab/make_labels \
                        full-context-labels \
                        cmu_us_${Voice}/festival/utts \
                        ${FESTDIR}/examples/dumpfeats \
                        ${frontend}/festival_utt_to_lab 

echo "normalizing label files for merlin..."
python ${frontend}/utils/normalize_lab_for_merlin.py \
                        full-context-labels/full \
                        label_phone_align \
                        phone_align \
                        file_id_list.scp

### return to working directory
cd ${WorkDir}

phone_labels=$lab_dir/label_phone_align

if [ ! "$(ls -A ${phone_labels})" ]; then
    echo "Force-alignment unsucessful!!"
else
    echo "You should have your labels ready in: $phone_labels !!"
fi


