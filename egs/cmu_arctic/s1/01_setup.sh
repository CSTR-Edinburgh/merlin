#!/bin/bash

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./01_setup.sh <voice_name>"
    echo ""
    echo "Give a voice name eg., slt"
    echo "Available speakers are bdl, slt, jmk"
    echo "################################"
    exit 1
fi

IFS='_' read -ra voice_name <<< "$1"
spk="${voice_name[0]}"
voice_name="$1"
echo "Speaker is ${spk}."

# Define a regex to select only parts of the database for the demo version.
if [[ "${voice_name}" == *"demo"* ]]; then
    corpus_select_rgx="arctic_a00[0-5][0-9]" # Use only the first 59 utterances in demo.
else
    corpus_select_rgx="*" # Use all utterances here.
fi

### Step 1: setup directories and the training data files ###
echo "Step 1:"

current_working_dir=$(pwd)
merlin_dir=$(dirname $(dirname $(dirname $current_working_dir)))
experiments_dir=${current_working_dir}/experiments
data_dir=${current_working_dir}/database

voice_name=$1
voice_dir=${experiments_dir}/${voice_name}

acoustic_dir=${voice_dir}/acoustic_model
duration_dir=${voice_dir}/duration_model
synthesis_dir=${voice_dir}/test_synthesis

mkdir -p ${data_dir}
mkdir -p ${experiments_dir}
mkdir -p ${voice_dir}
mkdir -p ${acoustic_dir}
mkdir -p ${duration_dir}
mkdir -p ${synthesis_dir}
mkdir -p ${acoustic_dir}/data
mkdir -p ${duration_dir}/data
mkdir -p ${synthesis_dir}/txt


audio_dir=database/wav
rawaudio_dir=database/rawaudio
txt_dir=database/txt
label_dir=database/labels

# URL of arctic DB.
arch=cmu_us_${spk}_arctic-WAVEGG.tar.bz2
url=http://festvox.org/cmu_arctic/cmu_arctic/orig/$arch
laburl=http://festvox.org/cmu_arctic/cmuarctic.data
# Download the data.
if [ ! -e $rawaudio_dir/$arch ]; then
    mkdir -p $rawaudio_dir
    cd $rawaudio_dir
    wget $url
    tar xjf $arch
    cd ../../
fi
rm -rf $txt_dir
if [ ! -e $txt_dir ]; then
    mkdir -p $txt_dir
    cd $txt_dir
    wget $laburl
    mv cmuarctic.data utts.data # For consistency.
    cd ../../
fi

# Collect utterances ids of necessary audio files.
utts=($(find "${rawaudio_dir}"/cmu_us_${spk}_arctic/orig/${corpus_select_rgx}.wav -exec basename {} .wav \;))
# Remove duplicates.
utts=($(printf "%s\n" "${utts[@]}" | sort -u))

# Audios have to be removed because demo/full could have been changed.
rm -rf $audio_dir
# Leave this check for fast testing, when $audio_dir does not have to be removed.
if [ ! -e $audio_dir ]; then
    mkdir -p $audio_dir
    # Collect necessary audio files.
    for utt in "${utts[@]}"; do
        # Sample down to 16k mono, script 03_prepare_acoustic_features cannot handle stereo.
        sox "${rawaudio_dir}"/cmu_us_${spk}_arctic/orig/${utt}.wav $audio_dir/${utt}.wav remix 1 rate -v -s -a 16000 dither -s
    done
fi

# Get labels, combine the selected utterances to a regex pattern.
export utts_pat=$(echo ${utts[@]}|tr " " "|")
# Select those labels of utts.data which belong to the selected utterances.
cat ${txt_dir}/utts.data | grep -wE "${utts_pat}" >| ${txt_dir}/utts_selected.data
# Turn every line of utts_selected.data into a txt file using the utterance id as file name.
awk -F' ' -v outDir=${txt_dir} '{print substr($0,2+length($2)+2,length($0)) > outDir"/"$2".txt"}' ${txt_dir}/utts_selected.data
# Remove unnecessary files.
rm ${txt_dir}/utts.data
rm ${txt_dir}/utts_selected.data

rm -rf $label_dir

### create some test files ###
echo "Hello world." > ${synthesis_dir}/txt/test_001.txt
echo "Hi, this is a demo voice from Merlin." > ${synthesis_dir}/txt/test_002.txt
echo "Hope you guys enjoy free open-source voices from Merlin." > ${synthesis_dir}/txt/test_003.txt
printf "test_001\ntest_002\ntest_003" > ${synthesis_dir}/test_id_list.scp

global_config_file=conf/global_settings.cfg

### default settings ###
echo "######################################" > $global_config_file
echo "############# PATHS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "MerlinDir=${merlin_dir}" >>  $global_config_file
echo "WorkDir=${current_working_dir}" >>  $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# PARAMS #################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "Voice=${voice_name}" >> $global_config_file
echo "Labels=state_align" >> $global_config_file
echo "QuestionFile=questions-radio_dnn_416.hed" >> $global_config_file
echo "Vocoder=WORLD" >> $global_config_file
echo "SamplingFreq=16000" >> $global_config_file
echo "SilencePhone='sil'" >> $global_config_file
echo "FileIDList=file_id_list.scp" >> $global_config_file
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "######### No. of files ###############" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

# Select 59 examples in the demo.
if [[ "${voice_name}" == *"demo"* ]]; then
    echo "Train=49" >> $global_config_file
    echo "Valid=5" >> $global_config_file 
    echo "Test=5" >> $global_config_file
else # In the full version 5% of the utterances are used for validation and test set each.
    num_files=$(ls -1 $audio_dir | wc -l)
    num_dev_set=$(awk "BEGIN { pc=${num_files}*0.05; print(int(pc)) }")
    num_train_set=$(($num_files-2*$num_dev_set))
    echo "Train=$num_train_set" >> $global_config_file
    echo "Valid=$num_dev_set" >> $global_config_file 
    echo "Test=$num_dev_set" >> $global_config_file
fi
echo "" >> $global_config_file

echo "######################################" >> $global_config_file
echo "############# TOOLS ##################" >> $global_config_file
echo "######################################" >> $global_config_file
echo "" >> $global_config_file

echo "ESTDIR=${merlin_dir}/tools/speech_tools" >> $global_config_file
echo "FESTDIR=${merlin_dir}/tools/festival" >> $global_config_file
echo "FESTVOXDIR=${merlin_dir}/tools/festvox" >> $global_config_file
echo "" >> $global_config_file
echo "HTKDIR=${merlin_dir}/tools/bin/htk" >> $global_config_file
echo "" >> $global_config_file

echo "Merlin default voice settings configured in \"$global_config_file\""
echo "Modify these params as per your data..."
echo "eg., sampling frequency, no. of train files etc.,"
echo "setup done...!"

