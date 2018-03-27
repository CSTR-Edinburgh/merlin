#!/bin/bash

if test "$#" -ne 1; then
    echo "################################"
    echo "Usage:"
    echo "./01_setup.sh <voice_name>"
    echo ""
    echo "Give a voice name: roger_demo or roger_full"
    echo "   Demo uses theherald1 (281 utterances, 42.8 minutes)"
    echo "   Full uses carroll, arcitc and theherald1-3 (4871 utterances, ~8h)"
    echo "################################"
    exit 1
fi

if [ ! -d "${ROGER_DB}" ]; then
    echo "ERROR: Variable ROGER_DB must be set to the roger database."
    echo "       Use: export ROGER_DB=path/to/db/"
    exit 1
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
txt_dir=database/txt
label_dir=database/labels

# Select the utterance list(s) to be used for training.
if [[ "$voice_name" == *"demo"* ]]
then
    # The demo version only uses theherald1 (281 utterances, 42.8 minutes)
    uttLists=("theherald1")
elif [[ "$voice_name" == *"full"* ]]
then
    # The full version uses all utterance lists with meaningful utterances.
    # Using: carroll, arcitc, theherald1-3 (4871 utterances, ~8h).
    uttLists=("carroll" "arctic" "theherald") # Can be any of carroll, unilex, address, spelling, arcitc, emphasis, theherald, theherald1, theherald2, theherald3, all_new, total.
else
    echo "Undefined voice name ($voice_name)...please use roger_demo or roger_full !!"
    exit 1
fi

# Collect utterance ids of necessary audio files.
utts=()
for uttList in "${uttLists[@]}"; do
    mapfile -t -O ${#utts[@]} utts < $ROGER_DB/stp/$uttList # -t remove trailing newline, -O start index to add entries.
done
# Remove duplicates.
utts=($(printf "%s\n" "${utts[@]}" | sort -u))

# Audios have to be removed because utterance list selection could have been changed.
rm -rf $audio_dir
# Leave this check for fast testing, when $audio_dir does not have to be removed.
if [ ! -e $audio_dir ]; then
    mkdir -p $audio_dir
    # Collect necessary audio files.
    for utt in "${utts[@]}"; do
        # cp $ROGER_DB/wav/${utt:0:7}/${utt}.wav $audio_dir/${utt}.wav
        ln -sf $ROGER_DB/wav/${utt:0:7}/${utt}.wav $audio_dir/${utt}.wav
    done
fi

# Labels have to be removed because utterance list selection could have been changed.
rm -rf $txt_dir
# Leave this check for fast testing, when $txt_dir does not have to be removed.
if [ ! -e $txt_dir ]; then
    mkdir -p $txt_dir
    # The utts.data file contains all labels.
    cp ${ROGER_DB}/utts.data ${txt_dir}/utts.data
    # Combine the selected utterances to a regex pattern.
    utts_pat=$(echo ${utts[@]}|tr " " "|")
    # Select those labes of utts.data which belong to the selected utterances.
    cat ${txt_dir}/utts.data | grep -wE "${utts_pat}" >| ${txt_dir}/utts_selected.data
    # Turn every line of utts.data into a txt file using the utterance id as file name.
    awk -F' ' -v outDir=${txt_dir} '{print substr($0,length($1)+2,length($0)) > outDir"/"substr($1,2,length($1)-1)".txt"}' ${txt_dir}/utts_selected.data
    # Remove unnecessary files.
    rm ${txt_dir}/utts.data
    rm ${txt_dir}/utts_selected.data
fi

# Clear the labels directory.
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

# Automatically select 5% of the data for validation and test set.
num_files=$(ls -1 $audio_dir | wc -l)
num_dev_set=$(awk "BEGIN { pc=${num_files}*0.05; print(int(pc)) }")
num_train_set=$(($num_files-2*$num_dev_set))
echo "Train=$num_train_set" >> $global_config_file 
echo "Valid=$num_dev_set" >> $global_config_file 
echo "Test=$num_dev_set" >> $global_config_file 
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

