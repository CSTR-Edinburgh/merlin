#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage: "
    echo "./07_run_merlin.sh <path_to_text_dir> <path_to_test_dur_conf_file> <path_to_test_synth_conf_file>"
    echo ""
#    echo "default path to text dir: experiments/${Voice}/test_synthesis/txt"
    echo "default path to test duration conf file: conf/test_dur_synth_${Voice}.conf"
    echo "default path to test synthesis conf file: conf/test_synth_${Voice}.conf"
    echo "################################"
    exit 1
fi

# inp_txt=$1
# test_dur_config_file=$2
# test_synth_config_file=$3

test_dur_config_file=$1
test_synth_config_file=$2

### Step 7: synthesize speech from text ###
echo "Step 7:" 
echo "synthesizing speech from text..."

# echo "preparing full-contextual labels using Festival frontend..."
# lab_dir=$(dirname $inp_txt)
# ./scripts/prepare_labels_from_txt.sh $inp_txt $lab_dir $global_config_file

echo "synthesizing durations..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_dur_config_file

echo "synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_synth_config_file

echo "deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"
echo "All successfull!! Your demo voice is ready :)"

