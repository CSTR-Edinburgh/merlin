#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage: "
    echo "./06_synthesise_waveforms.sh <path_to_dur_synth_conf_file> <path_to_acoustic_synth_conf_file>"
    echo ""
    echo "Default path to test duration conf file: conf/dur_synth_${Voice}.conf"
    echo "Default path to test synthesis conf file: conf/acous_synth_${Voice}.conf"
    echo "################################"
    exit 1
fi

dur_synth_config_file=$1
acous_synth_config_file=$2


### Step 6: synthesize speech ###
echo "Step 6:"

echo "synthesizing durations..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $dur_synth_config_file

echo "synthesizing speech..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $acous_synth_config_file

echo "deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"
echo "All successfull!! Your demo voice is ready :)"

