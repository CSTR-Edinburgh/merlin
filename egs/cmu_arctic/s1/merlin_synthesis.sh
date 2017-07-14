#!/bin/bash

source cmd.sh

if test "$#" -ne 0; then
    echo "Usage: ./merlin_synthesis.sh"
    exit 1
fi

global_config_file=conf/global_settings.cfg

if [ ! -f  $global_config_file ]; then
    echo "Please run steps from 1-5..."
    exit 1
else
    source $global_config_file
fi

### define few variables here
testDir=experiments/${Voice}/test_synthesis

txt_dir=${testDir}/txt

### Synthesize speech   ###
echo "Synthesizing speech..."
./07_run_merlin.sh $txt_dir conf/test_dur_synth_${Voice}.conf conf/test_synth_${Voice}.conf
#./scripts/${cuda_short_cmd} "experiments/${Voice}/test_synthesis/_synth_dur.log" "./scripts/submit.sh" "${MerlinDir}/src/run_merlin.py" "conf/test_dur_synth_${Voice}.conf"
#./scripts/${cuda_short_cmd} "experiments/${Voice}/test_synthesis/_synth_speech.log" "./scripts/submit.sh" "${MerlinDir}/src/run_merlin.py" "conf/test_synth_${Voice}.conf"

#echo "deleting intermediate synthesis files..."
#./scripts/remove_intermediate_files.sh $global_config_file

# echo "synthesized audio files are in: experiments/${Voice}/test_synthesis/wav"

