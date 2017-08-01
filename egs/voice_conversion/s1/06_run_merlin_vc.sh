#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 3; then
    echo "################################"
    echo "Usage: "
    echo "$0 <path_to_src_wav_dir> <path_to_test_source_conf_file> <path_to_test_synth_conf_file>"
    echo ""
    echo "default path to source wav dir: experiments/${Voice}/test_synthesis/${Source}"
    echo "default path to test source conf file: conf/test_synth_${Source}.conf"
    echo "default path to test synthesis conf file: conf/test_synth_${Voice}.conf"
    echo "################################"
    exit 1
fi

wav_dir=$1
test_source_config_file=$2
test_synth_config_file=$3

### Step 6: transform source voice to target voice ###
echo "Step 6:" 
echo "transforming source voice to target..."

echo "extracting source features using "${Vocoder}" vocoder..."
feat_dir=experiments/${Source}/acoustic_model/data
python ${MerlinDir}/misc/scripts/vocoder/${Vocoder,,}/extract_features_for_merlin.py ${MerlinDir} ${wav_dir} ${feat_dir} $SamplingFreq 

echo "preparing acoustic features for source voice..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_source_config_file

echo "transforming source voice to target voice..."
./scripts/submit.sh ${MerlinDir}/src/run_merlin.py $test_synth_config_file

echo "deleting intermediate synthesis files..."
./scripts/remove_intermediate_files.sh $global_config_file

echo "transformed audio files are in: experiments/${Voice}/test_synthesis/${Target}"

