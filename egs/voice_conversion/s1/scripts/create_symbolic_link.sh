#!/bin/bash -e

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 0; then
    echo "################################"
    echo "Usage:"
    echo "$0"
    echo "################################"
    exit 1
fi

# get cmp dimension based on sampling rate
if [ "$Vocoder" == "STRAIGHT" ]
then
    cmp_dim=259
elif [ "$Vocoder" == "WORLD" ]
then
    if [ "$SamplingFreq" == "16000" ]
    then
        cmp_dim=187
    elif [ "$SamplingFreq" == "48000" ]
    then
        cmp_dim=199
    fi
else
    echo "This vocoder ($Vocoder) is not supported as of now...please configure yourself!!"
fi

### Step 6: create symbolic link for source input ###
echo "Step 6:"
echo "creating symbolic link..."
cmd="ln -s ${WorkDir}/experiments/${Source}/acoustic_model/inter_module/nn_norm_mgc_lf0_vuv_bap_${cmp_dim}/ experiments/${Voice}/acoustic_model/inter_module/nn_no_silence_lab_norm_${cmp_dim}"
echo $cmd
$cmd
echo "done...!"
