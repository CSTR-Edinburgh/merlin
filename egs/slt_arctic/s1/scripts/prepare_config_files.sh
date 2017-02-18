#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/prepare_config_files.sh conf/global_settings.cfg"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "Global config file doesn't exist"
    exit 1
else
    source $1
fi

SED=sed
if [[ "$OSTYPE" == "darwin"* ]]; then
  which gsed > /dev/null
  if [[ "$?" != 0 ]]; then
    echo "You need to install GNU sed with 'brew install gnu-sed' on osX"
    exit 1
  fi
  SED=gsed
fi

#########################################
######## duration config file ###########
#########################################

duration_config_file=conf/duration_${Voice}.conf

# Start with a general recipe...
cp -f $MerlinDir/misc/recipes/duration_demo.conf $duration_config_file

# ... and modify it:

$SED -i s#'Merlin:.*'#'Merlin: '$MerlinDir# $duration_config_file
$SED -i s#'TOPLEVEL:.*'#'TOPLEVEL: '${WorkDir}# $duration_config_file
$SED -i s#'work:.*'#'work: %(TOPLEVEL)s/experiments/'${Voice}'/duration_model'# $duration_config_file

$SED -i s#'file_id_list:.*'#'file_id_list: %(data)s/'${FileIDList}# $duration_config_file

# [Labels]
$SED -i s#'label_type:.*'#'label_type: '${Labels}# $duration_config_file
$SED -i s#'label_align:.*'#'label_align: %(TOPLEVEL)s/experiments/'${Voice}'/duration_model/data/label_'${Labels}# $duration_config_file
$SED -i s#'question_file_name:.*'#'question_file_name: %(Merlin)s/misc/questions/'${QuestionFile}# $duration_config_file


# [Outputs]
if [ "$Labels" == "state_align" ]
then
    $SED -i s#'dur\s*:.*'#'dur: 5'# $duration_config_file
elif [ "$Labels" == "phone_align" ]
then
    $SED -i s#'dur\s*:.*'#'dur: 1'# $duration_config_file
else
    echo "These labels ($Lables) are not supported as of now...please use state_align or phone_align!!"
fi


# [Architecture]

if [ "$Voice" == "slt_arctic_demo" ]
then
    $SED -i s#'hidden_layer_size\s*:.*'#'hidden_layer_size: [512, 512, 512, 512]'# $duration_config_file
    $SED -i s#'hidden_layer_type\s*:.*'#'hidden_layer_type: ['\''TANH'\'', '\''TANH'\'', '\''TANH'\'', '\''TANH'\'']'# $duration_config_file
fi


# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $duration_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $duration_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $duration_config_file

echo "Duration configuration settings stored in $duration_config_file"




#########################################
######## acoustic config file ###########
#########################################

acoustic_config_file=conf/acoustic_${Voice}.conf

# Start with a general recipe...
cp -f $MerlinDir/misc/recipes/acoustic_demo.conf $acoustic_config_file

# ... and modify it:

$SED -i s#'Merlin:.*'#'Merlin: '$MerlinDir# $acoustic_config_file
$SED -i s#'TOPLEVEL:.*'#'TOPLEVEL: '${WorkDir}# $acoustic_config_file
$SED -i s#'work:.*'#'work: %(TOPLEVEL)s/experiments/'${Voice}'/acoustic_model'# $acoustic_config_file

$SED -i s#'file_id_list:.*'#'file_id_list: %(data)s/'${FileIDList}# $acoustic_config_file


# [Labels]

$SED -i s#'label_type:.*'#'label_type: '${Labels}# $acoustic_config_file
$SED -i s#'label_align:.*'#'label_align: %(TOPLEVEL)s/experiments/'${Voice}'/acoustic_model/data/label_'${Labels}# $acoustic_config_file
$SED -i s#'question_file_name:.*'#'question_file_name: %(Merlin)s/misc/questions/'${QuestionFile}# $acoustic_config_file

if [ "$Labels" == "state_align" ]
then
    $SED -i s#'subphone_feats:.*'#'subphone_feats: full'# $acoustic_config_file
elif [ "$Labels" == "phone_align" ]
then
    $SED -i s#'subphone_feats:.*'#'subphone_feats: coarse_coding'# $acoustic_config_file
else
    echo "These labels ($Labels) are not supported as of now...please use state_align or phone_align!!"
fi


# [Outputs]

$SED -i s#'mgc\s*:.*'#'mgc: 60'# $acoustic_config_file
$SED -i s#'dmgc\s*:.*'#'dmgc: 180'# $acoustic_config_file

if [ "$Vocoder" == "STRAIGHT" ]
then
    $SED -i s#'bap\s*:.*'#'bap: 25'# $acoustic_config_file
    $SED -i s#'dbap\s*:.*'#'dbap: 75'# $acoustic_config_file

elif [ "$Vocoder" == "WORLD" ]
then
    if [ "$SamplingFreq" == "16000" ]
    then
        $SED -i s#'bap\s*:.*'#'bap: 1'# $acoustic_config_file
        $SED -i s#'dbap\s*:.*'#'dbap: 3'# $acoustic_config_file
    elif [ "$SamplingFreq" == "48000" ]
    then
        $SED -i s#'bap\s*:.*'#'bap: 5'# $acoustic_config_file
        $SED -i s#'dbap\s*:.*'#'dbap: 15'# $acoustic_config_file
    fi
else
    echo "This vocoder ($Vocoder) is not supported as of now...please configure yourself!!"
fi

$SED -i s#'lf0\s*:.*'#'lf0: 1'# $acoustic_config_file
$SED -i s#'dlf0\s*:.*'#'dlf0: 3'# $acoustic_config_file


# [Waveform]
$SED -i s#'vocoder_type\s*:.*'#'vocoder_type: '${Vocoder}# $acoustic_config_file

$SED -i s#'samplerate\s*:.*'#'samplerate: '${SamplingFreq}# $acoustic_config_file
if [ "$SamplingFreq" == "16000" ]
then
    $SED -i s#'framelength\s*:.*'#'framelength: 1024'# $acoustic_config_file
    $SED -i s#'minimum_phase_order\s*:.*'#'minimum_phase_order: 511'# $acoustic_config_file
    $SED -i s#'fw_alpha\s*:.*'#'fw_alpha: 0.58'# $acoustic_config_file

elif [ "$SamplingFreq" == "48000" ]
then
    if [ "$Vocoder" == "WORLD" ]
    then
        $SED -i s#'framelength\s*:.*'#'framelength: 2048'# $acoustic_config_file
        $SED -i s#'minimum_phase_order\s*:.*'#'minimum_phase_order: 1023'# $acoustic_config_file
    else
        $SED -i s#'framelength\s*:.*'#'framelength: 4096'# $acoustic_config_file
        $SED -i s#'minimum_phase_order\s*:.*'#'minimum_phase_order: 2047'# $acoustic_config_file
    fi
    $SED -i s#'fw_alpha\s*:.*'#'fw_alpha: 0.77'# $acoustic_config_file
else
    echo "This sampling frequency ($SamplingFreq) never tested before...please configure yourself!!"
fi

# [Architecture]
if [ "$Voice" == "slt_arctic_demo" ]
then
    $SED -i s#'hidden_layer_size\s*:.*'#'hidden_layer_size: [512, 512, 512, 512]'# $acoustic_config_file
    $SED -i s#'hidden_layer_type\s*:.*'#'hidden_layer_type: ['\''TANH'\'', '\''TANH'\'', '\''TANH'\'', '\''TANH'\'']'# $acoustic_config_file
fi


# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $acoustic_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $acoustic_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $acoustic_config_file


echo "Acoustic configuration settings stored in $acoustic_config_file"
