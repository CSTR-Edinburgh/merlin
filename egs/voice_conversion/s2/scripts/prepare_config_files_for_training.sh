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
######## acoustic config file ###########
#########################################

acoustic_config_file=conf/acoustic_${Source}.conf

# Start with a general recipe...
cp -f conf/general/acoustic_demo.conf $acoustic_config_file

# ... and modify it:

$SED -i s#'Merlin:.*'#'Merlin: '$MerlinDir# $acoustic_config_file
$SED -i s#'TOPLEVEL:.*'#'TOPLEVEL: '${WorkDir}# $acoustic_config_file
$SED -i s#'work:.*'#'work: %(TOPLEVEL)s/experiments/'${Source}'/acoustic_model'# $acoustic_config_file

$SED -i s#'file_id_list:.*'#'file_id_list: %(data)s/'${FileIDList}# $acoustic_config_file


# [Architecture]
if [[ "$Voice" == *"demo"* ]]
then
    $SED -i s#'hidden_layer_size\s*:.*'#'hidden_layer_size: [512, 512, 512, 512]'# $acoustic_config_file
    $SED -i s#'hidden_layer_type\s*:.*'#'hidden_layer_type: ['\''TANH'\'', '\''TANH'\'', '\''TANH'\'', '\''TANH'\'']'# $acoustic_config_file
fi


# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $acoustic_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $acoustic_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $acoustic_config_file

# [Processes]

$SED -i s#'TRAINDNN\s*:.*'#'TRAINDNN: False'# $acoustic_config_file
$SED -i s#'DNNGEN\s*:.*'#'DNNGEN: False'# $acoustic_config_file
$SED -i s#'GENWAV\s*:.*'#'GENWAV: False'# $acoustic_config_file
$SED -i s#'CALMCD\s*:.*'#'CALMCD: False'# $acoustic_config_file


echo "Acoustic configuration settings stored in $acoustic_config_file"

#########################################
######## acoustic config file ###########
#########################################

acoustic_config_file=conf/acoustic_${Voice}.conf

# Start with a general recipe...
cp -f conf/general/acoustic_demo.conf $acoustic_config_file

# ... and modify it:

$SED -i s#'Merlin:.*'#'Merlin: '$MerlinDir# $acoustic_config_file
$SED -i s#'TOPLEVEL:.*'#'TOPLEVEL: '${WorkDir}# $acoustic_config_file
$SED -i s#'work:.*'#'work: %(TOPLEVEL)s/experiments/'${Voice}'/acoustic_model'# $acoustic_config_file

$SED -i s#'file_id_list:.*'#'file_id_list: %(data)s/'${FileIDList}# $acoustic_config_file

# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $acoustic_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $acoustic_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $acoustic_config_file


echo "Acoustic configuration settings stored in $acoustic_config_file"
