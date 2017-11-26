#!/bin/bash

if test "$#" -ne 1; then
    echo "Usage: ./scripts/prepare_config_files_for_synthesis.sh conf/global_settings.cfg"
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

acoustic_config_file=conf/test_synth_${Source}.conf

# Start with a general recipe...
cp -f conf/general/acoustic_demo.conf $acoustic_config_file

# ... and modify it:

$SED -i s#'Merlin\s*:.*'#'Merlin: '$MerlinDir# $acoustic_config_file
$SED -i s#'TOPLEVEL\s*:.*'#'TOPLEVEL: '${WorkDir}# $acoustic_config_file
$SED -i s#'work\s*:.*'#'work: %(TOPLEVEL)s/experiments/'${Source}'/acoustic_model'# $acoustic_config_file

$SED -i s#'file_id_list\s*:.*'#'file_id_list: %(data)s/'${FileIDList}# $acoustic_config_file
$SED -i s#'test_id_list\s*:.*'#'test_id_list: %(TOPLEVEL)s/experiments/'${Voice}'/test_synthesis/test_id_list.scp'# $acoustic_config_file

# [Waveform]
$SED -i s#'test_synth_dir\s*:.*'#'test_synth_dir: %(TOPLEVEL)s/experiments/'${Voice}'/test_synthesis/'${Target}# $acoustic_config_file

# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $acoustic_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $acoustic_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $acoustic_config_file


# [Processes]
$SED -i s#'GenTestList\s*:.*'#'GenTestList: True'# $acoustic_config_file

$SED -i s#'TRAINDNN\s*:.*'#'TRAINDNN: False'# $acoustic_config_file
$SED -i s#'DNNGEN\s*:.*'#'DNNGEN: False'# $acoustic_config_file
$SED -i s#'GENWAV\s*:.*'#'GENWAV: False'# $acoustic_config_file
$SED -i s#'CALMCD\s*:.*'#'CALMCD: False'# $acoustic_config_file


echo "Acoustic configuration settings stored in $acoustic_config_file"

#########################################
######## acoustic config file ###########
#########################################

acoustic_config_file=conf/test_synth_${Voice}.conf

# Start with a general recipe...
cp -f conf/general/acoustic_demo.conf $acoustic_config_file

# ... and modify it:

$SED -i s#'Merlin\s*:.*'#'Merlin: '$MerlinDir# $acoustic_config_file
$SED -i s#'TOPLEVEL\s*:.*'#'TOPLEVEL: '${WorkDir}# $acoustic_config_file
$SED -i s#'work\s*:.*'#'work: %(TOPLEVEL)s/experiments/'${Voice}'/acoustic_model'# $acoustic_config_file

$SED -i s#'file_id_list\s*:.*'#'file_id_list: %(data)s/'${FileIDList}# $acoustic_config_file
$SED -i s#'test_id_list\s*:.*'#'test_id_list: %(TOPLEVEL)s/experiments/'${Voice}'/test_synthesis/test_id_list.scp'# $acoustic_config_file

# [Waveform]

$SED -i s#'test_synth_dir\s*:.*'#'test_synth_dir: %(TOPLEVEL)s/experiments/'${Voice}'/test_synthesis/'${Target}# $acoustic_config_file

# [Data]
$SED -i s#'train_file_number\s*:.*'#'train_file_number: '${Train}# $acoustic_config_file
$SED -i s#'valid_file_number\s*:.*'#'valid_file_number: '${Valid}# $acoustic_config_file
$SED -i s#'test_file_number\s*:.*'#'test_file_number: '${Test}# $acoustic_config_file

# [Processes]
$SED -i s#'GenTestList\s*:.*'#'GenTestList: True'# $acoustic_config_file

$SED -i s#'MAKECMP\s*:.*'#'MAKECMP: False'# $acoustic_config_file
$SED -i s#'NORMCMP\s*:.*'#'NORMCMP: False'# $acoustic_config_file
$SED -i s#'TRAINDNN\s*:.*'#'TRAINDNN: False'# $acoustic_config_file
$SED -i s#'CALMCD\s*:.*'#'CALMCD: False'# $acoustic_config_file


echo "Acoustic configuration settings stored in $acoustic_config_file"
