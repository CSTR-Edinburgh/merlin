#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

if test "$#" -ne 2; then
    echo "################################"
    echo "Usage:"
    echo "./04_build_hybrid_model.sh <path_to_label_align_dir> <path_to_feat_dir>"
    echo ""
    echo "default path to lab dir(Input): database/label_${Labels}"
    echo "default path to feat dir(Input): database/lab"
    echo "################################"
    exit 1
fi

script_dir=${MerlinDir}/misc/scripts/hybrid_voice

# step 1 - convert labels to festival format
python ${script_dir}/convert_hts_label_format_to_festival.py ${label_align_dir} ${festival_lab_dir} database/${FileIDList} ${Labels}

# step 2 - prepare tcoef features
# requires mgc, bap, lf0 as well in database of training data 
# generated data is always better than natural data

feat_dir=$(dirname $festival_lab_dir)
sed -i s#'data_dir =.*'#'data_dir = "'$WorkDir/${feat_dir}'"'# ${script_dir}/compute_tcoef_features.py

python ${script_dir}/compute_tcoef_features.py

# step 3 - prepare hybrid voice
cp -r $FESTDIR/lib/voices/english/cstr_edi_${Voice} $FESTDIR/lib/voices/english/cstr_edi_${Voice}_hybrid 
cp -r database/tcoef $FESTDIR/lib/voices/english/cstr_edi_${Voice}_hybrid/${SPEAKER}/ 

