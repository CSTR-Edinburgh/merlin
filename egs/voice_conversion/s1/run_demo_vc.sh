#!/bin/bash

# download demo data (300 utterances)
wget http://104.131.174.95/downloads/voice_conversion/bdl_arctic.zip
wget http://104.131.174.95/downloads/voice_conversion/slt_arctic.zip

# unzip files
unzip -q bdl_arctic.zip
unzip -q slt_arctic.zip

# step 1: run setup
./01_setup.sh bdl slt

# copy data
mv bdl_arctic/wav/ database/bdl/
mv slt_arctic/wav/ database/slt/

# step 2: extract acoustic features
./02_prepare_acoustic_features.sh database/bdl/wav/ database/bdl/feats
./02_prepare_acoustic_features.sh database/slt/wav/ database/slt/feats

# step 3: align source features with target (create parallel data)
./03_align_src_with_target.sh database/bdl/feats/ database/slt/feats/ database/bdl_aligned_with_slt/feats

# step 4: prepare config files for training and testing
./04_prepare_conf_files.sh conf/global_settings.cfg

# step 5: train voice-conversion model
# prepare acoustic features for source
./05_train_acoustic_model.sh conf/acoustic_bdl.conf

# create a symbolic link for source features in target-voice directory
./scripts/create_symbolic_link.sh

# train an acoustic model with mapping from source to target
./05_train_acoustic_model.sh conf/acoustic_bdl2slt.conf

# step 6: run voice conversion
# let's copy some files for testing
mkdir experiments/bdl2slt/test_synthesis/bdl
mkdir experiments/bdl2slt/test_synthesis/slt
cp database/bdl/wav/arctic_a0001.wav experiments/bdl2slt/test_synthesis/bdl/test_01.wav
cp database/bdl/wav/arctic_a0002.wav experiments/bdl2slt/test_synthesis/bdl/test_02.wav
cp database/bdl/wav/arctic_a0003.wav experiments/bdl2slt/test_synthesis/bdl/test_03.wav
basename --suffix=.wav -- experiments/bdl2slt/test_synthesis/bdl/* > experiments/bdl2slt/test_synthesis/test_id_list.scp

# run voice conversion bdl2slt
./06_run_merlin_vc.sh experiments/bdl2slt/test_synthesis/bdl conf/test_synth_bdl.conf conf/test_synth_bdl2slt.conf

echo "done...!"
