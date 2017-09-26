#!/bin/bash

global_config_file=conf/global_settings.cfg
source $global_config_file

setup_data=true
build_voice=true
copy_voice=true

#################################
########### lexicon #############
#################################
# Choose lexicon:
# 1. cmulex
# 2. unilex-rpx
# 3. combilex-rpx

#################################
########### gender ##############
#################################
# Choose gender:
# 1. 'm' for male
# 2. 'f' for female

## use default options: unilex and female assuming slt database
if test "$#" -ne 2; then
    lexicon="unilex-rpx"
    gender="f"
else
    lexicon=$1
    gender=$2
fi

voice_dir=experiments/${Voice}
SPEAKER=slt

# setup directory structure and download data
if [ "$setup_data" = true ]; then
    # download demo data (60 utterances)
    wget http://104.131.174.95/downloads/build_your_own_voice/slt_demo/wav.zip
    wget http://104.131.174.95/downloads/build_your_own_voice/slt_demo/txt.data

    # unzip files
    unzip -q wav.zip

    mkdir -p database

    # copy data
    mv wav database/wav
    mv txt.data database/

    rm -rf wav.zip 
fi

# build unit-selection system
if [ "$build_voice" = true ]; then
    # step 1 - setup
    cd ${voice_dir}
    source $MULTISYN_BUILD/multisyn_build.sh
    $MULTISYN_BUILD/bin/setup

    # step 2 - copy audio files and text
    cp ${WorkDir}/database/wav/* wav/
    cp ${WorkDir}/database/txt.data utts.data

    # step 3 - prepare labels 
    $MULTISYN_BUILD/bin/setup_alignment

    cp $MULTISYN_BUILD/resources/phone_list.${lexicon} alignment/phone_list
    cp $MULTISYN_BUILD/resources/phone_substitutions.${lexicon} alignment/phone_substitutions

    printf "postlex_apos_s_check\npostlex_the_vs_thee\npostlex_intervoc_r\npostlex_a" > postlex_rules
    touch my_lexicon.scm
    $MULTISYN_BUILD/bin/make_initial_phone_labs utts.data utts.mlf $lexicon postlex_rules my_lexicon.scm

    # step 4 - add noise
    $MULTISYN_BUILD/bin/add_noise wav utts.data

    # step 5 - prepare mfcc
    $MULTISYN_BUILD/bin/make_mfccs alignment wav utts.data

    # step 6 - force-alignment
    cd alignment
    $MULTISYN_BUILD/bin/make_mfcc_list  ../utts.data train.scp ../mfcc
    ln -s ../utts.mlf aligned.0.mlf
    $MULTISYN_BUILD/bin/do_alignment .
    cd..
    $MULTISYN_BUILD/bin/break_mlf alignment/aligned.4.mlf lab

    # step 7 - extract pitchmarks
    $MULTISYN_BUILD/bin/make_pm_wave -${gender} pm wav utts.data
    $MULTISYN_BUILD/bin/make_pm_fix pm utts.data

    # step 8 - compute power factors
    $MULTISYN_BUILD/bin/find_powerfactors lab utts.data
    $MULTISYN_BUILD/bin/make_wav_powernorm wav_fn wav utts.data

    # repeating steps 4-7 with wav_fn
    $MULTISYN_BUILD/bin/add_noise wav_fn utts.data

    $MULTISYN_BUILD/bin/make_mfccs alignment wav_fn utts.data

    cd alignment
    $MULTISYN_BUILD/bin/make_mfcc_list  ../utts.data train.scp ../mfcc
    $MULTISYN_BUILD/bin/do_alignment .
    cd..
    $MULTISYN_BUILD/bin/break_mlf alignment/aligned.4.mlf lab

    $MULTISYN_BUILD/bin/make_pm_wave -${gender} pm wav_fn utts.data
    $MULTISYN_BUILD/bin/make_pm_fix pm utts.data

    # step 9 - flag bad energy phones
    $MULTISYN_BUILD/bin/make_frame_ene utts.data
    $MULTISYN_BUILD/bin/Get_lr_ene utts.data
    $MULTISYN_BUILD/bin/Flag_bad_energy utts.data

    # step 10 - calculate duration
    $MULTISYN_BUILD/bin/phone_lengths dur lab utts.data

    # step 11 - build utts
    $MULTISYN_BUILD/bin/build_utts utts.data ${lexicon} postlex_rules

    # step 12 - alignment
    cd alignment
    $MULTISYN_BUILD/bin/do_final_alignment ../utts.data ${lexicon} ../postlex_rules n
    cd ..

    # step 13 - compute F0
    $MULTISYN_BUILD/bin/make_f0 -${gender} wav_fn utts.data

    # step 14 - prepare coefs
    $MULTISYN_BUILD/bin/make_norm_join_cost_coefs coef f0 mfcc utts.data
    $MULTISYN_BUILD/bin/strip_join_cost_coefs coef coef_stripped utt utts.data

    # step 15 - prepare LPC
    $MULTISYN_BUILD/bin/make_lpc wav utts.data 
fi

# copy voice into festival/lib/voices
if [ "$copy_voice" = true ]; then
    mkdir $FESTDIR/lib/voices/english/cstr_edi_${Voice}
    mkdir $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}
    mkdir $FESTDIR/lib/voices/english/cstr_edi_${Voice}/festvox

    cp -r wav_fn $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/wav
    cp -r coef $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/coef 
    cp -r f0 $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/f0
    cp -r pm $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/pm
    cp -r utt $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/utt
    cp -r utts.data $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/utts.data

    cp -r ${SPEAKER}_pauses.data $FESTDIR/lib/voices/english/cstr_edi_${Voice}/${SPEAKER}/
fi

