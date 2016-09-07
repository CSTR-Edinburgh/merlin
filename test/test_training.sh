#!/bin/bash
# Script for testing Merlin pipeline using characterization tests
# This script has to be run in its directory, as shows the usage.


# Some functions to extract results from logs

get_duration_rmse_corr(){
    cat $1 |grep "main: Test" |sed 's/^.*RMSE: \(.*\) frames.*CORR: \(.*\);/\1 \2/'
}
get_acoustic_mcd_bap_f0rmse_f0corr_vuv(){
    cat $1 |grep "main: Test" |sed 's/^.*MCD: \(.*\) dB; BAP: \(.*\) dB;.*RMSE: \(.*\) Hz; CORR: \(.*\); VUV: \(.*\)%/\1 \2 \3 \4 \5/'
}

#  main ------------------------------------------------------------------------

if test "$#" -ne 0; then
    echo "Usage: ./test_training.sh"
    exit 1
fi


TESTROOT=`pwd`

FAILED=""

GIT_DESCR=`git describe --tags --always`
GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
GIT_DIFFS=`git diff`
if [[ -z "$GIT_DIFFS" ]]; then
    echo "The git indicates no difference against $GIT_BRANCH-$GIT_DESCR"
    echo "(you're testing the git version, tests are supposed to pass flawlessly)"
else
    echo "There are differences against $GIT_BRANCH-$GIT_DESCR"
    echo "(you're testing local changes, differences can be expected)"
fi
echo " "


TESTNAME=charac_slt_short
    CHARAC_SHORTEXP=../egs/slt_arctic/s1
    echo "Characterization test on short data ($CHARAC_SHORTEXP)..."
    cd $CHARAC_SHORTEXP

    # Run the pipeline
    ./run_demo.sh
    if [[ "$?" != "0" ]]; then
        FAILED="$FAILED ${TESTNAME}_merlin_execution"
    else
        REF=$(get_duration_rmse_corr testrefs/slt_arctic_demo/duration_model/log/DNN_TANH_TANH_TANH_TANH_LINEAR__dur_50_259_4_512_0.002000*.log)
        TESTFILE=`grep -Rnsl 'main: Test' experiments/slt_arctic_demo/duration_model/log/DNN_TANH_TANH_TANH_TANH_LINEAR__dur_50_259_4_512_0.002000*.log`
        TEST=$(get_duration_rmse_corr $TESTFILE)
        python $TESTROOT/test_compare.py --ref $REF --test $TEST --colnames RMSE CORR
        if [[ "$?" == "1" ]]; then FAILED="$FAILED ${TESTNAME}_duration"; fi

        REF=$(get_acoustic_mcd_bap_f0rmse_f0corr_vuv testrefs/slt_arctic_demo/acoustic_model/log/DNN_TANH_TANH_TANH_TANH_LINEAR__mgc_lf0_vuv_bap_50_259_4_512_0.002000*.log)
        TESTFILE=`grep -Rnsl 'main: Test' experiments/slt_arctic_demo/acoustic_model/log/DNN_TANH_TANH_TANH_TANH_LINEAR__mgc_lf0_vuv_bap_50_259_4_512_0.002000*.log`
        TEST=$(get_acoustic_mcd_bap_f0rmse_f0corr_vuv $TESTFILE)
        python $TESTROOT/test_compare.py --ref $REF --test $TEST --colnames MCD BAP F0-RMSE F0-CORR VUV
        if [[ "$?" == "1" ]]; then FAILED="$FAILED ${TESTNAME}_acoustic"; fi
    fi
    echo " "

# Add new tests here

# TESTNAME=charac_slt_full ?


    
if [[ -n "$FAILED" ]]; then
    echo "Failed tests: "$FAILED
else
    echo "All test passed!"
fi

cd $TESTROOT

