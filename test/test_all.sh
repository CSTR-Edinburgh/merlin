#!/bin/bash
# Script for running all tests in this directory
# This script has to be run in its directory, as shows the usage.


#  main ------------------------------------------------------------------------

if test "$#" -ne 0; then
    echo "Usage: ./test_all.sh"
    exit 1
fi

mkdir -p log

TIMESTAMP=`date +'%Y-%m-%d-%H-%M-%S'`
LOGFILE=log/$TIMESTAMP-test_all.sh
GITVERSION=`git version`

if [[ "$GITVERSION" ]]; then
    echo 'Git is available in the working directory:' >> $LOGFILE 2>&1
    echo '  Merlin version: ' "`git describe --tags --always`" >> $LOGFILE 2>&1
    echo '  branch: ' "`git rev-parse --abbrev-ref HEAD`" >> $LOGFILE 2>&1
    echo '  status: ' >> ${LOGFILE}.gitstatus 2>&1
    git status >> ${LOGFILE}.gitstatus 2>&1
    echo '  diff to Merlin version: ' >> ${LOGFILE}.gitdiff 2>&1
    git diff >> ${LOGFILE}.gitdiff 2>&1
    echo ' '
fi

bash ./test_install.sh >> $LOGFILE 2>&1

python ./test_classes.py  >> $LOGFILE 2>&1

bash ./test_training.sh >> $LOGFILE 2>&1
