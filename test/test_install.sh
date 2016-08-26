#!/bin/bash
# Script for testing the installation of Merlin pipeline
# python modules are checked, libraries versions, basic theano calls, etc.
# This script has to be run in its directory, as shows the usage.

if test "$#" -ne 0; then
    echo "Usage: ./test_install.sh"
    exit 1
fi

src_dir=../src

# Source install-related environment variables
source ${src_dir}/setup_env.sh


# Start checking versions

echo -n "Python version: "
python -c 'import sys; print("version "+sys.version.replace("\n",""))'
if [[ "$?" == "0" ]]; then echo "OK"; else echo "No python installed      FAILED"; fi
echo " "

echo -n "Python Numpy version: "
python -c 'import numpy; print("version "+numpy.version.version)'
if [[ "$?" == "0" ]]; then echo "OK"; else echo "numpy not accessible  FAILED"; fi
echo " "

echo -n "Python Theano version: "
python -c 'import theano; print("version "+theano.version.version)'
if [[ "$?" == "0" ]]; then echo "OK"; else echo "theano not accessible  FAILED"; fi
echo " "


# # Run full theano tests (very heavy)
# 
# echo -n "Test python module: theano: "
# # Try to lock a GPU...
# gpu_id=$(python ${src_dir}/gpu_lock.py --id-to-hog)
# 
# if [ $gpu_id -gt -1 ]; then
#     echo "Running on GPU id=$gpu_id ..."
# 
#     THEANO_FLAGS="mode=FAST_RUN,device=gpu$gpu_id,"$MERLIN_THEANO_FLAGS
#     export THEANO_FLAGS
#     
#     python -c 'import theano; theano.test()'
#     
#     python ${src_dir}/gpu_lock.py --free $gpu_id
# else
#     echo "No GPU is available! Running on CPU..."
# 
#     THEANO_FLAGS=$MERLIN_THEANO_FLAGS
#     export THEANO_FLAGS
#     
#     python -c 'import theano; theano.test()'
# fi
