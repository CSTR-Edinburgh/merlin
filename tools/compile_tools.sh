#!/bin/bash

# 1. Get and compile SPTK

echo "downloading SPTK-3.9..."
sptk_url=http://downloads.sourceforge.net/sp-tk/SPTK-3.9.tar.gz
if hash curl 2>/dev/null; then
    curl -L -O $sptk_url
elif hash wget 2>/dev/null; then
    wget $sptk_url
else
    echo "please download the SPTK-3.9 from $sptk_url"
    exit 1
fi
tar xzf SPTK-3.9.tar.gz

echo "compiling SPTK..."
(
    cd SPTK-3.9;
    ./configure --prefix=$PWD/build;
    make;
    make install
)

# 2. Getting WORLD

echo "compiling WORLD..."
(
    cd WORLD;
    make
    make analysis synth
    make clean
)

# 3. Copy binaries

SPTK_BIN_DIR=bin/SPTK-3.9
WORLD_BIN_DIR=bin/WORLD

mkdir -p bin
mkdir -p $SPTK_BIN_DIR
mkdir -p $WORLD_BIN_DIR

cp SPTK-3.9/build/bin/* $SPTK_BIN_DIR/
cp WORLD/build/analysis $WORLD_BIN_DIR/
cp WORLD/build/synth $WORLD_BIN_DIR/

if [[ ! -f ${SPTK_BIN_DIR}/x2x ]]; then
    echo "Error installing SPTK tools"
    exit 1
elif [[ ! -f ${WORLD_BIN_DIR}/analysis ]]; then
    echo "Error installing WORLD tools"
    exit 1
else
    echo "All tools successfully compiled!!"
fi
