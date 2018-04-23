#!/bin/bash

#########################################
######### Install Dependencies ##########
#########################################
#sudo apt-get install csh realpath

tools_dir=$(dirname $0)
cd $tools_dir

install_sptk=true
install_postfilter=true
install_world=true
install_reaper=true
install_magphase=true

# 1. Get and compile SPTK
if [ "$install_sptk" = true ]; then
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

    # Get and compile Postfilter
    if [ "$install_postfilter" = true ]; then
        echo "downloading postfilter..."
        postfilter_url=http://104.131.174.95/downloads/tools/postfilter.tar.gz
        if hash curl 2>/dev/null; then
            curl -L -O $postfilter_url
        elif hash wget 2>/dev/null; then
            wget $postfilter_url
        else
            echo "please download the postfilter from $postfilter_url"
            exit 1
        fi
        tar xzf postfilter.tar.gz
        
        echo "compiling postfilter..."
        (
            cd postfilter/src;
            ./00_make.sh
        )
    fi
fi


# 2. Getting WORLD
if [ "$install_world" = true ]; then
    echo "compiling WORLD..."
    (
        cd WORLD;
        make
        make analysis synth
        make clean
    )
fi


# 3. Getting REAPER
if [ "$install_reaper" = true ]; then
    echo "downloading REAPER..."
    git clone https://github.com/google/REAPER.git
    echo "compiling REAPER..."
    (
        cd REAPER
        mkdir build   # In the REAPER top-level directory
        cd build
        cmake ..
        make
    )
fi


SPTK_BIN_DIR=bin/SPTK-3.9
WORLD_BIN_DIR=bin/WORLD
REAPER_BIN_DIR=bin/REAPER

# 4. Getting MagPhase vocoder:
if [ "$install_magphase" = true ]; then
    echo "downloading MagPhase vocoder..."
    rm -rf magphase
    git clone https://github.com/CSTR-Edinburgh/magphase.git
    #git clone https://github.com/felipeespic/magphase.git

    echo "configuring MagPhase..."
    (
        mkdir -p magphase/tools/bin
        cp -n SPTK-3.9/build/bin/b2mc   magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/bcp    magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/c2acr  magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/freqt  magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/mc2b   magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/mcep   magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/merge  magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/sopr   magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/vopr   magphase/tools/bin/
        cp -n SPTK-3.9/build/bin/x2x    magphase/tools/bin/
        cp -n REAPER/build/reaper       magphase/tools/bin/
    )
fi


# 5. Copy binaries
echo "deleting downloaded tar files..."
rm -rf $tools_dir/*.tar.gz



mkdir -p bin
mkdir -p $SPTK_BIN_DIR
mkdir -p $WORLD_BIN_DIR
mkdir -p $REAPER_BIN_DIR

cp SPTK-3.9/build/bin/* $SPTK_BIN_DIR/
cp postfilter/bin/* $SPTK_BIN_DIR/
cp WORLD/build/analysis $WORLD_BIN_DIR/
cp WORLD/build/synth $WORLD_BIN_DIR/
cp REAPER/build/reaper $REAPER_BIN_DIR/

if [[ ! -f ${SPTK_BIN_DIR}/x2x ]]; then
    echo "Error installing SPTK tools! Try installing dependencies!!"
    echo "sudo apt-get install csh"
    exit 1
elif [[ ! -f ${SPTK_BIN_DIR}/mcpf ]]; then
    echo "Error installing postfilter tools! Try installing dependencies!!"
    echo "sudo apt-get install realpath"
    echo "sudo apt-get install autotools-dev"
    echo "sudo apt-get install automake"
    exit 1
elif [[ ! -f ${WORLD_BIN_DIR}/analysis ]]; then
    echo "Error installing WORLD tools"
    exit 1
else
    echo "All tools successfully compiled!!"
fi
