#!/bin/bash

# 1. Get and compile SPTK

echo "downloading SPTK-3.9..."
wget http://downloads.sourceforge.net/sp-tk/SPTK-3.9.tar.gz
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
)
