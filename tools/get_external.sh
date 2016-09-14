#!/bin/bash

# 1. Get and compile SPTK
wget http://downloads.sourceforge.net/sp-tk/SPTK-3.9.tar.gz
tar xzvf SPTK-3.9.tar.gz
(
    cd SPTK-3.9;
    ./configure --prefix=$PWD/../env;
    make;
    make install
)

# 2. Getting WORLD
