#!/bin/bash

#########################################
######### Install Dependencies ##########
#########################################
#sudo apt-get -y install libncurses5 libncurses5-dev libcurses-ocaml # for sudo users only

current_working_dir=$(pwd)
tools_dir=${current_working_dir}/$(dirname $0)
cd $tools_dir

install_speech_tools=true
install_festival=true
install_multisyn=true

# 1. Get and compile speech tools
if [ "$install_speech_tools" = true ]; then
    echo "downloading speech tools..."
    speech_tools_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-with-wrappers.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $speech_tools_url
    elif hash wget 2>/dev/null; then
        wget $speech_tools_url
    else
        echo "please download speech tools from $speech_tools_url"
        exit 1
    fi
    tar xzf speech_tools-2.4-with-wrappers.tar.gz

    echo "compiling speech tools..."
    (
        cd speech_tools;
        ./configure;
        make;
        make install
    )

fi

# export paths
export ESTDIR=$tools_dir/speech_tools
export LD_LIBRARY_PATH=$ESTDIR/lib:$LD_LIBRARY_PATH
export PATH=$ESTDIR/bin:$PATH

# 2. Get and compile festival, download dicts and some voices
if [ "$install_festival" = true ]; then
    echo "downloading festival..."
    festival_url=http://104.131.174.95/downloads/tools/festival-2.4-current.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $festival_url
    elif hash wget 2>/dev/null; then
        wget $festival_url
    else
        echo "please download Festival from $festival_url"
        exit 1
    fi
    tar xzf festival-2.4-current.tar.gz

    echo "compiling festival..."
    (
        cd festival;
        ./configure;
        make;
        make install
    )

    echo "downloading some useful lexicons..."
    dict1_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_CMU.tar.gz
    dict2_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_OALD.tar.gz
    dict3_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_POSLEX.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $dict1_url
        curl -L -O $dict2_url
        curl -L -O $dict3_url
    elif hash wget 2>/dev/null; then
        wget $dict1_url
        wget $dict2_url
        wget $dict3_url
    else
        echo "please download dictionaries from $festival_url"
        exit 1
    fi
    tar xzf festlex_CMU.tar.gz
    tar xzf festlex_OALD.tar.gz
    tar xzf festlex_POSLEX.tar.gz

    echo "downloading some voices for English..."
    festival_voice_url=http://festvox.org/packed/festival/2.4/voices
    voice1_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_kallpc16k.tar.gz
    voice2_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_rablpc16k.tar.gz
    voice3_url=http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_cmu_us_slt_cg.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $voice1_url
        curl -L -O $voice2_url
        curl -L -O $voice3_url
    elif hash wget 2>/dev/null; then
        wget $voice1_url
        wget $voice2_url
        wget $voice3_url
    else
        echo "please download Festival voices from $festival_voice_url"
        exit 1
    fi
    tar xzf festvox_kallpc16k.tar.gz
    tar xzf festvox_rablpc16k.tar.gz
    tar xzf festvox_cmu_us_slt_cg.tar.gz

fi

# export paths
export FESTDIR=$tools_dir/festival
export PATH=$FESTDIR/bin:$PATH

# 3. Get multisyn build
if [ "$install_multisyn" = true ]; then
    echo "downloading multisyn..."
    multisyn_url=http://www.cstr.ed.ac.uk/downloads/festival/multisyn_build/multisyn_build-1.20.tgz
    if hash curl 2>/dev/null; then
        curl -L -O $multisyn_url
    elif hash wget 2>/dev/null; then
        wget $multisyn_url
    else
        echo "please download multisyn from $multisyn_url"
        exit 1
    fi
    tar xzf multisyn_build-1.20.tgz

    echo "compiling multisyn..."
    (
        cd multisyn_build;
        make;
    )

fi

# export paths
export MULTISYN_BUILD=$tools_dir/multisyn_build

echo "deleting downloaded tar files..."
rm -rf $tools_dir/*.tar.gz
rm -rf $tools_dir/*.tgz

if [[ ! -f ${ESTDIR}/bin/ch_track ]]; then
    echo "Error installing speech tools"
    exit 1
elif [[ ! -f ${FESTDIR}/bin/festival ]]; then
    echo "Error installing Festival"
    exit 1
else
    echo "All tools successfully compiled!!"
fi

