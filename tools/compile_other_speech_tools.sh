#!/bin/bash

#########################################
######### Install Dependencies ##########
#########################################
#sudo apt-get -y install libncurses5 libncurses5-dev libcurses-ocaml # for sudo users only

current_working_dir=$(pwd)
tools_dir=${current_working_dir}/$(dirname $0)
cd $tools_dir

FESTIVAL_BASELINE_URL=http://www.cstr.ed.ac.uk/downloads/festival/2.4
install_speech_tools=false
install_festival=false
install_festvox=true

# 1. Get and compile speech tools
if [ "$install_speech_tools" = true ]; then

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

    echo "compiling festival..."
    (
        cd festival;
        ./configure;
        make;
        make install
    )

    echo "downloading some useful lexicons..."
    dict1_url=$FESTIVAL_BASELINE_URL/festlex_CMU.tar.gz
    dict2_url=$FESTIVAL_BASELINE_URL/festlex_OALD.tar.gz
    dict3_url=$FESTIVAL_BASELINE_URL/festlex_POSLEX.tar.gz
    if hash curl 2>/dev/null; then
        curl -L -O $dict1_url
        curl -L -O $dict2_url
        curl -L -O $dict3_url
    elif hash wget 2>/dev/null; then
        wget $dict1_url
        wget $dict2_url
        wget $dict3_url
    else
        echo "please download dictionaries from $FESTIVAL_BASELINE_URL"
        exit 1
    fi
    tar xzf festlex_CMU.tar.gz
    tar xzf festlex_OALD.tar.gz
    tar xzf festlex_POSLEX.tar.gz

    echo "downloading some voices for English..."
    festival_voice_url=http://festvox.org/packed/festival/2.4/voices
    voice1_url=$FESTIVAL_BASELINE_URL/voices/festvox_kallpc16k.tar.gz
    voice2_url=$FESTIVAL_BASELINE_URL/voices/festvox_rablpc16k.tar.gz
    voice3_url=$FESTIVAL_BASELINE_URL/voices/festvox_cmu_us_slt_cg.tar.gz
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

# 3. Get and compile festvox
if [ "$install_festvox" = true ]; then
    echo "compiling festvox..."
    (
        cd festvox;
        ./configure;
        make;
    )

fi

# export paths
export FESTVOXDIR=$tools_dir/festvox

echo "deleting downloaded tar files..."
rm -rf $tools_dir/*.tar.gz

if [[ ! -f ${ESTDIR}/bin/ch_track ]]; then
    echo "Error installing speech tools"
    exit 1
elif [[ ! -f ${FESTDIR}/bin/festival ]]; then
    echo "Error installing Festival"
    exit 1
elif [[ ! -f ${FESTVOXDIR}/src/vc/build_transform ]]; then
    echo "Error installing Festvox"
    exit 1
else
    echo "All tools successfully compiled!!"
fi
