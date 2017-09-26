#!/bin/bash

download_data=true
extract_data=true

if [ "$download_data" = true ]; then
    # download VCTK corpus (~11GB)
    echo "downloading data....."
    data_url=http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz
    if hash curl 2>/dev/null; then
        curl -O $data_url
    elif hash wget 2>/dev/null; then
        wget $data_url
    else
        echo "please download the data from $data_url"
        exit 1
    fi
fi

if [ "$extract_data" = true ]; then
    echo "extracting files......"
    # It's recommend to check the README and speaker-info.txt files
    # after extraction
    tar -zxvf VCTK-Corpus.tar.gz
fi

# Uncomment the below line if you want to delete the tar file
# rm -rf VCTK-Corpus.tar.gz