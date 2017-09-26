#!/bin/bash

setup_train_data=true
setup_eval_data=true

# download vcc2016 training data
if [ "$setup_train_data" = true ]; then
    # download vcc2016 training data
    wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/vcc2016_training.zip

    # unzip files
    unzip -q vcc2016_training.zip

    # delete zip file
    rm -rf vcc2016_training.zip
fi

# download vcc2016 evaluation data
if [ "$setup_eval_data" = true ]; then
    # download evaluation data
    wget http://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/evaluation_all.zip
    
    # unzip files
    unzip -q evaluation_all.zip

    # delete zip file
    rm -rf evaluation_all.zip
fi

echo "done...!"
