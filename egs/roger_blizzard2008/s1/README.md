Download Merlin
---------------

git clone https://github.com/CSTR-Edinburgh/merlin.git

Setup
-----

To setup voice: 

./01_setup.sh give_a_voice_name

Prepare Data
------------

To derive labels, use alignment scripts provided below: <br/>
a) state_align - https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/alignment/state_align <br/>
b) phone_align - https://github.com/CSTR-Edinburgh/merlin/tree/master/misc/scripts/alignment/phone_align

Then, chose the vocoder: <br/>
a) STRAIGHT - extracts 60-dim MGC, 25-dim BAP, 1-dim LF0 <br/>
b) WORLD    - extracts 60-dim MGC, variable-dim BAP, 1-dim LF0 <br/>
            - BAP dim (1 for 16Khz, 5 for 48Khz)  <br/>
c) WORLD_v2 - extracts 60-dim MGC, 5-dim BAP, 1-dim LF0 <br/>

To derive acousitc features, use vocoder scripts provided below: <br/>
a) STRAIGHT - https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/straight/extract_features_for_merlin.sh <br/>
b) WORLD    - https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world/extract_features_for_merlin.sh <br/>
c) WORLD_v2 - https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world_v2/extract_features_for_merlin.sh <br/>

Run below script for instructions:
./02_prepare_data.sh

Run Merlin
----------

Once after setup, use below script to create acoustic, duration models and perform final test synthesis:

./03_run_merlin.sh


Generate new sentences
----------------------

To generate new sentences, please follow [steps] (https://github.com/CSTR-Edinburgh/merlin/issues/28) in below script:

./04_merlin_synthesis.sh

