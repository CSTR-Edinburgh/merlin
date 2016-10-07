Create training labels for merlin with festvox tools
----------------------------------------------------

Step 0: Install [festvox] (http://festvox.org/download.html) and configure it's path.

Below here, we provide some steps which are adapted from: <br/>
[http://festvox.org/bsv/c3174.html] (http://festvox.org/bsv/c3174.html)

Step 1: mkdir cmu_us_slt_arctic <br/>
Step 2: cd cmu_us_slt_arctic <br/>

Step 3: $FESTVOXDIR/src/clustergen/setup_cg cmu us slt_arctic <br/>

Step 4: Download [cmuarctic.data] (http://festvox.org/cmu_arctic/cmuarctic.data) and copy into etc/txt.done.data <br/>
a) wget http://festvox.org/cmu_arctic/cmuarctic.data <br/>
b) cp cmuarctic.data etc/txt.done.data

Step 4: ./bin/do_build build_prompts <br/>
Step 5: ./bin/do_build label <br/>

After all above steps have been performed successfully, you should be able to find the labelled data <br/>
in the directory "lab" and corresponding utts in "festival/utts".

From here, you should use the scripts in Merlin to convert festival utts to full-contextual phone labels.

Step 6: cd ../

Step 7: configure frontend and FESTDIR paths. <br/>
a) frontend=${MerlinDir}/misc/scripts/frontend <br/>
b) FESTDIR=${MerlinDir}/tools/festival

Step 8: create a file id list <br/>
cat slt_arctic_demo_data/cmuarctic.data | cut -d " " -f 2 > file_id_list.scp

Step 9: convert festival utts to lab <br/>
${frontend}/festival_utt_to_lab/make_labels \ <br/>
                            full-context-labels \ <br/>
                            cmu_us_slt_arctic/festival/utts \ <br/>
                            ${FESTDIR}/examples/dumpfeats \ <br/>
                            ${frontend}/festival_utt_to_lab 

Step 10: normalize lab for merlin <br/>
python ${frontend}/utils/normalize_lab_for_merlin.py \ <br/>
                            full-context-labels/full \ <br/>
                            label_phone_align \ <br/>
                            phone_align \ <br/>
                            file_id_list.scp <br/>

You should have your labels ready !! :)

