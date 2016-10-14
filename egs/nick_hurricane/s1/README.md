Download Merlin
---------------

git clone https://github.com/CSTR-Edinburgh/merlin.git

Setup
-----

To setup demo voice: 

./01_setup.sh nick_hurricane_demo

    (or)   

To setup full voice:

./01_setup.sh nick_hurricane_full

Demo setup makes use of short amount of data (60 utterances) for training, validation and testing. <br/>
Full setup makes use of whole data (2542 utterances) for training, validation and testing. 

Run Merlin
----------

Once after setup, use below script to create acoustic, duration models and perform final test synthesis:

./02_run_merlin.sh

If demo setup is used, merlin trains only on 50 utterances and should not take more than 5 min. <br/>
Compare the results in log files to baseline results from demo data in [RESULTS.md](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/nick_hurricane/s1/RESULTS.md)

If full setup is used, merlin utilizes the whole cstr-hurricane data (2542 utterances). The training of the voice approximately takes 1 to 2 hours. <br/>
Compare the results in log files to baseline results from full data in [RESULTS.md](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/nick_hurricane/s1/RESULTS.md)

Generate new sentences
----------------------

To generate new sentences, please follow [steps] (https://github.com/CSTR-Edinburgh/merlin/issues/28) in below script:

./03_merlin_synthesis.sh

