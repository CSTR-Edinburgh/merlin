Download Merlin
---------------

Step 1: git clone https://github.com/CSTR-Edinburgh/merlin.git 

Install tools
-------------

Step 2: cd merlin/tools <br/>
Step 3: ./compile_tools.sh

Demo voice
----------

To run demo voice, please follow below steps:
 
Step 4: cd merlin/egs/slt_arctic/s1 <br/>
Step 5: ./run_demo.sh

Demo voice trains only on 50 utterances and shouldn't take more than 5 min. 

Compare the results in log files to baseline results from demo data in [RESULTS.md](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/slt_arctic/s1/RESULTS.md)

Full voice
----------

To run full voice, please follow below steps:

Step 6: cd merlin/egs/slt_arctic/s1 <br/>
Step 7: ./run_full_voice.sh

Full voice utilizes the whole arctic data (1132 utterances). The training of the voice approximately takes 1 to 2 hours. 

Compare the results in log files to baseline results from full data in [RESULTS.md](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/slt_arctic/s1/RESULTS.md)

Generate new sentences
----------------------

To generate new sentences, please follow below steps:

Step 8: Run either demo voice or full voice. <br/>
Step 9: ./merlin_synthesis.sh

