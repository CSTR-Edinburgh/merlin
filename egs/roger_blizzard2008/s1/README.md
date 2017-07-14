Download Merlin
---------------

Step 1: git clone https://github.com/CSTR-Edinburgh/merlin.git

Install tools
-------------

Step 2: cd merlin/tools <br/>
Step 3: ./compile_tools.sh
Step 4: install festival and HTS at merlin/tools/
	Possible help: [Issue96](https://github.com/CSTR-Edinburgh/merlin/issues/96)

Setup
-----

This example uses the roger database from the blizzard 2008 challange.
The database is not freely available, you can only run this example, if you have access to the database.
Use
Step 5: export ROGER_DB=/path/to/your/roger_database/
in your console before running the example.

To setup voice: 

Take a look at ./01_setup.sh
You probably have to change the way the database is accessed, this depends on how your database is structured.
Check the lines 70-95, the comments should guide you through the process.

Demo voice
----------

To run demo voice, please follow below steps:
 
Step 6: cd merlin/egs/roger_blizzard2008/s1 <br/>
Step 7: ./run_demo_voice.sh

Demo voice trains only on 281 utterances (theherald1) and shouldn't take long.

Full voice
----------

To run full voice, please follow below steps:

Step 6: cd merlin/egs/roger_blizzard2008/s1 <br/>
Step 7: ./run_full_voice.sh

Full voice utilizes carroll, arctic and theherald1-3 (4871 utterances). The training of the voice approximately takes 3 to 4 hours. 

Generate new sentences
----------------------

To generate new sentences, please follow below steps:

Step  8: Run either demo voice or full voice. <br/>
Step  9: Place the txt files containing the utterances in experiments/roger_demo OR roger_full/test_synthesis/txt
Step 10: ./merlin_synthesis.sh

