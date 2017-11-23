
# Demo: slt_arctic data + MagPhase Vocoder


## Install

### Download Merlin
Step 1: git clone https://github.com/CSTR-Edinburgh/merlin.git

### Install tools

Step 2: cd merlin/tools <br/>
Step 3: ./compile_tools.sh


## Usage
### Demo voice (small subset)
To run demo voice, please follow below steps:

Step 4a: cd merlin/egs/slt_arctic/s2 <br/>
Step 5a: ./run_demo.sh

Demo voice trains only on 50 utterances and shouldn't take more than 5 min.

### Full voice (whole dataset)

To run full voice, please follow below steps:

Step 4b: cd merlin/egs/slt_arctic/s2 <br/>
Step 5b: ./run_full_voice.sh

Full voice utilizes the whole arctic data (1132 utterances). The training of the voice approximately takes 1 to 2 hours.


### Generate new sentences
To generate new sentences, please follow below steps:

Step 6: Run either demo voice or full voice. <br/>
Step 7: ./merlin_synthesis.sh

## Notes
You can chooose the postfilter switching the variable **use_magphase_pf**. If True, it uses the internal MagPhase postfiler (default). If False, Merlin's postfilter is applied.

## TODO
* Tunning of MagPhase internal postfilter.
* MCD computation.
* Make the **"build_your_own_voice"** section for MagPhase vocoder.


