# SLT Arctic TTS Demo using MagPhase Vocoder

## Overview

It is a Text-To-Speech demo using the [new release of the MagPhase vocoder (v2.0)](https://github.com/CSTR-Edinburgh/magphase), which now also supports:

* Constant frame-rate.
* Improved sound quality.
* Two types of post-filter available.
* Selectable number of coefficients for phase features (*real* and *imag*).
* Selectable number of coefficients for the magnitude feature (*mag*).


As a difference with other demos, it also includes **acoustic feature extraction**, thus the recipe works using as input data:
* label state aligned files (.lab).
* wav files (.wav).

Both are downloaded automatically during running the demo script.

## Run Demo Voice

Assuming Merlin is installed, just run:
```
cd merlin/egs/slt_arctic/s2/
python run_demo.py
```
Basically, ```run_demo.py``` script will:

1. Download the input data for you (.lab, .wav).
2. Create the experiment directory in ```./experiments```.
3. Perform acoustic feature extraction with MagPhase vocoder.
4. Build and train duration and acoustic models using Merlin.
5.  Synthesise waveforms using predicted durations. The synthesised waveforms will be stored in: ```/<experiment_dir>/test_synthesis/gen_acous_wav_pf_<postfilter_type>```

## Changing Parameters
Alternatively, you can also experiment by changing the input parameters (See section "INPUT" in *run_demo.py*):

* **exper_type:** Type of experiment. "demo" (50 training utts) or "full" (1k training utts.)

**Steps:**
* **b_download_data:** Download wavs and label data.
* **b_setup_data:** Copy downloaded data into the experiment directory. Plus, make a backup copy of *run_demo.py* script inside the experiment directory.
* **b_config_merlin:** Save new configuration files for Merlin.
* **b_feat_extr:** Perform acoustic feature extraction using the MagPhase vocoder.
* **b_conv_labs_rate:** Convert the state aligned labels to variable rate if running in variable frame rate mode (*b_const_rate=False*).
* **b_dur_train:** Merlin: Training of duration model.
* **b_acous_train:** Merlin: Training of acoustic model.
* **b_dur_syn:** Merlin: Generation of state durations using the duration model.
* **b_acous_syn:** Merlin: Waveform generation for the utterances provided in ```./test_synthesis/prompt-lab```


**MagPhase Vocoder:**

* **mag_dim:** Number of coefficients for magnitude feature (*mag*).
* **phase_dim:** Number of coefficients for phase features (*real* and *imag*).
* **b_const_rate:** To work in constant frame rate mode.
* **l_pf_type:** List containing the postfilters to apply during waveform generation.

* **b_feat_ext_multiproc:** Acoustic feature extraction done in multiprocessing mode (faster).