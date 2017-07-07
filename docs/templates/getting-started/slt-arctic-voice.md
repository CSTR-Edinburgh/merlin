# Arctic voices

The CMU_ARCTIC databases were constructed at the Language Technologies Institute at 
Carnegie Mellon University as phonetically balanced, 
US English single speaker databases designed for unit selection speech synthesis research.

The databases consist of around 1150 utterances carefully selected from out-of-copyright texts from Project Gutenberg. 
The databses include US English male (bdl), female (slt) speakers (both experinced voice talent) and few other accented speakers.

To run one of these voices, `cd egs/slt_arctic/s1` and follow the below steps:

## Setting up

The first step is to run setup as it creates directories and downloads the required training data files.

To see the list of available voices, run:
```sh
./01_setup.sh
```
The next steps demonstrate on how to setup slt arctic voice. 

- To run on short data(about 50 utterances for training)
```sh
./01_setup.sh slt_arctic_demo
```
- To run on full data(about 1000 sentences for training)
```sh
./01_setup.sh slt_arctic_full
```

It also creates a global config file: `conf/global_settings.cfg`, where default settings are stored.
 
## Prepare config files

At this point, we have to prepare two config files to train DNN models
- Acoustic Model
- Duration Model

To prepare config files:
```sh
./02_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

## Train duration model

To train duration model:
```sh
./03_train_duration_model.sh <path_to_duration_conf_file>
```

## Train acoustic model

To train acoustic model:
```sh
./04_train_acoustic_model.sh <path_to_acoustic_conf_file>
```
## Synthesize speech

To synthesize speech:
```sh
./05_run_merlin.sh <path_to_test_dur_conf_file> <path_to_test_synth_conf_file>
```

