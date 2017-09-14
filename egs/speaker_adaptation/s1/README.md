# Speaker adaptation

This directory contains required files to build a speaker adaptation system. The VCTK-Corpus was used for the experiments.
1) Build an average voice model (AVM) over multiple speakers
2) Build a stand alone model of adapt speaker
3) Adapt the AVM for the adapt speaker 

## Download the data
The data is available for free to download. It comprises of a total 108 speakers of which 47 are male speakers. To download and extract the data:

```sh
./download_data.sh
```
Note: the size of tar file is around 11GB and afer extraction it will be around 16GB. Thus you need atleast 27GB of free space.

# Build an average voice model (AVM)
For Demo purpose we are building the AVM over 9 speakers (3 male and 6 female).

## Setting up

The first step is to run setup as it creates directories and some text files for testing.

The next steps demonstrate on how to setup voice. 

```sh
./01_setup.sh vctk_avm
```

It also creates a global config file: `conf/global_settings.cfg`, where default settings are stored. You need to modify these params as per the data.

## Prepare labels

To prepare labels
```sh
./02_prepare_labels.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir>
```

## Prepare acoustic features
 
To prepare acoustic features
```sh
./03_prepare_acoustic_features.sh <path_to_wav_dir> <path_to_feat_dir>
```

## Prepare config files

At this point, we have to prepare two config files to train DNN models
- Acoustic Model
- Duration Model

To prepare config files:
```sh
./04_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

## Train duration model

To train duration model:
```sh
./05_train_duration_model.sh <path_to_duration_conf_file>
```

## Train acoustic model

To train acoustic model:
```sh
./06_train_acoustic_model.sh <path_to_acoustic_conf_file>
```
## Synthesize speech

To synthesize speech:
```sh
./07_run_merlin.sh <path_to_text_dir> <path_to_test_dur_conf_file> <path_to_test_synth_conf_file>
```

# Build voice of speaker p234 without adaptation
Just follow the above steps to creat the voice of p234


# Adapt the speaker `p234` on the AVM

## Setting up
To setup the data and directories

```sh
./08_setup_adapt.sh <voice_name> <average_duration_model> <average_acoustic_model> <adaptation_method>
```

It creates a global config file: `conf/global_settings_adapt.cfg`, where default settings are stored. You need to modify these params as per the data.

## Prepare labels

To prepare labels
```sh
./09_prepare_labels_adapt.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir>
```

## Prepare acoustic features
 
To prepare acoustic features
```sh
./10_prepare_acoustic_features.sh <path_to_wav_dir> <path_to_feat_dir>
```

## Prepare config files

At this point, we have to prepare two config files to adapt DNN models
- Acoustic Model
- Duration Model

To prepare config files:
```sh
./11_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

## Train duration model

To train duration model:
```sh
./12_adapt_duration_model.sh <path_to_duration_conf_file>
```

## Train acoustic model

To train acoustic model:
```sh
./13_adapt_acoustic_model.sh <path_to_acoustic_conf_file>
```
## Synthesize speech
