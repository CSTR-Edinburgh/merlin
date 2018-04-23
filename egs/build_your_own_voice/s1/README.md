# Build your own voice


## Requirements

You need to have installed:
* [Merlin](https://github.com/CSTR-Edinburgh/merlin#installation)
* festival: ```bash tools/compile_other_speech_tools.sh```
* htk: ```bash tools/compile_htk.sh```


## Building Steps

To build your own voice, `cd egs/build_your_own_voice/s1` and follow the below steps:

### Setting up

The first step is to run setup as it creates directories and some text files for testing.

The next steps demonstrate on how to setup voice. 

```sh
./01_setup.sh my_voice
```

It also creates a global config file: `conf/global_settings.cfg`, where default settings are stored.
You need to modify these params as per your own data.

### Prepare labels

To prepare labels
```sh
./02_prepare_labels.sh <path_to_wav_dir> <path_to_text_dir> <path_to_labels_dir>
```

### Prepare acoustic features
 
To prepare acoustic features
```sh
./03_prepare_acoustic_features.sh <path_to_wav_dir> <path_to_feat_dir>
```

### Prepare config files

At this point, we have to prepare two config files to train DNN models
- Acoustic Model
- Duration Model

To prepare config files:
```sh
./04_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

### Train duration model

To train duration model:
```sh
./05_train_duration_model.sh <path_to_duration_conf_file>
```

### Train acoustic model

To train acoustic model:
```sh
./06_train_acoustic_model.sh <path_to_acoustic_conf_file>
```
### Synthesize speech

To synthesize speech:
```sh
./07_run_merlin.sh <path_to_text_dir> <path_to_test_dur_conf_file> <path_to_test_synth_conf_file>
```

