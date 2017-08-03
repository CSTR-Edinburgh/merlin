# Voice Conversion

To manipulate source speaker's voice to sound like target without changing language content. 

## Dependency tools

Along with Merlin, we need to install few other speech tools in order to run voice conversion. 
- [speech tools](http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz)
- [festival](http://www.cstr.ed.ac.uk/downloads/festival/2.4/festival-2.4-release.tar.gz)
- [festvox](http://festvox.org/festvox-2.7/festvox-2.7.0-release.tar.gz)

```sh
bash merlin/tools/compile_other_speech_tools.sh 
```

All these tools are required for only one task i.e., dynamic time warping (DTW) to create parallel data. 
You can check this [tutorial](http://speech.zone/exercises/dtw-in-python) for DTW implementation. 

To convert source voice to target voice, `cd egs/voice_conversion/s1` and follow the below steps:

## Demo data

You can download the data from [here](http://104.131.174.95/downloads/voice_conversion/).
- Source: bdl (300 utterances)
- Target: slt (300 utterances)

To run voice conversion on demo data
```sh
./run_demo_vc.sh
```

However, we recommend using step-by-step procedure to correct any errors if raised. 

## Setting up

The first step is to run setup as it creates directories.

The next steps demonstrate on how to setup voice. 

```sh
./01_setup.sh speakerA speakerB
```

It also creates a global config file: `conf/global_settings.cfg`, where default settings are stored.
You need to modify these params as per your own data.

## Prepare acoustic features

To prepare acoustic features
```sh
./02_prepare_acoustic_features.sh <path_to_wav_dir> <path_to_feat_dir>
```

You have to run this script twice, for speakerA and speakerB

## Align source features with target

For voice conversion, we require parallel sentences for training. For this, we use dynamic-time-warping 
to align source features with target. 

To align source features with target
```sh
./03_align_src_with_target.sh <path_to_src_feat_dir> <path_to_tgt_feat_dir> <path_to_src_align_dir>
```

## Prepare config files

At this point, we have to prepare two config files for voice conversion
- Source acoustic model
- Source2Target acoustic Model

To prepare config files:
```sh
./04_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

## Prepare source acoustic features 

To prepare acoustic features for SpeakerA
```sh
./05_train_acoustic_model.sh <path_to_acoustic_source_conf_file>
```

## Create a symbolic link for source acoustic features 

At this point, we have to create a symbolic link for source features in the main voice directory.

To prepare symbolic link for source features
```sh
./scripts/create_symbolic_link.sh
```

Input dimension for source features is computed based on sampling rate provided in global conf file. 

## Train acoustic model for voice conversion

To train acoustic model:
```sh
./05_train_acoustic_model.sh <path_to_acoustic_voice_conf_file>
```

## Voice conversion from source to target

To transform voice from speakerA to speakerB:
```sh
./06_run_merlin_vc.sh <path_to_src_wav_dir> <path_to_test_source_conf_file> <path_to_test_synth_conf_file>
```

