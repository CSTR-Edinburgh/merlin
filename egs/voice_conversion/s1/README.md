# Voice Conversion

To manipulate source speaker's voice to sound like target without changing language content. 

## Install Merlin

Before proceeding any further, you first [install Merlin](https://github.com/CSTR-Edinburgh/merlin#installation) and then run the below steps.

## Dependency tools

Along with Merlin, we need to install few other speech tools in order to run voice conversion. 
- [speech tools](http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz)
- [festvox](http://festvox.org/festvox-2.7/festvox-2.7.0-release.tar.gz)

```sh
bash merlin/tools/compile_other_speech_tools.sh 
```

All these tools are required for only one task i.e., dynamic time warping (DTW) to create parallel data. 
You can check this [tutorial](http://speech.zone/exercises/dtw-in-python) for DTW implementation.

As an alternative, [fastdtw](https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/voice_conversion/dtw_aligner.py) from python bindings can also be used.  
Please check [step 3](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/voice_conversion/s1/README.md#align-source-features-with-target) for its usage.
 
To convert source voice to target voice, `cd egs/voice_conversion/s1` and follow the below steps:

## Voice conversion challenge 2016 data

Now, you can run Merlin voice conversion using VC2016 data. 

To download the data:
```sh
./scripts/download_vcc2016_data.sh
```

To run voice conversion between any source-target pair, give the speaker names as arguments:
```sh
./run_vcc2016_benchmark.sh [SOURCE_SPEAKER] [TARGET_SPEAKER]
```

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

For voice conversion, we require parallel sentences for training. For this, we use dynamic-time-warping from Festvox 
to align source features with target. 

To align source features with target
```sh
./03_align_src_with_target.sh <path_to_src_feat_dir> <path_to_tgt_feat_dir> <path_to_src_align_dir>
```

Alternatively, [fastdtw](https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/voice_conversion/dtw_aligner.py) from python bindings can also be used.  

```bash
pip install fastdtw
```

To use fastdtw, replace `dtw_aligner_festvox.py` with `dtw_aligner.py` at line number 60 in `03_align_src_with_target.sh`.

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

