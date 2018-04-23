# Voice Conversion Using MagPhase Vocoder

To manipulate source speaker's voice to sound like target without changing language content. 

MagPhase vocoder for VC has the advantage that in addition to using magnitude spectra derived features, it makes possible to map **phase features** from one speaker to another.

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
 
To convert source voice to target voice, `cd egs/voice_conversion/s2` and follow the below steps:

## Run with demo data

To do voice conversion on demo data, you can simply run:
```sh
./run_demo_vc.sh
```
Which will download the [data](http://104.131.174.95/downloads/voice_conversion/) and run the whole voice conversion recipe for you. The data consists of:

* Source: bdl (300 utterances)
* Target: slt (300 utterances)

However, we recommend using step-by-step procedure to correct any errors if raised:

### I. Setting up

The first step is to run setup as it creates directories.

The next steps demonstrate on how to setup voice. 

```sh
./01_setup.sh speakerA speakerB
```

It also creates a global config file: `conf/global_settings.cfg`, where default settings are stored.
You need to modify these params as per your own data.

### II. Prepare acoustic features

To prepare acoustic features
```sh
./02_prepare_acoustic_features.sh <path_to_wav_dir> <path_to_feat_dir>
```

You have to run this script twice, for speakerA and speakerB

### III. Align source features with target

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

### IV. Prepare config files

At this point, we have to prepare two config files for voice conversion
- Source acoustic model
- Source2Target acoustic Model

To prepare config files:
```sh
./04_prepare_conf_files.sh conf/global_settings.cfg
```
Four config files will be generated: two for training, and two for testing. 

### V. Prepare source acoustic features

To prepare acoustic features for SpeakerA
```sh
./05_train_acoustic_model.sh <path_to_acoustic_source_conf_file>
```

### VI. Create a symbolic link for source acoustic features

At this point, we have to create a symbolic link for source features in the main voice directory.

To prepare symbolic link for source features
```sh
./scripts/create_symbolic_link.sh
```

Input dimension for source features is computed based on sampling rate provided in global conf file. 

### VII. Train acoustic model for voice conversion

To train acoustic model:
```sh
./05_train_acoustic_model.sh <path_to_acoustic_voice_conf_file>
```

### VIII. Voice conversion from source to target

To transform voice from speakerA to speakerB:
```sh
./06_run_merlin_vc.sh <path_to_src_wav_dir> <path_to_test_source_conf_file> <path_to_test_synth_conf_file>
```

## TODO

* Constant frame rate support.
* Selectable phase and magnitude dimensions.
* Selectable postfilter.

