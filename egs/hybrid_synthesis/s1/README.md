# Hybrid speech synthesis

## ATTENTION: still in experimental phase...won't work straight-forward...requires lot of debugging!!

Hybrid speech synthesis is one of the main driving force behind most of the commercial systems that are present today. 

Festival offers a general framework for building speech synthesis systems as well as including examples of various modules. Multisyn is an open-source toolkit for building unit selection voice with any speech corpus. This post gives detailed instructions on how to use Multisyn to build an unit selection model and Festival for final waveform synthesis. 

## Tools required

1. [Speech tools](http://www.cstr.ed.ac.uk/projects/speech_tools)
2. [Festival](http://www.cstr.ed.ac.uk/projects/festival)
3. [Multisyn](http://www.cstr.ed.ac.uk/downloads/festival/multisyn_build)
4. [HTK](http://htk.eng.cam.ac.uk)

To build a new voice with Festival Multisyn, follow the step-by-step procedure given below:

## Step-by-step procedure

### Install tools

You might be familiar with most of these tools, but there are some differences in the way we setup these tools. 

- A version of [speech tools](http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-with-wrappers.tar.gz) with python wrappers has to be installed in order to work with Multisyn.
- Latest version of [Festival](http://104.131.174.95/downloads/tools/festival-2.4-current.tar.gz) has to be installed in order to use hybrid unit selection. 

Therefore, we recommend installing a fresh copy of these tools following the [scripts provided in Merlin](https://github.com/CSTR-Edinburgh/merlin/blob/master/tools/compile_unit_selection_tools.sh). 

To install speech tools, Festival and Multisyn:

```bash
bash compile_unit_selection_tools.sh
```

To install HTK:

```bash
bash compile_htk.sh
```

Make sure you install all these tools without any errors and check environment variables before proceeding further. 

### Demo data

At this point, make sure you have data ready:

- a [directory containing audio files](http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/) with file extension `.wav` 
- a [text file](http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/etc/txt.done.data) with transcriptions in the typical festival format.

For demo purpose, we use [SLT corpus from CMU Arctic Database](http://festvox.org/cmu_arctic/cmu_arctic/packed/cmu_us_slt_arctic-0.95-release.zip). 

### Setting up

The first step is to run setup as it creates directories and some text files for testing.

The next steps demonstrate on how to setup voice. 

```sh
./01_setup.sh conf/global_settings.cfg
```

You need to modify these params as per your own data.

### Build unit-selection model with Multisyn

Choose one of the lexicons:
1. cmulex
2. unilex-rpx
3. combilex-rpx

Choose gender:
1. 'm' for male
2. 'f' for female

If no arguments provided, the script uses default options: unilex and female assuming slt database

```sh
./02_build_unit_selection_model.sh
```

### Build parametric model with Merlin

```sh
./03_build_parametric_model.sh
```

### Build hybrid model

```sh
./04_build_hybrid_model.sh
```

### Synthesis with Festival

The below instructions are for Festival Multisyn voice:

```sh
$FESTDIR/bin/festival
```

Make festival speak "Hello world!" with new voice:

```sh
festival> (voice_cstr_edi_slt_multisyn)
festival> (SayText "Hello world!")
festival> (utt.save.wave (utt.synth (Utterance Text "Hello world!" )) "hello_world.wav")
```

For batch processing:

```sh
./05_run_hybrid_voice.sh <path_to_text_dir> <path_to_wav_dir>
```

For hybrid voice, please use the scm file in `scripts`.
