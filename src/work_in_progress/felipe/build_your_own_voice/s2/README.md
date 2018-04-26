# Build your own voice using the MagPhase vocoder


## Requirements

You need to have installed:
* [Merlin](https://github.com/CSTR-Edinburgh/merlin#installation)
* festival: ```bash tools/compile_other_speech_tools.sh```
* htk: ```bash tools/compile_htk.sh```

## Building Steps

### Generating state-aligned label files
1. Run the recipe **egs/build_your_own_voice/s1** until the step **02_prepare_labels.sh**

    As a result, you will have state aligned label files for your data.

### Acoustic Modelling and Synthesis

2. Edit the script **egs/build_your_own_voice/s2/build_voice.py** according to your data.

3. Run the script:
    ```
    python build_voice.py
    ```

