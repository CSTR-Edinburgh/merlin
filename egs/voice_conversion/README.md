# About voice conversion

Voice conversion aims at transforming the characteristics of a speech
signal uttered by a source speaker in such a way that the transformed
speech sounds like the target speaker. Such a conversion requires 
transformation of spectral and prosody features. 

Most of the current voice conversion techniques need a parallel
database where the source and target speakers record the
same set of utterances. 

CMU ARCTIC databases consists of 7 speakers with each speaker recording 
a set of 1132 phonetically balanced utterances. Therefore, it is an ideal 
choice to perform voice conversion experiments. 

Each subdirectory of this directory contains the
scripts for a sequence of experiments.

- s1: To run voice conversion with either world or straight vocoder 

## Demo data

You can download the data from [here](http://104.131.174.95/downloads/voice_conversion/).
- Source: bdl (300 utterances)
- Target: slt (300 utterances)

# About the Arctic corpus

The CMU ARCTIC databases were constructed at the Language Technologies Institute at Carnegie Mellon University as phonetically balanced, US English single speaker databases designed for unit selection speech synthesis research.

The databases consist of around 1150 utterances carefully selected from out-of-copyright texts from Project Gutenberg. The databses include US English male (bdl) and female (slt) speakers (both experinced voice talent) as well as other accented speakers.

For more details, please visit [cmu arctic page](http://www.festvox.org/cmu_arctic/).


