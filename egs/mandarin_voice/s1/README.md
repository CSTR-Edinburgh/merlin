# Mandarin Voice

## To run_demo
```
bash run_demo.sh
```

## To train with your own dataset

(1) Create the following dir and copy your file to dir (suppose current dir is merlin/egs/mandarin_voice/s1/)

* database/wav 
* database/labels/label_phone_align 
* database/prompt-lab 
* copy your own Question file to merlin/misc/questions

(2) modify params as per your own data in 01_setup.sh file, especially

* Voice Name
* QuestionFile
* Labels_Type(phone_align or state_align)
* SamplingFreq
* Train
* Valid
* Test

default setting is 

* QuestionFile=questions-mandarin.hed
* Labels=phone_align
* SamplingFreq=16000
* Train=200
* Valid=25
* Test=25

(3) then run

```
./run_mandarin_voice.sh
```
