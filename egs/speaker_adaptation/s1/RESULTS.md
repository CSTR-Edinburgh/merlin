VCTK corpus
=================

Average vocie model (AVM) data (demo): 
--------------------------------
Speakers: 6 female + 3 male, from `p225` to `p233` (3358 utterances) <br/>
Data distribution: Train: 3258; Valid: 50; Test: 50;

Speaker p235 voice model
-------------------------
Speaker: Female (357 utterances) <br/>
Data distribution: Train: 307; Valid: 25; Test: 25;

RESULTS
=======

Baseline results of AVM from demo data
-------------------------------
Duration model tains in about 20 minutes on GPU (Nvidia Titan)

Objective scores from duration model: <br/>

Labels: state_align; Network: [416->5] LR 0.002 [6 TANH] [6*1024]; <br/>
Develop -- RMSE: 5.751 frames/phoneme; CORR: 0.791; <br/>
Test  -- RMSE: 5.502 frames/phoneme; CORR: 0.808;

Acoustic model trains in about 1 hour 30 minutes on GPU (Nvidia Titan)

Objective scores from acoustic model: <br/> 

Labels: state_align; Network: [425->199], LR 0.002 [4 TANH] [4*512]; <br/>
Develop -- MCD: 6.145 dB; BAP: 0.345 dB; F0:- RMSE: 47.039 Hz; CORR: 0.678; VUV: 8.771%  <br/>
Test  -- MCD: 6.097 dB; BAP: 0.329 dB; F0:- RMSE: 46.723 Hz; CORR: 0.627; VUV: 8.474%


Baseline results of speaker `p234` from full data
-------------------------------------------------
Duration model trains in about 5 minutes
Objective scores from duration model: <br/>

Labels: state_align; Network: [416->5] LR 0.002 [6 TANH] [6*1024]; <br/>
Develop -- RMSE: 6.145 frames/phoneme; CORR: 0.664; <br/>
Test  -- RMSE: 7.604 frames/phoneme; CORR: 0.687;

Acoustic model trains in about 10 minutes
Objective scores from acoustic model: <br/> 

Labels: state_align; Network: [425->199], LR 0.002 [6 TANH] [6*1024]; <br/>
Develop: DNN -- MCD: 5.476 dB; BAP: 0.401 dB; F0:- RMSE: 15.202 Hz; CORR: 0.537; VUV: 13.190%
Test   : DNN -- MCD: 5.441 dB; BAP: 0.407 dB; F0:- RMSE: 17.097 Hz; CORR: 0.542; VUV: 16.853%


Adapt speaker `p234` on the average voice model
-----------------------------------------------
Here we used the fine-tune method for the adaptation.

Duration model adapts in about 5 minutes
Objective scores from duration model: <br/>

Labels: state_align; Network: [416->5] LR 0.002 [6 TANH] [6*1024]; <br/>
Develop -- RMSE: 5.214 frames/phoneme; CORR: 0.797; <br/>
Test  -- RMSE: 5.483 frames/phoneme; CORR: 0.851;

Acoustic model adapts in about 10 minutes
Objective scores from acoustic model: <br/> 

Labels: state_align; Network: [425->199], LR 0.002 [6 TANH] [6*1024]; <br/>
Develop: DNN -- MCD: 5.240 dB; BAP: 0.383 dB; F0:- RMSE: 15.034 Hz; CORR: 0.524; VUV: 12.280%  <br/>
Test   : DNN -- MCD: 5.213 dB; BAP: 0.389 dB; F0:- RMSE: 18.190 Hz; CORR: 0.473; VUV: 16.459%
