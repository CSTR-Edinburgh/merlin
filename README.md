The Neural Network (NN) based Speech Synthesis System
=====================================================
  
This repository contains the Neural Network (NN) based Speech Synthesis System  
developed at the Centre for Speech Technology Research (CSTR), University of 
Edinburgh. 


Step 0: download data from http://104.131.174.95/slt_arctic_demo.zip (will replace soon with doi) and put it in your working directory

Step 1: Install Theano version 0.6 and above (http://deeplearning.net/software/theano/). You may need other third-party python libraries such as matplotlib.

Step 2: ./setup.sh slt_arctic 

Step 3: ./prepare_config__files.sh configuration/merlin_voice_settings.cfg 
            
Step 4: Create logging_config.conf based on exampleloggingconfigfile.conf

Step 5: Run './submit.sh ./run_lstm.py configuration/duration_configfile.conf' script. 

Step 6: Run './submit.sh ./run_lstm.py configuration/acoustic_configfile.conf' script.
 
    a) It will lock a GPU automatically. If you do not want to lock GPU, please modify the ./submit.sh script.
    b) By default, the program will run through the following processes if you keep them to be 'True'. You could run step by step by tuning on one process and tuning off all the other processes.
        NORMLAB  : True
        MAKEDUR  : True
        MAKECMP  : True
        NORMCMP  : True
        TRAINDNN : True
        DNNGEN   : True
        GENWAV   : True
        CALMCD   : True
    
    Note: 
    - Current recipe requires C-version STRAIGHT, which cannot be distributed. 
    - Therefore, we support WORLD vocoder by default as a replacement for STRAIGHT.

If everything goes smoothly, after few minutes, the program will give objective results, and synthesised voices.
    
Feel free to contact us if you have any questions.
