The Neural Network (NN) based Speech Synthesis System
=====================================================
  
This repository contains the Neural Network (NN) based Speech Synthesis System  
developed at the Centre for Speech Technology Research (CSTR), University of 
Edinburgh. 


Step 0: download data from http://homepages.inf.ed.ac.uk/zwu2/dnn_baseline/dnn_baseline_practice.tar.gz and put it in your working directory

Step 1: Install Theano version 0.6 and above (http://deeplearning.net/software/theano/). You may need other third-party python libraries such as matplotlib.

Step 2: Modify the configuration file 'myconfigfile_mvn_DNN_basic.conf' to fit your working environment.
    a) Change the following settings:
            [Paths]
            work: <YOUR_OWN_WORKING_DIRECTORY>
            data: <WHERE_IS_YOUR_DATA>
            
            log_config_file: <CODE_DIRECTORY>/configuration/myloggingconfigfile.conf
    
            sptk: <WHERE_ARE_SPTK_BINARY_FILES>
            straight: <WHERE_ARE_STRAIGHT_BINARIES>
            
    b) By default, the program will run through the following processes if you keep them to be 'True'. You could run step by step by tuning on one process and tuning off all the other processes.
        NORMLAB  : True
        MAKECMP  : True
        NORMCMP  : True
        TRAINDNN : True
        DNNGEN   : True
        GENWAV   : False
        CALMCD   : True
    
    Note: current recipe required C-version STRAIGHT, which cannot be distributed. Please replace the GENWAV step by using your own vocoder.
    
Step 3: Create myloggingconfigfile.conf based on exampleloggingconfigfile.conf

Step 4: Run './submit.sh ./run_lstm.py <CONFIG FILE PATH>' script. It will lock a GPU automatically. If you do not want to lock GPU, please modify the ./run_dnn.sh script.

If everything goes smoothly, after 3 to 5 hours, the program will give objective results, and synthesised voices if you have your own vocoder.
    
Feel free to contact us if you have any questions.
