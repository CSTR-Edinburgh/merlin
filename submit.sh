PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/

## Generic script for submitting any Theano job to GPU
# usage: submit.sh [scriptname.py script_arguments ... ]

gpu_id=$(python ./gpu_lock.py --id-to-hog)



if [ $gpu_id -gt -1 ]; then
    #THEANO_FLAGS="cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
    THEANO_FLAGS="cuda.root=/opt/6.5.19,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32,on_unused_input=ignore"
    export THEANO_FLAGS
    
    python $@
    
    python ./gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'

fi
