## Generic script for submitting any Theano job to GPU
# usage: submit.sh [scriptname.py script_arguments ... ]

src_dir=$(dirname $1)
gpu_id=$(python ${src_dir}/gpu_lock.py --id-to-hog)

if [ $gpu_id -gt -1 ]; then
    THEANO_FLAGS="cuda.root=/opt/6.5.19,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32,on_unused_input=ignore"
    export THEANO_FLAGS
    
    python $@
    
    python ${src_dir}/gpu_lock.py --free $gpu_id
else
    #echo 'Let us wait! No GPU is available!'
    
    ### run on CPU ###
    echo "Running on CPU..."
    python $@
fi
