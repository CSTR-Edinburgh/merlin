PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/


gpu_id=$(python ./gpu_lock.py --id-to-hog)


CONFIG="./configuration/myconfigfile.conf"
if [ $# -gt 0 ]; then
	CONFIG=$1
fi


if [ $gpu_id -gt -1 ]; then
    #THEANO_FLAGS="cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
    THEANO_FLAGS="cuda.root=/opt/6.5.19,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
        export THEANO_FLAGS
    
    echo $CONFIG
    python  ./run_dnn_hourly_check.py ${CONFIG}

    python ./gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'

fi
