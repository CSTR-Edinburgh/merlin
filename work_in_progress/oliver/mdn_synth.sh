PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/


## routine to avoid using GPU card 0 on dechmont:-
init_gpu_id=$(python ./gpu_lock.py --id-to-hog)
echo $init_gpu_id
if [ $init_gpu_id -eq 0 ] ; then
    echo 'dont want to use 0th GPU card on dechmont -- get another....'
    gpu_id=$(python ./gpu_lock.py --id-to-hog)
    echo $gpu_id
    sleep 1
    python ./gpu_lock.py --free $init_gpu_id
    echo 'freed 0th card'
else
    echo 'use this card'
    gpu_id=$init_gpu_id
fi



CONFIG=$1
INDIR=$2
OUTDIR=$3



if [ $gpu_id -gt -1 ]; then
    THEANO_FLAGS="cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
    export THEANO_FLAGS
    
    echo $CONFIG
    python  ./mdn_synth.py ${CONFIG} ${INDIR} ${OUTDIR}

    python ./gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'

fi
