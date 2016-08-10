PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/

gpu_id=$(python ./gpu_lock.py --id-to-hog)

echo $gpu_id



CONFIG=$1
INDIR=$2
OUTDIR=$3



if [ $gpu_id -gt -1 ]; then
    THEANO_FLAGS="cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
    export THEANO_FLAGS
    
    echo $CONFIG
    python  ./dnn_synth.py ${CONFIG} ${INDIR} ${OUTDIR}

    python ./gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'

fi
