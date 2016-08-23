PYTHONPATH=${PYTHONPATH}:/afs/inf.ed.ac.uk/group/project/dnn_tts/tools/site-packages/

gpu_id=$(python ./gpu_lock.py --id-to-hog)

echo $gpu_id



CONFIG=$1
INDIR=$2
OUTDIR=$3
TOKEN_XPATH=$4
INDEX_ATTRIB_NAME=$5
SYNTH_MODE=$6
PROJECTION_END=$7


CMP_DIR=' '   ## default
if [ $# -eq 8 ] ; then 
    CMP_DIR=$8
fi






if [ $gpu_id -gt -1 ]; then
    THEANO_FLAGS="cuda.root=/opt/cuda-5.0.35,mode=FAST_RUN,device=gpu$gpu_id,floatX=float32"
    export THEANO_FLAGS
    
    echo $CONFIG
    python  ./dnn_synth_PROJECTION.py ${CONFIG} ${INDIR} ${OUTDIR} ${TOKEN_XPATH} ${INDEX_ATTRIB_NAME} ${SYNTH_MODE} ${PROJECTION_END} ${CMP_DIR}

    python ./gpu_lock.py --free $gpu_id
else
    echo 'Let us wait! No GPU is available!'

fi
