# export PYTHONPATH=<some lab-specific python path here>:${PYTHONPATH}

# Basic Theano flags
MERLIN_THEANO_FLAGS="cuda.root=/opt/6.5.19,floatX=float32,on_unused_input=ignore"
export MERLIN_THEANO_FLAGS
