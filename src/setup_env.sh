# Setup environment variables

export PYTHONBIN="python"
# To avoid python changing the sys.path variable, add an -S
#export PYTHONBIN='python -S'

# Maybe the following variables need to be setup
#export PYTHONPATH=/some/lab/specific/path:PYTHONPATH
#export PATH=/path:PATH
#export LD_LIBRARY_PATH=/anotherpath:LD_LIBRARY_PATH

# Basic Theano flags
MERLIN_THEANO_FLAGS="cuda.root=/opt/6.5.19,floatX=float32,on_unused_input=ignore"
export MERLIN_THEANO_FLAGS
	

# Log the resulting setup =====================================================

# Print some basic informations
echo Architecture: `uname -m`
echo Distribution: `lsb_release -d |sed 's/Description://'`
echo HOSTNAME=$HOSTNAME
echo USER=$USER
echo " "

# Print the Merlin's related environment variables
#echo PATH: $PATH
echo PATH:
echo "$PATH" |tr : '\n' |sed '/^$/d' |sed 's/^/    /'
#echo LD_LIBRARY_PATH: $LD_LIBRARY_PATH
echo LD_LIBRARY_PATH:
echo "$LD_LIBRARY_PATH" |tr : '\n' |sed '/^$/d' |sed 's/^/    /'
#echo PYTHONPATH: $PYTHONPATH
echo PYTHONPATH:
echo "$PYTHONPATH" |tr : '\n' |sed '/^$/d' |sed 's/^/    /'
echo PYTHONBIN: $PYTHONBIN
#echo MERLIN_THEANO_FLAGS: $MERLIN_THEANO_FLAGS
echo MERLIN_THEANO_FLAGS:
echo "$MERLIN_THEANO_FLAGS" |tr , '\n' |sed '/^$/d' |sed 's/^/    /'

echo " "


