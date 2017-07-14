#!/bin/bash

# "queue.pl" uses qsub.  The options to it are
# options to qsub.  If you have GridEngine installed,
# change this to a queue you have access to.
# Otherwise, use "run.pl", which will run jobs locally
# (make sure your --num-jobs options are no more than
# the number of cpus on your machine.

#a) Sun grid options (IDIAP)
# ATTENTION: Do that in your shell: SETSHELL grid
export cuda_cmd="queue.pl -l gpu"
export cuda_short_cmd="queue.pl -l sgpu"
#export cuda_cmd="queue.pl -l q1d,hostname=dynamix03"
#export cuda_cmd="..."

#b) BUT cluster options
#export cuda_cmd="queue.pl -q long.q@@pco203 -l gpu=1"
#export cuda_cmd="queue.pl -q long.q@pcspeech-gpu"

#c) run it locally...
#export cuda_cmd=run.pl
#export cuda_short_cmd=$cuda_cmd
