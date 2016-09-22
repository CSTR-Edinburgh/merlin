################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################

import numpy
import theano
import theano.tensor as T

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import  matplotlib.pyplot as plt
from collections import OrderedDict

def compile_RPROP_train_function(model, gparams, params_to_update=[]):

    if params_to_update == []:  ## then update all by default
        params_to_update = range(len(gparams))

    
    assert model.use_rprop in [2,4], 'RPROP version %s not implemented'%(use_rprop)
    ## 0 = don't use RPROP
    ## 1, 2, 3, 4: Rprop+ Rprop- iRprop+ iRprop-    
    ##    in Igel 2003 'Empirical evaluation of the improved Rprop learning algorithms'

    ## It would be easier to follow if these things were defined in __init__, but
    ## they are here to keep all RPROP-specific stuff in one place. 
    ## Also, make some or all
    ## rprop_init_update is configured during __init__, all of the others are hardcoded here
    ## for now:-
    model.eta_plus = 1.2
    model.eta_minus = 0.5
    model.max_update = 50.0
    model.min_update = 0.0000001
           
    model.previous_gparams = []
    model.update_values = []
    
    model.update_change_DEBUG = []
    
    for (i, weights) in enumerate(model.params):
        model.previous_gparams.append(theano.shared(value = numpy.zeros((numpy.shape(weights.get_value())),
                        dtype=theano.config.floatX), name='pg_%s'%(i)))
        model.update_values.append(theano.shared(value = 
                (numpy.ones(numpy.shape(weights.get_value()), \
                        dtype=theano.config.floatX) * model.rprop_init_update ), name='uv_%s'%(i)))
                                                                                                                                     
        model.update_change_DEBUG.append(theano.shared(value = numpy.zeros((numpy.shape(weights.get_value())),
                        dtype=theano.config.floatX), name='pcd_%s'%(i)))


    if model.use_rprop in [2,4]:
    
        updates = OrderedDict()

        for (i, (prev_gparam, gparam, update_step, param))  in enumerate(zip(model.previous_gparams, gparams, \
                                                        model.update_values, model.params)):
            if i in params_to_update:                                                             
                ## first update update_values:
                sign_change_test = prev_gparam * gparam
                increase_update_size = T.gt(sign_change_test, 0.0) * model.eta_plus
                decrease_update_size = T.lt(sign_change_test, 0.0) * model.eta_minus
                retain_update_size   = T.eq(sign_change_test, 0.0)
                update_changes = increase_update_size + decrease_update_size + retain_update_size
                new_update_step = update_step * update_changes
                ## apply floor/ceiling to updates:
                new_update_step = T.minimum(model.max_update, T.maximum(model.min_update, new_update_step)) 
                updates[update_step] = new_update_step
    
                if model.use_rprop == 4:
                    ## zero gradients where sign changed: reduce step size but don't change weight
                    gparam = gparam * (T.gt(sign_change_test, 0.0) + T.eq(sign_change_test, 0.0))
    
                ## then update params:
                updates[param] = param - T.sgn(gparam) * new_update_step
                        
                ## store previous iteration gradient to check for sign change in next iteration:
                updates[prev_gparam] = gparam                
             
                updates[model.update_change_DEBUG[i]] = param  # gparam # sign_change_test #  update_changes    # 
    
    else:
        sys.exit('RPROP version %s not implemented'%(model.use_rprop))   
        
    return updates   
    
    
def check_rprop_values(model):
    print '=== Update steps: ==='
    for (i, update_step) in enumerate(model.update_values):
        print '   param no. %s'%(i)
        print get_stats(update_step)
        v = update_step.get_value()
        if len(v.shape) == 2:
            print v[:4, :4]
        else:
            print v[:4]
        print '   Update changes:--'
        u = model.update_change_DEBUG[i].get_value()
        if len(u.shape) == 2:
            print u[:4, :4]
        else:
            print u[:4]
            
                    
def get_stats(theano_shared_params):
    vals = theano_shared_params.get_value()
    #m,n = numpy.shape(vals)
    print '   shape, minm max, mean, 5th and 95th percentile'
    print '   %s %s %s %s %s %s'%(numpy.shape(vals),vals.min(), vals.max(),vals.mean(), numpy.percentile(vals, 5), numpy.percentile(vals, 95))

## This is generic, and not specific to RPROP:
def plot_weight_histogram(model, outfile, lower=-0.25, upper=0.25):
    n = len(model.params)
    plt.clf()
    for (i, theano_shared_params) in enumerate(model.params):
        weights = theano_shared_params.get_value()
        values = weights.flatten()
        plt.subplot(n,1,i+1)
        frame = plt.gca()
        frame.axes.get_yaxis().set_ticks([])
        if i != n-1:  ## only keep bottom one
            frame.axes.get_xaxis().set_ticks([])
        plt.hist(values, 100)
        plt.xlim(lower, upper)
        print '   param no. %s'%(i)
        print get_stats(theano_shared_params)
    plt.savefig(outfile)
    print 'Made plot %s'%(outfile)
    
    
    
