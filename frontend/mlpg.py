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


## use theano to benefit from GPU computation
from theano import tensor as T
import  theano
    
import numpy
from numpy import dot
import logging


class MLParameterGeneration(object):
    def __init__(self, delta_win = [-0.5, 0.0, 0.5], acc_win = [1.0, -2.0, 1.0]):
        self.delta_win = delta_win
        self.acc_win   = acc_win
        ###assume the delta and acc windows have the same length
        self.win_length = int(len(delta_win)/2)

    def build_theano_function_wdw(self):

        W_static = T.matrix('W_static')
        W_delta  = T.matrix('W_delta')
        W_acc    = T.matrix('W_acc')
        D_static = T.matrix('D_static')
        D_delta  = T.matrix('D_delta')
        D_acc    = T.matrix('D_acc')

        WDW = T.dot(T.dot(W_static.T, D_static), W_static) + T.dot(T.dot(W_delta.T, D_delta), W_delta) + T.dot(T.dot(W_acc.T, D_acc), W_acc)
        
        fn = theano.function(inputs=[W_static, W_delta, W_acc, D_static, D_delta, D_acc], outputs=WDW)

        return  fn

    def build_theano_function_wdu(self):

        W_static = T.matrix('W_static')
        W_delta  = T.matrix('W_delta')
        W_acc    = T.matrix('W_acc')
        D_static = T.matrix('D_static')
        D_delta  = T.matrix('D_delta')
        D_acc    = T.matrix('D_acc')
        U_static = T.matrix('U_static')
        U_delta  = T.matrix('U_delta')
        U_acc    = T.matrix('U_acc')

        WDU = T.dot(T.dot(W_static.T, D_static), U_static) + T.dot(T.dot(W_delta.T, D_delta), U_delta) + T.dot(T.dot(W_acc.T, D_acc), U_acc)
        
        fn = theano.function(inputs=[W_static, W_delta, W_acc, D_static, D_delta, D_acc, U_static, U_delta, U_acc], outputs=WDU)

        return  fn

    def generation(self, features, covariance, static_dimension):
        '''
        plan: use theano to do the parameter generation to benefit from GPU
        '''

        logger = logging.getLogger('param_generation')
        logger.debug('starting MLParameterGeneration.generation')
        
        frame_number = features.shape[0]

        gen_parameter = numpy.zeros((frame_number, static_dimension))

        W_static, W_delta, W_acc = self.prepare_window(frame_number)
        
        WT_static = numpy.transpose(W_static)
        WT_delta  = numpy.transpose(W_delta)
        WT_acc    = numpy.transpose(W_acc)

        fn_wdw = self.build_theano_function_wdw()
        fn_wdu = self.build_theano_function_wdu()

        for d in xrange(static_dimension):
            logger.debug('static dimension %3d of %3d' % (d+1,static_dimension) )
            
            D_static = self.prepare_D(frame_number, covariance[d, 0])
            D_delta  = self.prepare_D(frame_number, covariance[static_dimension + d, 0])
            D_acc    = self.prepare_D(frame_number, covariance[2*static_dimension + d, 0])

            U_static = self.prepare_U(frame_number, features[:, d:d+1])
            U_delta  = self.prepare_U(frame_number, features[:, static_dimension + d:static_dimension + d + 1])
            U_acc    = self.prepare_U(frame_number, features[:, 2*static_dimension + d:2*static_dimension + d + 1])

#            WDW = dot(dot(WT_static, D_static), W_static) + dot(dot(WT_delta, D_delta), W_delta) + dot(dot(WT_acc, D_acc), W_acc)
#            WDU = dot(dot(WT_static, D_static), U_static) + dot(dot(WT_delta, D_delta), U_delta) + dot(dot(WT_acc, D_acc), U_acc)
#            temp_obs = dot(numpy.linalg.inv(WDW), WDU)
            

            WDW = fn_wdw(W_static, W_delta, W_acc, D_static, D_delta, D_acc)
            WDU = fn_wdu(W_static, W_delta, W_acc, D_static, D_delta, D_acc, U_static, U_delta, U_acc)
            ###only theano-dev version support matrix inversion
            temp_obs = dot(numpy.linalg.inv(WDW), WDU)
            
            gen_parameter[0:frame_number, d] = temp_obs[self.win_length:frame_number+self.win_length, 0]

        return  gen_parameter


    def prepare_window(self, frame_number):
        win_length = self.win_length

        w_static = numpy.zeros((frame_number+win_length*2, frame_number+win_length*2), dtype=theano.config.floatX)
        w_delta  = numpy.zeros((frame_number+win_length*2, frame_number+win_length*2), dtype=theano.config.floatX)
        w_acc    = numpy.zeros((frame_number+win_length*2, frame_number+win_length*2), dtype=theano.config.floatX)

        for i in xrange(frame_number+win_length*2):
            w_static[i, i] = 1.0
            w_delta[i, i]  = self.delta_win[win_length]
            w_acc[i, i]    = self.acc_win[win_length]

            for j in xrange(win_length):
                if i - j > 0:
                    w_delta[i, i-j-1] = self.delta_win[win_length-j-1]
                    w_acc[i, i-j-1]   = self.acc_win[win_length-j-1]
                
                if i + j + 1 < frame_number+win_length*2:
                    w_delta[i, i+j+1] = self.delta_win[win_length+j+1]
                    w_acc[i, i+j+1]   = self.acc_win[win_length+j+1]

        return  w_static, w_delta, w_acc

    def prepare_D(self, frame_number, D_value):
        win_length = self.win_length
        D_matrix = numpy.zeros((frame_number+win_length*2, frame_number+win_length*2), dtype=theano.config.floatX)

        for i in xrange(win_length):
            D_matrix[i, i] = 1.0
            D_matrix[frame_number+win_length+i, frame_number+win_length+i] = 1.0

        for i in xrange(frame_number):
            D_matrix[win_length+i, win_length+i] = 1.0 / D_value

        return  D_matrix

    def prepare_U(self, frame_number, U_vector):

        win_length = self.win_length

        U_expanded = numpy.zeros((frame_number+win_length*2, 1), dtype=theano.config.floatX)

        U_expanded[win_length:frame_number+win_length, :] = U_vector

        return  U_expanded


