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

class LearningRate(object):

    def __init__(self):
        '''constructor'''    
        
    def get_rate(self):
        pass

    def get_next_rate(self, current_error):
        pass

class LearningRateConstant(LearningRate):

    def __init__(self, learning_rate = 0.08, epoch_num = 20):

        self.learning_rate = learning_rate
        self.epoch = 1
        self.epoch_num = epoch_num
        self.rate = learning_rate

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):

        if ( self.epoch >=  self.epoch_num):
            self.rate = 0.0
        else:
            self.rate = self.learning_rate
        self.epoch += 1

        return self.rate

class LearningRateExpDecay(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 min_derror_decay_start = 0.05, min_derror_stop = 0.05, init_error = 100,
                 decay=False, min_epoch_decay_start=15, zero_rate = 0.0):

        self.start_rate = start_rate
        self.init_error = init_error
        
        self.rate = start_rate
        self.scale_by = scale_by
        self.min_derror_decay_start = min_derror_decay_start
        self.min_derror_stop = min_derror_stop
        self.lowest_error = init_error
        
        self.epoch = 1
        self.decay = decay
        self.zero_rate = zero_rate

        self.min_epoch_decay_start = min_epoch_decay_start


    def get_rate(self):
        return self.rate  
    
    def get_next_rate(self, current_error):
        diff_error = 0.0
        diff_error = self.lowest_error - current_error
            
        if (current_error < self.lowest_error):
            self.lowest_error = current_error
    
        if (self.decay):
            if (diff_error < self.min_derror_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if ((diff_error < self.min_derror_decay_start) and (self.epoch > self.min_epoch_decay_start)):
                self.decay = True
                self.rate *= self.scale_by
            
        self.epoch += 1
        return self.rate


class LearningMinLrate(LearningRate):

    def __init__(self, start_rate = 0.08, scale_by = 0.5,
                 min_lrate_stop = 0.0002, init_error = 100,
                 decay=False, min_epoch_decay_start=15):

        self.start_rate = start_rate
        self.init_error = init_error

        self.rate = start_rate
        self.scale_by = scale_by
        self.max_epochs = max_epochs
        self.min_lrate_stop = min_lrate_stop
        self.lowest_error = init_error

        self.epoch = 1
        self.decay = decay
        self.min_epoch_decay_start = min_epoch_decay_start

    def get_rate(self):
        return self.rate

    def get_next_rate(self, current_error):
        diff_error = 0.0

        diff_error = self.lowest_error - current_error

        if (current_error < self.lowest_error):
            self.lowest_error = current_error

        if (self.decay):
            if (self.rate < self.min_lrate_stop):
                self.rate = 0.0
            else:
                self.rate *= self.scale_by
        else:
            if (self.epoch >= self.min_epoch_decay_start):
                self.decay = True
                self.rate *= self.scale_by

        self.epoch += 1
        return self.rate

class   ExpDecreaseLearningRate(object):
    def __init__(self, start_rate = 0.02, end_rate = 0.001, maximum_epoch = 5):
        self.start_rate = start_rate
        self.end_rate = end_rate
        self.maximum_epoch = maximum_epoch
        
        self.rate_diff = self.start_rate - self.end_rate
        
        self.decrease_ratio = numpy.zeros((1, maximum_epoch+1))
        for i in xrange(maximum_epoch):
            self.decrease_ratio[0, i+1] = maximum_epoch - i
            
        self.decrease_ratio = numpy.exp(self.decrease_ratio)
        self.decrease_ratio /= numpy.sum(self.decrease_ratio)    
        
        self.decrease_ratio[0, 0] = 1.0
        
    def get_rate(self, epoch):
        
        if epoch < 0:
            epoch = 0
            
        current_rate = self.end_rate
        if epoch <= self.maximum_epoch:
            current_rate = self.end_rate + self.decrease_ratio[0, epoch] * self.rate_diff

        return  float(current_rate)    
        
