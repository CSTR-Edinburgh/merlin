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


from io_funcs.htk_io import HTK_Parm_IO
from io_funcs.binary_io import BinaryIOCollection
import numpy
import logging

class CMPNormalisation(object):
    def __init__(self, mgc_dim=0, bap_dim=0, lf0_dim = 0):
        self.mgc_dim = mgc_dim * 3
        self.bap_dim = bap_dim * 3
        self.lf0_dim = lf0_dim * 3
        
    def load_cmp_file(self, file_name):
        
        logger = logging.getLogger("acoustic_norm")
        
        htk_reader = HTK_Parm_IO()
        htk_reader.read_htk(file_name)
        
        cmp_data = htk_reader.data

        mgc_data = cmp_data[:, 0:self.mgc_dim]

        # this only extracts the static lf0 because we need to interpolate it, then add deltas ourselves later
        lf0_data = cmp_data[:, self.mgc_dim]
        
        bap_data = cmp_data[:, self.mgc_dim+self.lf0_dim:self.mgc_dim+self.lf0_dim+self.bap_dim]

        logger.debug('loaded %s of shape %s' % (file_name, cmp_data.shape))
        logger.debug('  with: %d mgc + %d lf0 + %d bap = %d' % (self.mgc_dim,self.lf0_dim,self.bap_dim,self.mgc_dim+self.lf0_dim+self.bap_dim))

        assert( (self.mgc_dim+self.lf0_dim+self.bap_dim) == cmp_data.shape[1])

        return  mgc_data, bap_data, lf0_data

    def interpolate_f0(self, data):
        
        data = numpy.reshape(data, (data.size, 1))
        
        vuv_vector = numpy.zeros((data.size, 1))
        vuv_vector[data > 0.0] = 1.0
        vuv_vector[data <= 0.0] = 0.0        

        ip_data = data        

        frame_number = data.size
        last_value = 0.0
        for i in xrange(frame_number):
            if data[i] <= 0.0:
                j = i+1
                for j in range(i+1, frame_number):
                    if data[j] > 0.0:
                        break
                if j < frame_number-1:
                    if last_value > 0.0:
                        step = (data[j] - data[i-1]) / float(j - i)
                        for k in range(i, j):
                            ip_data[k] = data[i-1] + step * (k - i + 1)
                    else:
                        for k in range(i, j):
                            ip_data[k] = data[j]
                else:
                    for k in range(i, frame_number):
                        ip_data[k] = last_value
            else:
                ip_data[i] = data[i]
                last_value = data[i]

        return  ip_data, vuv_vector

    def compute_delta(self, vector, delta_win):
#        delta_win = [-0.5, 0.0, 0.5]
#        acc_win   = [1.0, -2.0, 1.0]
        
        frame_number = vector.size
        win_length = len(delta_win)
        win_width = int(win_length/2)
        temp_vector = numpy.zeros((frame_number + 2 * win_width, 1))
        delta_vector = numpy.zeros((frame_number, 1))

        temp_vector[win_width:frame_number+win_width, ] = vector
        for w in xrange(win_width):
            temp_vector[w, 0] = vector[0, 0]
            temp_vector[frame_number+win_width+w, 0] = vector[frame_number-1, 0]

        for i in xrange(frame_number):
            for w in xrange(win_length):
                delta_vector[i] += temp_vector[i+w, 0] * delta_win[w]

        return  delta_vector

    def produce_nn_cmp(self, in_file_list, out_file_list):


        logger = logging.getLogger("acoustic_norm")

        delta_win = [-0.5, 0.0, 0.5]
        acc_win   = [1.0, -2.0, 1.0]
        
        file_number = len(in_file_list)
        logger.info('starting creation of %d files' % file_number)

        for i in xrange(file_number):
            
            mgc_data, bap_data, lf0_data = self.load_cmp_file(in_file_list[i])
            ip_lf0, vuv_vector = self.interpolate_f0(lf0_data)
            
            delta_lf0 = self.compute_delta(ip_lf0, delta_win)
            acc_lf0 = self.compute_delta(ip_lf0, acc_win)

            frame_number = ip_lf0.size

            cmp_data = numpy.concatenate((mgc_data, ip_lf0, delta_lf0, acc_lf0, vuv_vector, bap_data), axis=1)
            
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(cmp_data, out_file_list[i])
            
        logger.info('finished creation of %d binary files' % file_number)
            
if __name__ == '__main__':
    in_file_list = ['/group/project/dnn_tts/data/nick/cmp/herald_001.cmp']
    out_file_list = ['/group/project/dnn_tts/herald_001.out.cmp']

    cmp_norm = CMPNormalisation(mgc_dim=50, bap_dim=25, lf0_dim = 1)

    cmp_norm.produce_nn_cmp(in_file_list, out_file_list)

