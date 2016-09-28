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


import logging
import sys

## a generic class of linguistic feature extraction
## 
class LinguisticBase(object):
    def __init__(self, dimension=0):
        self.dimension = dimension  ##the feature dimensionality of output (should that read 'input' ?)

        ## the number of utterances to be normalised
        self.utterance_num = 0

    
    ## the ori_file_list contains the file paths of the raw linguistic data
    ## the output_file_list contains the file paths of the normalised linguistic data
    ## 
    def perform_normalisation(self, ori_file_list, output_file_list, label_type="state_align", dur_file_list=None):
        
        logger = logging.getLogger("perform_normalisation")
        logger.info('perform linguistic feature extraction')
        self.utterance_num = len(ori_file_list)
        if self.utterance_num != len(output_file_list):
            logger.error('the number of input and output linguistic files should be the same!\n')
            sys.exit(1)

        for i in xrange(self.utterance_num):
            if not dur_file_list:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type)
            else:
                self.extract_linguistic_features(ori_file_list[i], output_file_list[i], label_type, dur_file_list[i])
    
    ## the exact function to do the work
    ## need to be implemented in the specific class
    ## the function will write the linguistic features directly to the output file
    def extract_linguistic_features(self, in_file_name, out_file_name, label_type, dur_file_name=None):
        pass
