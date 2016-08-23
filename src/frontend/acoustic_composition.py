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

from io_funcs.binary_io import BinaryIOCollection
import numpy
import logging
from acoustic_base import AcousticBase
import os
#io_funcs.

class   AcousticComposition(AcousticBase):

    ###prepare_nn_data(self, in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict):

    '''
    variables inheritate from AcousticBase:
        self.compute_dynamic = {}
        self.file_number = 0
        self.data_stream_number = 0
        self.data_stream_list = []

        self.out_dimension = 0
        self.record_vuv    = False
    '''
    def make_equal_frames(self, in_file_list, ref_file_list, in_dimension_dict):
        logger = logging.getLogger("acoustic_comp")
        
        logger.info('making equal number of lines...')
        
        io_funcs = BinaryIOCollection()

        utt_number = len(in_file_list)

        for i in xrange(utt_number):
            in_file_name = in_file_list[i]
            in_data_stream_name = in_file_name.split('.')[-1]
            in_feature_dim = in_dimension_dict[in_data_stream_name]
            in_features, in_frame_number = io_funcs.load_binary_file_frame(in_file_name, in_feature_dim)
            
            ref_file_name = ref_file_list[i]
            ref_data_stream_name = ref_file_name.split('.')[-1]
            ref_feature_dim = in_dimension_dict[ref_data_stream_name]
            ref_features, ref_frame_number = io_funcs.load_binary_file_frame(ref_file_name, ref_feature_dim)

            target_features = numpy.zeros((ref_frame_number, in_feature_dim))
            if in_frame_number == ref_frame_number:
                continue;
            elif in_frame_number > ref_frame_number:
                target_features[0:ref_frame_number, ] = in_features[0:ref_frame_number, ]
            elif in_frame_number < ref_frame_number:
                target_features[0:in_frame_number, ] = in_features[0:in_frame_number, ]
            io_funcs.array_to_binary_file(target_features, in_file_name)
        
        logger.info('Finished: made equal rows in data stream %s with reference to data stream %s ' %(in_data_stream_name, ref_data_stream_name))

                
    def prepare_data(self, in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict):

        logger = logging.getLogger("acoustic_comp")

        stream_start_index = {}
        stream_dim_index = 0
        for stream_name in out_dimension_dict.keys():
            if not stream_start_index.has_key(stream_name):
                stream_start_index[stream_name] = stream_dim_index

            stream_dim_index += out_dimension_dict[stream_name]
                
        io_funcs = BinaryIOCollection()

        for i in xrange(self.file_number):
            out_file_name = out_file_list[i]

            #if os.path.isfile(out_file_name):
            #    logger.info('processing file %4d of %4d : %s exists' % (i+1, self.file_number, out_file_name))
		    #    continue

            logger.info('processing file %4d of %4d : %s' % (i+1,self.file_number,out_file_name))

            out_data_matrix = None
            out_frame_number = 0


            for k in xrange(self.data_stream_number):
                data_stream_name = self.data_stream_list[k]

                in_file_name = in_file_list_dict[data_stream_name][i]

                in_feature_dim = in_dimension_dict[data_stream_name]
                features, frame_number = io_funcs.load_binary_file_frame(in_file_name, in_feature_dim)

                if k == 0:
                    out_frame_number = frame_number
                    out_data_matrix = numpy.zeros((out_frame_number, self.out_dimension))

                if frame_number > out_frame_number:
                    features = features[0:out_frame_number, ]
                    frame_number = out_frame_number
                
                try:
                    assert  out_frame_number == frame_number
                except AssertionError:
                    logger.critical('the frame number of data stream %s is not consistent with others: current %d others %d' 
                                         %(data_stream_name, out_frame_number, frame_number))
                    raise

                dim_index = stream_start_index[data_stream_name]

                if data_stream_name in ['lf0', 'F0']:   ## F0 added for GlottHMM
                    features, vuv_vector = self.interpolate_f0(features)

                    ### if vuv information to be recorded, store it in corresponding column
                    if self.record_vuv:
                        out_data_matrix[0:out_frame_number, stream_start_index['vuv']:stream_start_index['vuv']+1] = vuv_vector

                out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = features
                dim_index = dim_index+in_feature_dim

                if self.compute_dynamic[data_stream_name]: 

                    delta_features = self.compute_dynamic_matrix(features, self.delta_win, frame_number, in_feature_dim)
                    acc_features   = self.compute_dynamic_matrix(features, self.acc_win, frame_number, in_feature_dim)


                    out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = delta_features
                    dim_index = dim_index+in_feature_dim

                    out_data_matrix[0:out_frame_number, dim_index:dim_index+in_feature_dim] = acc_features
            
            ### write data to file
            io_funcs.array_to_binary_file(out_data_matrix, out_file_name)
            logger.debug(' wrote %d frames of features',out_frame_number )
            
    def acoustic_decomposition(self, in_file_list, out_dimension_dict, file_extension_dict):

        stream_start_index = {}
        dimension_index = 0
        recorded_vuv = False
        vuv_dimension = None
        for feature_name in out_dimension_dict.keys():
            if feature_name != 'vuv':
                stream_start_index[feature_name] = dimension_index
            else:
                vuv_dimension = dimension_index
                recorded_vuv = True
            
            dimension_index += out_dimension_dict[feature_name]

        for file_name in in_file_list:
            dir_name = os.path.dirname(file_name)
            file_id = os.path.splitext(os.path.basename(file_name))[0]


if __name__ == '__main__':

    acoustic_cmper = AcousticPreparation()

    in_dimension_dict = { 'mgc' : 50,
                          'lf0' : 1,
                          'bap' : 25}    
    out_dimension_dict = { 'mgc' : 150,
                           'lf0' : 3,
                           'vuv' : 1,
                           'bap' : 75}
    
    in_file_list_dict = {}
    in_file_list_dict['mgc'] = ['/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/mgc/herald_001.mgc', '/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/mgc/herald_002.mgc']
    in_file_list_dict['lf0'] = ['/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/lf0/herald_001.lf0', '/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/lf0/herald_002.lf0']
    in_file_list_dict['bap'] = ['/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/bap/herald_001.bap', '/afs/inf.ed.ac.uk/group/project/dnn_tts/data/nick/bap/herald_002.bap']

    out_file_list = ['/afs/inf.ed.ac.uk/group/project/dnn_tts/herald_001.cmp', '/afs/inf.ed.ac.uk/group/project/dnn_tts/herald_002.cmp']

    acoustic_cmper.prepare_nn_data(in_file_list_dict, out_file_list, in_dimension_dict, out_dimension_dict)

