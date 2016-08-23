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
from io_funcs.binary_io import BinaryIOCollection
import logging

class MinMaxNormalisation(object):
    def __init__(self, feature_dimension, min_value = 0.01, max_value = 0.99, min_vector = 0.0, max_vector = 0.0, exclude_columns=[]):

        # this is the wrong name for this logger because we can also normalise labels here too
        logger = logging.getLogger("acoustic_norm")

        self.target_min_value = min_value
        self.target_max_value = max_value

        self.feature_dimension = feature_dimension

        self.min_vector = min_vector
        self.max_vector = max_vector

        self.exclude_columns = exclude_columns

        if type(min_vector) != float:
            try:
                assert( len(self.min_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of min_vector (%d)' % (self.feature_dimension,len(self.min_vector)))
                raise
            
        if type(max_vector) != float:
            try:
                assert( len(self.max_vector) == self.feature_dimension)
            except AssertionError:
                logger.critical('inconsistent feature_dimension (%d) and length of max_vector (%d)' % (self.feature_dimension,len(self.max_vector)))
                raise

        logger.debug('MinMaxNormalisation created for feature dimension of %d' % self.feature_dimension)

    def load_min_max_values(self, label_norm_file):

        logger = logging.getLogger("acoustic_norm")

        io_funcs = BinaryIOCollection()
        min_max_vector, frame_number = io_funcs.load_binary_file_frame(label_norm_file, 1)
        min_max_vector = numpy.reshape(min_max_vector, (-1, ))
        self.min_vector = min_max_vector[0:frame_number/2]
        self.max_vector = min_max_vector[frame_number/2:]

        logger.info('Loaded min max values from the trained data for feature dimension of %d' % self.feature_dimension)

    def find_min_max_values(self, in_file_list):

        logger = logging.getLogger("acoustic_norm")

        file_number = len(in_file_list)
        min_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        max_value_matrix = numpy.zeros((file_number, self.feature_dimension))
        io_funcs = BinaryIOCollection()
        for i in xrange(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)
            
            temp_min = numpy.amin(features, axis = 0)
            temp_max = numpy.amax(features, axis = 0)
            
            min_value_matrix[i, ] = temp_min;
            max_value_matrix[i, ] = temp_max;

        self.min_vector = numpy.amin(min_value_matrix, axis = 0)
        self.max_vector = numpy.amax(max_value_matrix, axis = 0)
        self.min_vector = numpy.reshape(self.min_vector, (1, self.feature_dimension))
        self.max_vector = numpy.reshape(self.max_vector, (1, self.feature_dimension))

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('across %d files found min/max values of length %d:' % (file_number,self.feature_dimension) )
        logger.info('  min: %s' % self.min_vector)
        logger.info('  max: %s' % self.max_vector)
        # restore the print options
        # numpy.set_printoptions(po)

    def normalise_data(self, in_file_list, out_file_list):
        file_number = len(in_file_list)

        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        fea_max_min_diff = numpy.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = numpy.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)
        
        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        
        io_funcs = BinaryIOCollection()
        for i in xrange(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            frame_number = features.size / self.feature_dimension
            fea_min_matrix = numpy.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = numpy.tile(self.target_min_value, (frame_number, self.feature_dimension))
            
            fea_diff_matrix = numpy.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = numpy.tile(target_max_min_diff, (frame_number, 1)) / fea_diff_matrix

            norm_features = diff_norm_matrix * (features - fea_min_matrix) + target_min_matrix

            ## If we are to keep some columns unnormalised, use advanced indexing to 
            ## reinstate original values:
            m,n = numpy.shape(features)
            for col in self.exclude_columns:
                norm_features[range(m),[col]*m] = features[range(m),[col]*m]
                
            io_funcs.array_to_binary_file(norm_features, out_file_list[i])
            			
#            norm_features = numpy.array(norm_features, 'float32')
#            fid = open(out_file_list[i], 'wb')
#            norm_features.tofile(fid)
#            fid.close()

    def denormalise_data(self, in_file_list, out_file_list):

        logger = logging.getLogger("acoustic_norm")

        file_number = len(in_file_list)
        logger.info('MinMaxNormalisation.denormalise_data for %d files' % file_number)
        
        # print   self.max_vector, self.min_vector
        fea_max_min_diff = self.max_vector - self.min_vector
        diff_value = self.target_max_value - self.target_min_value
        # logger.debug('reshaping fea_max_min_diff from shape %s to (1,%d)' % (fea_max_min_diff.shape, self.feature_dimension) )
        
        fea_max_min_diff = numpy.reshape(fea_max_min_diff, (1, self.feature_dimension))

        target_max_min_diff = numpy.zeros((1, self.feature_dimension))
        target_max_min_diff.fill(diff_value)
        
        target_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        fea_max_min_diff[fea_max_min_diff <= 0.0] = 1.0
        
        io_funcs = BinaryIOCollection()
        for i in xrange(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            frame_number = features.size / self.feature_dimension
#            print   frame_number
            fea_min_matrix = numpy.tile(self.min_vector, (frame_number, 1))
            target_min_matrix = numpy.tile(self.target_min_value, (frame_number, self.feature_dimension))
            
            fea_diff_matrix = numpy.tile(fea_max_min_diff, (frame_number, 1))
            diff_norm_matrix = fea_diff_matrix / numpy.tile(target_max_min_diff, (frame_number, 1))
            norm_features = diff_norm_matrix * (features - target_min_matrix) + fea_min_matrix
            io_funcs.array_to_binary_file(norm_features, out_file_list[i])
            
    def normal_standardization(self, in_file_list, out_file_list):
        mean_vector = self.compute_mean(in_file_list)
        std_vector = self.compute_std(in_file_list, mean_vector)
        
        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)
        for i in xrange(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)
            current_frame_number = features.size / self.feature_dimension
            
            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))
            
            norm_features = (features - mean_matrix) / std_matrix
            
            io_funcs.array_to_binary_file(norm_features, out_file_list[i])
                    
    def compute_mean(self, file_list):
        
        logger = logging.getLogger("acoustic_norm")
        
        mean_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features = io_funcs.load_binary_file(file_name, self.feature_dimension)
            current_frame_number = features.size / self.feature_dimension
            mean_vector += numpy.reshape(numpy.sum(features, axis=0), (1, self.feature_dimension))
            all_frame_number += current_frame_number
            
        mean_vector /= float(all_frame_number)
        
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)
        
        return  mean_vector
    
    def compute_std(self, file_list, mean_vector):
        
        logger = logging.getLogger("acoustic_norm")
        
        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features = io_funcs.load_binary_file(file_name, self.feature_dimension)
            current_frame_number = features.size / self.feature_dimension
            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            
            std_vector += numpy.reshape(numpy.sum((features - mean_matrix) ** 2, axis=0), (1, self.feature_dimension))
            all_frame_number += current_frame_number
            
        std_vector /= float(all_frame_number)
        
        std_vector = std_vector ** 0.5
        
        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)
        
        return  std_vector
            
        
if __name__ == '__main__':
    
    in_file_list = ['/group/project/dnn_tts/data/nick/sp/nick/herald_001.sp']
    out_file_list = ['/group/project/dnn_tts/herald_001.sp']
    out_file_list1 = ['/group/project/dnn_tts/herald_001.test.sp']
    
    feature_dimension = 1025    

    normaliser = MinMaxNormalisation(feature_dimension, min_value = 0.01, max_value = 0.99)
    normaliser.find_min_max_values(in_file_list)
    tmp_min_vector = normaliser.min_vector
    tmp_max_vector = normaliser.max_vector
    normaliser.normalise_data(in_file_list, out_file_list)

    denormaliser = MinMaxNormalisation(feature_dimension, min_value = 0.01, max_value = 0.99, min_vector = tmp_min_vector, max_vector = tmp_max_vector)
    denormaliser.denormalise_data(out_file_list, out_file_list1)



