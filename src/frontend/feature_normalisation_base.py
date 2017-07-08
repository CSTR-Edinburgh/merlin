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

class FeatureNormBase(object):
    '''
    to normalise feature into specific range
    to de-normalise feature back
    support min-max norm, MVN,
    this is a genetic class
    '''
    def __init__(self):
        self.logger = logging.getLogger('feature_normalisation')

        self.dimension_dict = {}
        self.start_index_dict = {}
        self.feature_dimension = 0

    def feature_normalisation(self):
        pass

    def feature_denormalisation(self):
        pass


    def normal_standardization(self, in_file_list, out_file_list, feature_dimension):

#        self.dimension_dict = dimension_dict
        self.feature_dimension = feature_dimension

        mean_vector = self.compute_mean(in_file_list, 0, feature_dimension)
        std_vector = self.compute_std(in_file_list, mean_vector, 0, feature_dimension)

        io_funcs = BinaryIOCollection()
        file_number = len(in_file_list)

        for i in range(file_number):

            features, current_frame_number = io_funcs.load_binary_file_frame(in_file_list[i], self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))
            std_matrix = numpy.tile(std_vector, (current_frame_number, 1))

            norm_features = (features - mean_matrix) / std_matrix

            io_funcs.array_to_binary_file(norm_features, out_file_list[i])

        return  mean_vector, std_vector

    def find_min_max_values(self, in_file_list, start_index, end_index):

        local_feature_dimension = end_index - start_index

        file_number = len(in_file_list)
        min_value_matrix = numpy.zeros((file_number, local_feature_dimension))
        max_value_matrix = numpy.zeros((file_number, local_feature_dimension))
        io_funcs = BinaryIOCollection()
        for i in range(file_number):
            features = io_funcs.load_binary_file(in_file_list[i], self.feature_dimension)

            temp_min = numpy.amin(features[:, start_index:end_index], axis = 0)
            temp_max = numpy.amax(features[:, start_index:end_index], axis = 0)

            min_value_matrix[i, ] = temp_min;
            max_value_matrix[i, ] = temp_max;

        self.min_vector = numpy.amin(min_value_matrix, axis = 0)
        self.max_vector = numpy.amax(max_value_matrix, axis = 0)
        self.min_vector = numpy.reshape(self.min_vector, (1, local_feature_dimension))
        self.max_vector = numpy.reshape(self.max_vector, (1, local_feature_dimension))

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('found min/max values of length %d:' % local_feature_dimension)
        self.logger.info('  min: %s' % self.min_vector)
        self.logger.info('  max: %s' % self.max_vector)
        # restore the print options
        # numpy.set_printoptions(po)

    def compute_mean(self, file_list, start_index, end_index):

        local_feature_dimension = end_index - start_index

        mean_vector = numpy.zeros((1, local_feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_vector += numpy.reshape(numpy.sum(features[:, start_index:end_index], axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        mean_vector /= float(all_frame_number)

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('computed mean vector of length %d :' % mean_vector.shape[1] )
        self.logger.info(' mean: %s' % mean_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  mean_vector

    def compute_std(self, file_list, mean_vector, start_index, end_index):
        local_feature_dimension = end_index - start_index

        std_vector = numpy.zeros((1, self.feature_dimension))
        all_frame_number = 0

        io_funcs = BinaryIOCollection()
        for file_name in file_list:
            features, current_frame_number = io_funcs.load_binary_file_frame(file_name, self.feature_dimension)

            mean_matrix = numpy.tile(mean_vector, (current_frame_number, 1))

            std_vector += numpy.reshape(numpy.sum((features[:, start_index:end_index] - mean_matrix) ** 2, axis=0), (1, local_feature_dimension))
            all_frame_number += current_frame_number

        std_vector /= float(all_frame_number)

        std_vector = std_vector ** 0.5

        # po=numpy.get_printoptions()
        # numpy.set_printoptions(precision=2, threshold=20, linewidth=1000, edgeitems=4)
        self.logger.info('computed  std vector of length %d' % std_vector.shape[1] )
        self.logger.info('  std: %s' % std_vector)
        # restore the print options
        # numpy.set_printoptions(po)

        return  std_vector
