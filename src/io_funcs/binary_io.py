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

class   BinaryIOCollection(object):

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        features = features[:(dimension * (features.size // dimension))]
        features = features.reshape((-1, dimension))

        return  features

    def array_to_binary_file(self, data, output_file_name):
        data = numpy.array(data, 'float32')

        fid = open(output_file_name, 'wb')
        data.tofile(fid)
        fid.close()

    def load_binary_file_frame(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        assert features.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
        frame_number = features.size // dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))

        return  features, frame_number
