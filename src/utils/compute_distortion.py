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


import sys, numpy
from io_funcs.binary_io import BinaryIOCollection
import  logging
from scipy.stats.stats import pearsonr

class   DistortionComputation(object):
    def __init__(self, cmp_dim, mgc_dim, bap_dim, lf0_dim):
        self.total_frame_number = 0
        self.distortion = 0.0
        self.bap_distortion = 0.0
        self.f0_distortion = 0.0
        self.vuv_error = 0.0

        self.cmp_dim = cmp_dim
        self.mgc_dim = mgc_dim
        self.bap_dim = bap_dim
        self.lf0_dim = lf0_dim

    def compute_distortion(self, file_id_list, reference_dir, generation_dir, cmp_ext, mgc_ext, bap_ext, lf0_ext):

        total_voiced_frame_number = 0
        for file_id in file_id_list:
            reference_file_name = reference_dir + '/' + file_id + cmp_ext
            mgc_file_name = generation_dir + '/' + file_id + mgc_ext
            bap_file_name = generation_dir + '/' + file_id + bap_ext
            lf0_file_name = generation_dir + '/' + file_id + lf0_ext

            reference_cmp, ref_frame_number = self.load_binary_file(reference_file_name, self.cmp_dim)
            generation_mgc, mgc_frame_number = self.load_binary_file(mgc_file_name, self.mgc_dim)
            generation_bap, bap_frame_number = self.load_binary_file(bap_file_name, self.bap_dim)
            generation_lf0, lf0_frame_number = self.load_binary_file(lf0_file_name, self.lf0_dim)

            if ref_frame_number != mgc_frame_number:
                print("The number of frames is not the same: %d vs %d (%s). Error in compute_distortion.py\n." %(ref_frame_number, mgc_frame_number, file_id))
                sys.exit(1)

            reference_mgc = reference_cmp[:, 0:self.mgc_dim]
            reference_lf0 = reference_cmp[:, self.mgc_dim*3:self.mgc_dim*3+self.lf0_dim]
            reference_vuv = reference_cmp[:, self.mgc_dim*3+self.lf0_dim*3:self.mgc_dim*3+self.lf0_dim*3+1]
            reference_bap = reference_cmp[:, self.mgc_dim*3+self.lf0_dim*3+1:self.mgc_dim*3+self.lf0_dim*3+1+self.bap_dim]

            reference_lf0[reference_vuv<0.5] = 0.0
#            print   reference_vuv
            temp_distortion = self.compute_mse(reference_mgc[:, 1:self.mgc_dim], generation_mgc[:, 1:self.mgc_dim])
            self.distortion += temp_distortion * (10 /numpy.log(10)) * numpy.sqrt(2.0)

            temp_bap_distortion = self.compute_mse(reference_bap, generation_bap)
            self.bap_distortion += temp_bap_distortion * (10 /numpy.log(10)) * numpy.sqrt(2.0)

            temp_f0_distortion, temp_vuv_error, voiced_frame_number = self.compute_f0_mse(reference_lf0, generation_lf0)
            self.f0_distortion += temp_f0_distortion
            self.vuv_error += temp_vuv_error

            self.total_frame_number += ref_frame_number
            total_voiced_frame_number += voiced_frame_number

        self.distortion /= float(self.total_frame_number)
        self.bap_distortion /= float(self.total_frame_number)

        self.f0_distortion /= total_voiced_frame_number
        self.f0_distortion = numpy.sqrt(self.f0_distortion)

        self.vuv_error /= float(self.total_frame_number)

        return  self.distortion, self.bap_distortion, self.f0_distortion, self.vuv_error

    def compute_f0_mse(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        voiced_frame_number = voiced_gen_data.size

        f0_mse = numpy.sum(((numpy.exp(voiced_ref_data) - numpy.exp(voiced_gen_data)) ** 2))
#        f0_mse = numpy.sum((((voiced_ref_data) - (voiced_gen_data)) ** 2))

        vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        vuv_error = numpy.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

        return  f0_mse, vuv_error, voiced_frame_number

    def compute_mse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        sum_diff = numpy.sum(diff, axis=1)
        sum_diff = numpy.sqrt(sum_diff)       # ** 0.5
        sum_diff = numpy.sum(sum_diff, axis=0)

        return  sum_diff

    def load_binary_file(self, file_name, dimension):
        fid_lab = open(file_name, 'rb')
        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
        fid_lab.close()
        frame_number = features.size / dimension
        features = features[:(dimension * frame_number)]
        features = features.reshape((-1, dimension))

        return  features, frame_number


'''
to be refined. genertic class for various features
'''
class IndividualDistortionComp(object):

    def __init__(self):
        self.logger = logging.getLogger('computer_distortion')

    def compute_distortion(self, file_id_list, reference_dir, generation_dir, file_ext, feature_dim):
        total_voiced_frame_number = 0

        distortion = 0.0
        vuv_error = 0
        total_frame_number = 0

        io_funcs = BinaryIOCollection()

        ref_all_files_data = numpy.reshape(numpy.array([]), (-1,1))
        gen_all_files_data = numpy.reshape(numpy.array([]), (-1,1))
        for file_id in file_id_list:
            ref_file_name  = reference_dir + '/' + file_id + file_ext
            gen_file_name  = generation_dir + '/' + file_id + file_ext

            ref_data, ref_frame_number = io_funcs.load_binary_file_frame(ref_file_name, feature_dim)
            gen_data, gen_frame_number = io_funcs.load_binary_file_frame(gen_file_name, feature_dim)

            # accept the difference upto two frames
            if abs(ref_frame_number - gen_frame_number) <= 2:
                ref_frame_number = min(ref_frame_number, gen_frame_number)
                gen_frame_number = min(ref_frame_number, gen_frame_number)
                ref_data = ref_data[0:ref_frame_number, ]
                gen_data = gen_data[0:gen_frame_number, ]
            
            if ref_frame_number != gen_frame_number:
                self.logger.critical("The number of frames is not the same: %d vs %d (%s). Error in compute_distortion.py\n." %(ref_frame_number, gen_frame_number, file_id))
                raise

            if file_ext == '.lf0':
                ref_all_files_data = numpy.concatenate((ref_all_files_data, ref_data), axis=0)
                gen_all_files_data = numpy.concatenate((gen_all_files_data, gen_data), axis=0)
                temp_distortion, temp_vuv_error, voiced_frame_number = self.compute_f0_mse(ref_data, gen_data)
                vuv_error += temp_vuv_error
                total_voiced_frame_number += voiced_frame_number
            elif file_ext == '.dur':
                ref_data = numpy.reshape(numpy.sum(ref_data, axis=1), (-1, 1))
                gen_data = numpy.reshape(numpy.sum(gen_data, axis=1), (-1, 1))
                ref_all_files_data = numpy.concatenate((ref_all_files_data, ref_data), axis=0)
                gen_all_files_data = numpy.concatenate((gen_all_files_data, gen_data), axis=0)
                continue;
            elif file_ext == '.mgc':
                temp_distortion = self.compute_mse(ref_data[:, 1:feature_dim], gen_data[:, 1:feature_dim])
            else:
                temp_distortion = self.compute_mse(ref_data, gen_data)

            distortion += temp_distortion

            total_frame_number += ref_frame_number

        if file_ext == '.dur':
            dur_rmse = self.compute_rmse(ref_all_files_data, gen_all_files_data)
            dur_corr = self.compute_corr(ref_all_files_data, gen_all_files_data)

            return dur_rmse, dur_corr
        elif file_ext == '.lf0':
            distortion /= float(total_voiced_frame_number)
            vuv_error  /= float(total_frame_number)

            distortion = numpy.sqrt(distortion)
            f0_corr = self.compute_f0_corr(ref_all_files_data, gen_all_files_data)

            return  distortion, f0_corr, vuv_error
        else:
            distortion /= float(total_frame_number)

            return  distortion

    def compute_f0_mse(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        voiced_frame_number = voiced_gen_data.size

        f0_mse = (numpy.exp(voiced_ref_data) - numpy.exp(voiced_gen_data)) ** 2
        f0_mse = numpy.sum((f0_mse))

        vuv_error_vector = sum_ref_gen_vector[sum_ref_gen_vector == 0.0]
        vuv_error = numpy.sum(sum_ref_gen_vector[sum_ref_gen_vector == 1.0])

        return  f0_mse, vuv_error, voiced_frame_number

    def compute_f0_corr(self, ref_data, gen_data):
        ref_vuv_vector = numpy.zeros((ref_data.size, 1))
        gen_vuv_vector = numpy.zeros((ref_data.size, 1))

        ref_vuv_vector[ref_data > 0.0] = 1.0
        gen_vuv_vector[gen_data > 0.0] = 1.0

        sum_ref_gen_vector = ref_vuv_vector + gen_vuv_vector
        voiced_ref_data = ref_data[sum_ref_gen_vector == 2.0]
        voiced_gen_data = gen_data[sum_ref_gen_vector == 2.0]
        f0_corr = self.compute_corr(numpy.exp(voiced_ref_data), numpy.exp(voiced_gen_data))

        return f0_corr

    def compute_corr(self, ref_data, gen_data):
        corr_coef = pearsonr(ref_data, gen_data)

        return corr_coef[0]

    def compute_rmse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        total_frame_number = ref_data.size
        sum_diff = numpy.sum(diff)
        rmse = numpy.sqrt(sum_diff/total_frame_number)

        return rmse

    def compute_mse(self, ref_data, gen_data):
        diff = (ref_data - gen_data) ** 2
        sum_diff = numpy.sum(diff, axis=1)
        sum_diff = numpy.sqrt(sum_diff)       # ** 0.5
        sum_diff = numpy.sum(sum_diff, axis=0)

        return  sum_diff
