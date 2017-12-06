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


import sys, numpy, re, math
from io_funcs.binary_io import BinaryIOCollection
from multiprocessing.dummy import Pool as ThreadPool


class SilenceRemover(object):
    def __init__(self, n_cmp, silence_pattern=['*-#+*'], label_type="state_align", remove_frame_features=True,
                 subphone_feats="none"):
        self.silence_pattern = silence_pattern
        self.silence_pattern_size = len(silence_pattern)
        self.label_type = label_type
        self.remove_frame_features = remove_frame_features
        self.subphone_feats = subphone_feats
        self.n_cmp = n_cmp

    def remove_silence(self, in_data_list, in_align_list, out_data_list, dur_file_list=None):
        file_number = len(in_data_list)
        align_file_number = len(in_align_list)

        if file_number != align_file_number:
            print("The number of input and output files does not equal!\n")
            sys.exit(1)
        if file_number != len(out_data_list):
            print("The number of input and output files does not equal!\n")
            sys.exit(1)

        io_funcs = BinaryIOCollection()

        def _remove_silence(i):
            if self.label_type == "phone_align":
                if dur_file_list:
                    dur_file_name = dur_file_list[i]
                else:
                    dur_file_name = None
                nonsilence_indices = self.load_phone_alignment(in_align_list[i], dur_file_name)
            else:
                nonsilence_indices = self.load_alignment(in_align_list[i])

            ori_cmp_data = io_funcs.load_binary_file(in_data_list[i], self.n_cmp)

            frame_number = ori_cmp_data.size / self.n_cmp

            if len(nonsilence_indices) == frame_number:
                print('WARNING: no silence found!')
                # previsouly: continue -- in fact we should keep non-silent data!

            ## if labels have a few extra frames than audio, this can break the indexing, remove them:
            nonsilence_indices = [ix for ix in nonsilence_indices if ix < frame_number]

            new_cmp_data = ori_cmp_data[nonsilence_indices,]

            io_funcs.array_to_binary_file(new_cmp_data, out_data_list[i])

        pool = ThreadPool()
        pool.map(_remove_silence, range(file_number))
        pool.close()
        pool.join()

    ## OSW: rewrote above more succintly
    def check_silence_pattern(self, label):
        for current_pattern in self.silence_pattern:
            current_pattern = current_pattern.strip('*')
            if current_pattern in label:
                return 1
        return 0

    def load_phone_alignment(self, alignment_file_name, dur_file_name=None):

        if dur_file_name:
            io_funcs = BinaryIOCollection()
            dur_dim = 1  ## hard coded for now
            manual_dur_data = io_funcs.load_binary_file(dur_file_name, dur_dim)

        ph_count = 0
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)

            if len(temp_list) == 1:
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]

                # to do - support different frame shift - currently hardwired to 5msec
                # currently under beta testing: supports different frame shift
                if dur_file_name:
                    frame_number = manual_dur_data[ph_count]
                    ph_count = ph_count + 1
                else:
                    frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'none':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list

    def load_alignment(self, alignment_file_name, dur_file_name=None):

        state_number = 5
        base_frame_index = 0
        nonsilence_frame_index_list = []
        fid = open(alignment_file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            temp_list = re.split('\s+', line)
            if len(temp_list) == 1:
                state_index = state_number
                full_label = temp_list[0]
            else:
                start_time = int(temp_list[0])
                end_time = int(temp_list[1])
                full_label = temp_list[2]
                full_label_length = len(full_label) - 3  # remove state information [k]
                state_index = full_label[full_label_length + 1]
                state_index = int(state_index) - 1
                frame_number = int((end_time - start_time) / 50000)

            label_binary_flag = self.check_silence_pattern(full_label)

            if self.remove_frame_features:
                if label_binary_flag == 0:
                    for frame_index in range(frame_number):
                        nonsilence_frame_index_list.append(base_frame_index + frame_index)
                base_frame_index = base_frame_index + frame_number
            elif self.subphone_feats == 'state_only':
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1
            elif self.subphone_feats == 'none' and state_index == state_number:
                if label_binary_flag == 0:
                    nonsilence_frame_index_list.append(base_frame_index)
                base_frame_index = base_frame_index + 1

        fid.close()

        return nonsilence_frame_index_list


# def load_binary_file(self, file_name, dimension):

#        fid_lab = open(file_name, 'rb')
#        features = numpy.fromfile(fid_lab, dtype=numpy.float32)
#        fid_lab.close()
#        features = features[:(dimension * (features.size / dimension))]
#        features = features.reshape((-1, dimension))

#        return  features


def trim_silence(in_list, out_list, in_dimension, label_list, label_dimension, \
                 silence_feature_index, percent_to_keep=0):
    '''
    Function to trim silence from binary label/speech files based on binary labels.
        in_list: list of binary label/speech files to trim
        out_list: trimmed files
        in_dimension: dimension of data to trim
        label_list: list of binary labels which contain trimming criterion
        label_dimesion:
        silence_feature_index: index of feature in labels which is silence: 1 means silence (trim), 0 means leave.
    '''
    assert len(in_list) == len(out_list) == len(label_list)
    io_funcs = BinaryIOCollection()
    for (infile, outfile, label_file) in zip(in_list, out_list, label_list):

        data = io_funcs.load_binary_file(infile, in_dimension)
        label = io_funcs.load_binary_file(label_file, label_dimension)

        audio_label_difference = data.shape[0] - label.shape[0]
        assert math.fabs(audio_label_difference) < 3, '%s and %s contain different numbers of frames: %s %s' % (
            infile, label_file, data.shape[0], label.shape[0])

        ## In case they are different, resize -- keep label fixed as we assume this has
        ## already been processed. (This problem only arose with STRAIGHT features.)
        if audio_label_difference < 0:  ## label is longer -- pad audio to match by repeating last frame:
            print('audio too short -- pad')
            padding = numpy.vstack([data[-1, :]] * int(math.fabs(audio_label_difference)))
            data = numpy.vstack([data, padding])
        elif audio_label_difference > 0:  ## audio is longer -- cut it
            print('audio too long -- trim')
            new_length = label.shape[0]
            data = data[:new_length, :]
        # else: -- expected case -- lengths match, so do nothing

        silence_flag = label[:, silence_feature_index]
        #         print silence_flag
        if not (numpy.unique(silence_flag) == numpy.array([0, 1])).all():
            ## if it's all 0s or 1s, that's ok:
            assert (numpy.unique(silence_flag) == numpy.array([0]).all()) or \
                   (numpy.unique(silence_flag) == numpy.array([1]).all()), \
                'dimension %s of %s contains values other than 0 and 1' % (silence_feature_index, infile)
        print('Remove %d%% of frames (%s frames) as silence... ' % (
            100 * numpy.sum(silence_flag / float(len(silence_flag))), int(numpy.sum(silence_flag))))
        non_silence_indices = numpy.nonzero(
            silence_flag == 0)  ## get the indices where silence_flag == 0 is True (i.e. != 0)
        if percent_to_keep != 0:
            assert type(percent_to_keep) == int and percent_to_keep > 0
            # print silence_flag
            silence_indices = numpy.nonzero(silence_flag == 1)
            ## nonzero returns a tuple of arrays, one for each dimension of input array
            silence_indices = silence_indices[0]
            every_nth = 100 / percent_to_keep
            silence_indices_to_keep = silence_indices[::every_nth]  ## every_nth used +as step value in slice
            ## -1 due to weird error with STRAIGHT features at line 144:
            ## IndexError: index 445 is out of bounds for axis 0 with size 445
            if len(silence_indices_to_keep) == 0:
                silence_indices_to_keep = numpy.array([1])  ## avoid errors in case there is no silence
            print('   Restore %s%% (every %sth frame: %s frames) of silent frames' % (
                percent_to_keep, every_nth, len(silence_indices_to_keep)))

            ## Append to end of utt -- same function used for labels and audio
            ## means that violation of temporal order doesn't matter -- will be consistent.
            ## Later, frame shuffling will disperse silent frames evenly across minibatches:
            non_silence_indices = (numpy.hstack([non_silence_indices[0], silence_indices_to_keep]))
            ##  ^---- from tuple and back (see nonzero note above)

        trimmed_data = data[non_silence_indices, :]  ## advanced integer indexing
        io_funcs.array_to_binary_file(trimmed_data, outfile)


if __name__ == '__main__':
    cmp_file_list_name = ''
    lab_file_list_name = ''
    align_file_list_name = ''

    n_cmp = 229
    n_lab = 898

    in_cmp_list = ['/group/project/dnn_tts/data/nick/nn_cmp/nick/herald_001.cmp']
    in_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.lab']
    in_align_list = ['/group/project/dnn_tts/data/cassia/nick_lab/herald_001.lab']

    out_cmp_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.cmp']
    out_lab_list = ['/group/project/dnn_tts/data/nick/nn_new_lab/herald_001.tmp.no.lab']

    remover = SilenceRemover(in_cmp_list, in_align_list, n_cmp, out_cmp_list)
    remover.remove_silence()
