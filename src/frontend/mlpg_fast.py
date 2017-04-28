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



import numpy as np
from numpy import dot
import logging
from numpy import float64


# Adding this before the bandmat import lets us import .pyx files without running bandmat's setup.py:
#import pyximport; pyximport.install()



import bandmat as bm
import bandmat.linalg as bla

class MLParameterGenerationFast(object):
    def __init__(self, delta_win = [-0.5, 0.0, 0.5], acc_win = [1.0, -2.0, 1.0]):
        self.delta_win = delta_win
        self.acc_win   = acc_win
        ###assume the delta and acc windows have the same length
        self.win_length = int(len(delta_win)/2)

    def build_win_mats(self, windows, frames):
        win_mats = []
        for l, u, win_coeff in windows:
            assert l >= 0 and u >= 0
            assert len(win_coeff) == l + u + 1
            win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
            win_mat = bm.band_c_bm(u, l, win_coeffs).T
            win_mats.append(win_mat)

        return win_mats

    def build_poe(self, b_frames, tau_frames, win_mats, sdw=None):
#        tau_frames.astype('float64')

        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])
        num_windows = len(win_mats)
        frames = len(b_frames)
        assert np.shape(b_frames) == (frames, num_windows)
        assert np.shape(tau_frames) == (frames, num_windows)
        assert all([ win_mat.l + win_mat.u <= sdw for win_mat in win_mats ])

        b = np.zeros((frames,))
        prec = bm.zeros(sdw, sdw, frames)

        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mv_plus_equals(win_mat.T, b_frames[:, win_index], target=b)
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                                  diag=float64(tau_frames[:, win_index]))

        return b, prec

    def generation(self, features, covariance, static_dimension):

        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        num_windows = len(windows)

        frame_number = features.shape[0]

        logger = logging.getLogger('param_generation')
        logger.debug('starting MLParameterGeneration.generation')

        gen_parameter = np.zeros((frame_number, static_dimension))

        win_mats = self.build_win_mats(windows, frame_number)
        mu_frames = np.zeros((frame_number, 3))
        var_frames = np.zeros((frame_number, 3))

        for d in range(static_dimension):
            var_frames[:, 0] = covariance[:, d]
            var_frames[:, 1] = covariance[:, static_dimension+d]
            var_frames[:, 2] = covariance[:, static_dimension*2+d]
            mu_frames[:, 0] = features[:, d]
            mu_frames[:, 1] = features[:, static_dimension+d]
            mu_frames[:, 2] = features[:, static_dimension*2+d]
            var_frames[0, 1] = 100000000000;
            var_frames[0, 2] = 100000000000;
            var_frames[frame_number-1, 1] = 100000000000;
            var_frames[frame_number-1, 2] = 100000000000;

            b_frames = mu_frames / var_frames
            tau_frames = 1.0 / var_frames

            b, prec = self.build_poe(b_frames, tau_frames, win_mats)
            mean_traj = bla.solveh(prec, b)

            gen_parameter[0:frame_number, d] = mean_traj

        return  gen_parameter
