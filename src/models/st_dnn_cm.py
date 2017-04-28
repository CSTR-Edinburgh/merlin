
import pickle
import os
import sys
import time

import numpy as np
import gnumpy as gnp

from numpy import float64

import bandmat as bm
import bandmat.linalg as bla

from guppy import hpy

import logging

class SequentialDNN(object):

    def __init__(self, numpy_rng, n_ins=100,
                 n_outs=100, l1_reg = None, l2_reg = None,
                 hidden_layer_sizes=[500, 500],
                 hidden_activation='tanh', output_activation='linear'):

        logger = logging.getLogger("DNN initialization")

        self.n_layers = len(hidden_layer_sizes)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

        assert self.n_layers > 0

        self.W_params = []
        self.b_params = []
        self.mW_params = []
        self.mb_params = []

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layer_sizes[i-1]
            W_value = gnp.garray(numpy_rng.normal(0.0, 1.0/np.sqrt(input_size), size=(input_size, hidden_layer_sizes[i])))
            b_value = gnp.zeros(hidden_layer_sizes[i])
            mW_value = gnp.zeros((input_size, hidden_layer_sizes[i]))
            mb_value = gnp.zeros(hidden_layer_sizes[i])
            self.W_params.append(W_value)
            self.b_params.append(b_value)
            self.mW_params.append(mW_value)
            self.mb_params.append(mb_value)

        #output layer
        input_size = hidden_layer_sizes[self.n_layers-1]
        W_value = gnp.garray(numpy_rng.normal(0.0, 1.0/np.sqrt(input_size), size=(input_size, n_outs)))
        b_value = gnp.zeros(n_outs)
        mW_value = gnp.zeros((input_size, n_outs))
        mb_value = gnp.zeros(n_outs)
        self.W_params.append(W_value)
        self.b_params.append(b_value)
        self.mW_params.append(mW_value)
        self.mb_params.append(mb_value)


    def backpropagation(self, train_set_y, mean_matrix, std_matrix):

        final_layer_output = self.final_layer_output

        final_layer_output = final_layer_output * gnp.garray(std_matrix) + gnp.garray(mean_matrix)
        frame_number = final_layer_output.shape[0]

        final_layer_output = final_layer_output.T
        obs_mat = gnp.zeros((61, frame_number*3))
        traj_err_mat = gnp.zeros((61, frame_number))
        observation_error = gnp.zeros((frame_number, 259))

        var_base = np.zeros((61, 3))
        static_indice = []
        delta_indice = []
        acc_indice = []

        for i in range(60):
            static_indice.append(i)
            delta_indice.append(i+60)
            acc_indice.append(i+120)
        static_indice.append(181)
        delta_indice.append(182)
        acc_indice.append(183)
#        for i in xrange(25):
#            static_indice.append(i+184)
#            delta_indice.append(i+184+25)
#            acc_indice.append(i+184+50)

        obs_mat[:, 0:frame_number] = final_layer_output[static_indice, :]
        obs_mat[:, frame_number:frame_number*2] = final_layer_output[delta_indice, :]
        obs_mat[:, frame_number*2:frame_number*3] = final_layer_output[acc_indice, :]

        var_base[:, 0] = std_matrix[0, static_indice].T
        var_base[:, 1] = std_matrix[0, delta_indice].T
        var_base[:, 2] = std_matrix[0, acc_indice].T
        var_base = np.reshape(var_base, (61*3, 1))
        var_base = var_base ** 2

        sub_dim_list = []
        for i in range(61):
            sub_dim_list.append(1)

        sub_dim_start = 0
        for sub_dim in sub_dim_list:
            wuw_mat, wu_mat = self.pre_wuw_wu(frame_number, sub_dim, var_base[sub_dim_start*3:sub_dim_start*3+sub_dim*3])

            obs_mu = obs_mat[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*3*sub_dim, 1))
            wuwwu = gnp.dot(wuw_mat, wu_mat)

            mlpg_traj = gnp.dot(wuwwu, obs_mu)

            sub_std_mat = std_matrix[:, static_indice].T
            sub_mu_mat  = mean_matrix[:, static_indice].T
            sub_std_mat = sub_std_mat[sub_dim_start:sub_dim_start+sub_dim, :]

#            print   sub_std_mat
            sub_std_mat = sub_std_mat.reshape((frame_number*sub_dim, 1))
            sub_mu_mat = sub_mu_mat[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*sub_dim, 1))

            sub_o_std_vec = var_base[sub_dim_start*3:sub_dim_start*3+sub_dim*3]
            sub_o_std_mat = np.tile(sub_o_std_vec.T, (frame_number, 1))
            sub_o_std_mat = (sub_o_std_mat.T) ** 0.5
            sub_o_std_vec = sub_o_std_mat.reshape((frame_number*sub_dim*3, 1))
#            print   sub_o_std_vec, var_base[sub_dim_start*3:sub_dim_start*3+sub_dim*3] ** 0.5

            ref_y = train_set_y[:, static_indice].T
            ref_y = ref_y[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*sub_dim, 1))

            ref_y = ref_y * sub_std_mat + sub_mu_mat
            traj_err = (mlpg_traj - ref_y)

            traj_err_mat[sub_dim_start:sub_dim_start+sub_dim] = traj_err.reshape((sub_dim, frame_number))

            traj_err = traj_err / sub_std_mat

            obs_err_vec = gnp.dot(wuwwu.T, traj_err)
#            temp_obs_err_vec = gnp.dot(traj_err.T, wuwwu)
#            print   obs_err_vec, temp_obs_err_vec
#            print   obs_err_vec.shape, temp_obs_err_vec.shape
            obs_err_vec = obs_err_vec * sub_o_std_vec
#            print   obs_mu, mlpg_traj, ref_y
#            print   obs_err_vec.shape, sub_o_std_vec.shape, frame_number, wuwwu.shape, traj_err.shape
            obs_mat[sub_dim_start:sub_dim_start+sub_dim, :] = obs_err_vec.reshape((sub_dim, frame_number*3))

            sub_dim_start = sub_dim_start + sub_dim

        self.errors  = gnp.sum(traj_err_mat[0:60, :].T ** 2, axis=1)

        observation_error[:, 0:60]    = obs_mat[0:60, 0:frame_number].T
        observation_error[:, 60:120]  = obs_mat[0:60, frame_number:frame_number*2].T
        observation_error[:, 120:180] = obs_mat[0:60, frame_number*2:frame_number*3].T
        observation_error[:, 181]     = obs_mat[60, 0:frame_number].T
        observation_error[:, 182]     = obs_mat[60, frame_number:frame_number*2].T
        observation_error[:, 183]     = obs_mat[60, frame_number*2:frame_number*3].T

        self.W_grads = []
        self.b_grads = []
        current_error = observation_error
        current_activation = self.activations[-1]
        current_W_grad = gnp.dot(current_activation.T, observation_error)
        current_b_grad = gnp.dot(gnp.ones((1, observation_error.shape[0])), observation_error)
        propagate_error = gnp.dot(observation_error, self.W_params[self.n_layers].T) # final layer is linear output, gradient is one
        self.W_grads.append(current_W_grad)
        self.b_grads.append(current_b_grad)
        for i in reversed(list(range(self.n_layers))):
            current_activation = self.activations[i]
            current_gradient = 1.0 - current_activation ** 2
            current_W_grad = gnp.dot(current_activation.T, propagate_error)
            current_b_grad = gnp.dot(gnp.ones((1, propagate_error.shape[0])), propagate_error)
            propagate_error = gnp.dot(propagate_error, self.W_params[i].T) * current_gradient

            self.W_grads.insert(0, current_W_grad)
            self.b_grads.insert(0, current_b_grad)

    def feedforward(self, train_set_x):
        self.activations = []

        self.activations.append(train_set_x)

        for i in range(self.n_layers):
            input_data = self.activations[i]
            current_activations = gnp.tanh(gnp.dot(input_data, self.W_params[i]) + self.b_params[i])
            self.activations.append(current_activations)

        #output layers
        self.final_layer_output = gnp.dot(self.activations[self.n_layers], self.W_params[self.n_layers]) + self.b_params[self.n_layers]

    def gradient_update(self, batch_size, learning_rate, momentum):

        multiplier = learning_rate / batch_size;
        for i in range(len(self.W_grads)):

            if i >= len(self.W_grads) - 2:
                local_multiplier = multiplier * 0.5
            else:
                local_multiplier = multiplier

            self.W_grads[i] = (self.W_grads[i] + self.W_params[i] * self.l2_reg) * local_multiplier
            self.b_grads[i] = self.b_grads[i] * local_multiplier   # + self.b_params[i] * self.l2_reg

            #update weights and record momentum weights
            self.mW_params[i] = (self.mW_params[i] * momentum) - self.W_grads[i]
            self.mb_params[i] = (self.mb_params[i] * momentum) - self.b_grads[i]
            self.W_params[i] += self.mW_params[i]
            self.b_params[i] += self.mb_params[i]



    def finetune(self, train_xy, batch_size, learning_rate, momentum, mean_matrix, std_matrix):
        (train_set_x, train_set_y) = train_xy

        train_set_x = gnp.as_garray(train_set_x)
        train_set_y = gnp.as_garray(train_set_y)

        self.feedforward(train_set_x)
        self.backpropagation(train_set_y, mean_matrix, std_matrix)
        self.gradient_update(batch_size, learning_rate, momentum)

#        self.errors = gnp.sum((self.final_layer_output - train_set_y) ** 2, axis=1)

        return  self.errors.as_numpy_array()

    def parameter_prediction(self, test_set_x):
        test_set_x = gnp.garray(test_set_x)

        current_activations = test_set_x

        for i in range(self.n_layers):
            input_data = current_activations
            current_activations = gnp.tanh(gnp.dot(input_data, self.W_params[i]) + self.b_params[i])

        final_layer_output = gnp.dot(current_activations, self.W_params[self.n_layers]) + self.b_params[self.n_layers]

        return  final_layer_output.as_numpy_array()

    def parameter_prediction_trajectory(self, test_set_x, test_set_y, mean_matrix, std_matrix):
        test_set_x = gnp.garray(test_set_x)

        current_activations = test_set_x

        for i in range(self.n_layers):
            input_data = current_activations
            current_activations = gnp.tanh(gnp.dot(input_data, self.W_params[i]) + self.b_params[i])

        final_layer_output = gnp.dot(current_activations, self.W_params[self.n_layers]) + self.b_params[self.n_layers]

        final_layer_output = final_layer_output * gnp.garray(std_matrix) + gnp.garray(mean_matrix)
        frame_number = final_layer_output.shape[0]

        final_layer_output = final_layer_output.T
        obs_mat = gnp.zeros((60, frame_number*3))
        traj_err_mat = gnp.zeros((60, frame_number))

        var_base = np.zeros((60, 3))
        static_indice = []
        delta_indice = []
        acc_indice = []

        for i in range(60):
            static_indice.append(i)
            delta_indice.append(i+60)
            acc_indice.append(i+120)

        obs_mat[:, 0:frame_number] = final_layer_output[static_indice, :]
        obs_mat[:, frame_number:frame_number*2] = final_layer_output[delta_indice, :]
        obs_mat[:, frame_number*2:frame_number*3] = final_layer_output[acc_indice, :]

        var_base[:, 0] = std_matrix[0, static_indice].T
        var_base[:, 1] = std_matrix[0, delta_indice].T
        var_base[:, 2] = std_matrix[0, acc_indice].T

        var_base = np.reshape(var_base, (60*3, 1))
        var_base = var_base ** 2

        sub_dim_list = []
        for i in range(60):
            sub_dim_list.append(1)

        sub_dim_start = 0
        for sub_dim in sub_dim_list:
            wuw_mat, wu_mat = self.pre_wuw_wu(frame_number, sub_dim, var_base[sub_dim_start*3:sub_dim_start*3+sub_dim*3])

            obs_mu = obs_mat[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*3*sub_dim, 1))
            wuwwu = gnp.dot(wuw_mat, wu_mat)
            mlpg_traj = gnp.dot(wuwwu, obs_mu)

            sub_std_mat = std_matrix[:, static_indice].T
            sub_mu_mat  = mean_matrix[:, static_indice].T
            sub_std_mat = sub_std_mat[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*sub_dim, 1))
            sub_mu_mat = sub_mu_mat[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*sub_dim, 1))

            ref_y = test_set_y[:, static_indice].T
            ref_y = ref_y[sub_dim_start:sub_dim_start+sub_dim, :].reshape((frame_number*sub_dim, 1))

            ref_y = ref_y * sub_std_mat + sub_mu_mat
            traj_err = (mlpg_traj - ref_y)  #mlpg_traj ref_y

            traj_err_mat[sub_dim_start:sub_dim_start+sub_dim, :] = traj_err.reshape((sub_dim, frame_number))

            sub_dim_start = sub_dim_start + sub_dim

        validation_losses = gnp.sum(traj_err_mat[1:60, :].T ** 2, axis=1)
        validation_losses = validation_losses ** 0.5

        return  validation_losses.as_numpy_array()


    def set_parameters(self, W_params, b_params):

        assert len(self.W_params) == len(W_params)

#        for i in xrange(len(self.W_params)):
        for i in range(len(self.W_params)):
            self.W_params[i] = W_params[i]
            self.b_params[i] = b_params[i]

    def set_delta_params(self, mW_params, mb_params):
        assert len(self.mW_params) == len(mW_params)

        for i in range(len(self.mW_params)):
            self.mW_params[i] = mW_params[i]
            self.mb_params[i] = mb_params[i]

    '''
    #############following function for MLPG##################
    '''
    def pre_wuw_wu(self, frame_number, static_dimension, var_base):

        wuw_mat = gnp.zeros((frame_number*static_dimension, frame_number*static_dimension))
        wu_mat  = gnp.zeros((frame_number*static_dimension, 3*frame_number*static_dimension))

        for i in range(static_dimension):
            temp_var_base = [var_base[i*3], var_base[i*3+1], var_base[i*3+2]]
            temp_wuw, temp_wu = self.pre_compute_wuw(frame_number, temp_var_base)
            wuw_mat[frame_number*i:frame_number*(i+1), frame_number*i:frame_number*(i+1)] = gnp.garray(temp_wuw[:])
            wu_mat[frame_number*i:frame_number*(i+1), frame_number*i:frame_number*(i+3)] = gnp.garray(temp_wu[:])

        return  wuw_mat, wu_mat

    def pre_compute_wuw(self, frame_number, var_base):
        windows = [
            (0, 0, np.array([1.0])),
            (1, 1, np.array([-0.5, 0.0, 0.5])),
            (1, 1, np.array([1.0, -2.0, 1.0])),
        ]
        num_windows = len(windows)

        win_mats = self.build_win_mats(windows, frame_number)

        var_base = np.array(var_base)
        var_base = np.reshape(var_base, (1, 3))

        var_frames = np.tile(var_base, (frame_number, 1))
        var_frames[0, 1] = 100000000000;
        var_frames[0, 2] = 100000000000;
        var_frames[frame_number-1, 1] = 100000000000;
        var_frames[frame_number-1, 2] = 100000000000;


        tau_frames = 1.0 / var_frames

        prec = self.build_wuw(frame_number, tau_frames, win_mats)
        inv_prec_full = bla.solveh(prec, np.eye(frame_number))

        wu_list = self.build_wu(frame_number, tau_frames, win_mats)

        wu_mat = np.zeros((frame_number, frame_number * 3))
        wu_mat[:, 0:frame_number] = wu_list[0]
        wu_mat[:, frame_number:frame_number*2] = wu_list[1]
        wu_mat[:, frame_number*2:frame_number*3] = wu_list[2]


        return  inv_prec_full, wu_mat


    def build_wuw(self, frame_number, tau_frames, win_mats, sdw=None):
        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])

        prec = bm.zeros(sdw, sdw, frame_number)

        for win_index, win_mat in enumerate(win_mats):
            bm.dot_mm_plus_equals(win_mat.T, win_mat, target_bm=prec,
                                  diag=float64(tau_frames[:, win_index]))

        return prec

    def build_wu(self, frame_number, tau_frames, win_mats, sdw=None):
        if sdw is None:
            sdw = max([ win_mat.l + win_mat.u for win_mat in win_mats ])

        wu_list = []

        for win_index, win_mat in enumerate(win_mats):
            temp_wu =  bm.zeros(sdw, sdw, frame_number)
            bm.dot_mm_plus_equals(win_mat.T, win_mats[0], target_bm=temp_wu,
                                  diag=float64(tau_frames[:, win_index]))
            wu_list.append(temp_wu.full())

        return  wu_list

    def build_win_mats(self, windows, frames):
        win_mats = []
        for l, u, win_coeff in windows:
            assert l >= 0 and u >= 0
            assert len(win_coeff) == l + u + 1
            win_coeffs = np.tile(np.reshape(win_coeff, (l + u + 1, 1)), frames)
            win_mat = bm.band_c_bm(u, l, win_coeffs).T
            win_mats.append(win_mat)

        return win_mats
