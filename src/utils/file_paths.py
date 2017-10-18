# -*- coding: utf-8 -*-
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Class to handle file paths.
"""
__author__ = 'pasindu@google.com (Pasindu De Silva)'

import os
from .utils import prepare_file_path_list
from .utils import read_file_list


class FilePaths(object):
  _NORM_INFO_FILE_NAME = 'norm_info_%s_%d_%s.dat'
  nn_cmp_dir = ''
  nn_cmp_norm_dir = ''
  model_dir = ''
  gen_dir = ''
  inter_data_dir = ''
  norm_info_file = ''
  var_dir = ''
  file_id_list = []
  test_id_list = []
  binary_label_dir = ''
  nn_label_dir = ''
  nn_label_norm_dir = ''
  bottleneck_features = ''
  binary_label_file_list = []
  nn_label_file_list = []
  nn_label_norm_file_list = []
  in_label_align_file_list = []
  dur_file_list = []
  seq_dur_file_list = []
  nn_cmp_norm_file_list = []

  def __init__(self, cfg):
    self.cfg = cfg

    self.inter_data_dir = cfg.inter_data_dir
    if not os.path.exists(self.inter_data_dir):
      os.makedirs(self.inter_data_dir)

    self.nn_cmp_dir = os.path.join(
        self.inter_data_dir,
        'nn' + self.cfg.combined_feature_name + '_' + str(self.cfg.cmp_dim))
    self.nn_cmp_norm_dir = os.path.join(
        self.inter_data_dir, 'nn_norm' + self.cfg.combined_feature_name + '_' +
        str(self.cfg.cmp_dim))
    self.model_dir = os.path.join(self.cfg.work_dir, 'nnets_model')
    self.gen_dir = os.path.join(self.cfg.work_dir, 'gen')
    self.file_id_list = read_file_list(self.cfg.file_id_scp)
    self.bottleneck_features = os.path.join(self.gen_dir, 'bottleneck_features')

    if self.cfg.GenTestList:
      self.test_id_list = read_file_list(cfg.test_id_scp)

    self.norm_info_file = os.path.join(self.inter_data_dir,
                                       self._NORM_INFO_FILE_NAME %
                                       (cfg.combined_feature_name, cfg.cmp_dim,
                                        cfg.output_feature_normalisation))

    ### save acoustic normalisation information for normalising the features back
    self.var_dir = os.path.join(self.inter_data_dir, 'var')
    if not os.path.exists(self.var_dir):
      os.makedirs(self.var_dir)

    if self.cfg.MAKEDUR:
      self.dur_file_list = prepare_file_path_list(
          self.file_id_list, self.cfg.in_dur_dir, self.cfg.dur_ext)

    if self.cfg.network_type=="S2S":
      self.seq_dur_file_list  = prepare_file_path_list(
          self.file_id_list, self.cfg.in_seq_dur_dir, self.cfg.dur_ext)

    self.nn_cmp_norm_file_list = prepare_file_path_list(
        self.file_id_list, self.nn_cmp_norm_dir, self.cfg.cmp_ext)

  def get_nnets_file_name(self):
    return '%s/%s.model' % (self.model_dir, self.cfg.model_file_name)

  def get_temp_nn_dir_name(self):
    return self.cfg.model_file_name

  def get_var_dic(self):
    var_file_dict = {}
    for feature_name in list(self.cfg.out_dimension_dict.keys()):
      var_file_dict[feature_name] = self._get_var_file_name(feature_name)
    return var_file_dict

  def get_train_list_x_y(self):
    start = 0
    end = self.cfg.train_file_number
    return self.nn_label_norm_file_list[start:end], self.nn_cmp_norm_file_list[
        start:end]

  def get_valid_list_x_y(self):
    start = self.cfg.train_file_number
    end = self.cfg.train_file_number + self.cfg.valid_file_number
    return self.nn_label_norm_file_list[start:end], self.nn_cmp_norm_file_list[
        start:end]

  def get_test_list_x_y(self):
    start = self.cfg.train_file_number + self.cfg.valid_file_number
    end = self.cfg.train_file_number + self.cfg.valid_file_number + self.cfg.test_file_number
    return self.nn_label_norm_file_list[start:end], self.nn_cmp_norm_file_list[
        start:end]

  def _get_var_file_name(self, feature_name):
    return os.path.join(
        self.var_dir,
        feature_name + '_' + str(self.cfg.out_dimension_dict[feature_name]))

  def set_label_dir(self, dimension, suffix, lab_dim):
    self.binary_label_dir = os.path.join(self.inter_data_dir,
                                         'binary_label_' + str(dimension))
    self.nn_label_dir = os.path.join(self.inter_data_dir,
                                     'nn_no_silence_lab_' + suffix)
    self.nn_label_norm_dir = os.path.join(self.inter_data_dir,
                                          'nn_no_silence_lab_norm_' + suffix)

    label_norm_file = 'label_norm_%s_%d.dat' % (self.cfg.label_style, lab_dim)
    self.label_norm_file = os.path.join(self.inter_data_dir, label_norm_file)

    out_feat_dir = os.path.join(self.inter_data_dir, 'binary_label_' + suffix)
    self.out_feat_file_list = prepare_file_path_list(
        self.file_id_list, out_feat_dir, self.cfg.lab_ext)

  def get_nn_cmp_file_list(self):
    return prepare_file_path_list(self.file_id_list, self.nn_cmp_dir,
                                  self.cfg.cmp_ext)

  def get_nn_cmp_norm_file_list(self):
    return self.nn_cmp_norm_file_list

  def get_lf0_file_list(self):
    return prepare_file_path_list(self.file_id_list, self.cfg.in_lf0_dir,
                                  self.cfg.lf0_ext)

  def set_label_file_list(self):
    if self.cfg.GenTestList:
      self.in_label_align_file_list = prepare_file_path_list(
          self.test_id_list, self.cfg.in_label_align_dir, self.cfg.lab_ext,
          False)
    else:
      self.in_label_align_file_list = prepare_file_path_list(
          self.file_id_list, self.cfg.in_label_align_dir, self.cfg.lab_ext,
          False)

    if self.cfg.GenTestList and self.cfg.test_synth_dir != 'None' and not self.cfg.VoiceConversion:
      test_binary_file_list = self._prepare_test_binary_label_file_path_list(
          self.cfg.test_synth_dir)
      test_file_list = self._prepare_test_label_file_path_list(
          self.cfg.test_synth_dir)
      self.binary_label_file_list = test_binary_file_list
      self.nn_label_file_list = test_file_list
      self.nn_label_norm_file_list = test_file_list
    elif self.cfg.GenTestList:
      self.binary_label_file_list = self._prepare_test_label_file_path_list(
          self.binary_label_dir)
      self.nn_label_file_list = self._prepare_test_label_file_path_list(
          self.nn_label_dir)
      self.nn_label_norm_file_list = self._prepare_test_label_file_path_list(
          self.nn_label_norm_dir)
    else:
      self.binary_label_file_list = self._prepare_file_label_file_path_list(
          self.binary_label_dir)
      self.nn_label_file_list = self._prepare_file_label_file_path_list(
          self.nn_label_dir)
      self.nn_label_norm_file_list = self._prepare_file_label_file_path_list(
          self.nn_label_norm_dir)

  def _prepare_file_label_file_path_list(self, list_dir):
    return prepare_file_path_list(self.file_id_list, list_dir, self.cfg.lab_ext)

  def _prepare_test_label_file_path_list(self, list_dir):
    return prepare_file_path_list(self.test_id_list, list_dir, self.cfg.lab_ext)

  def _prepare_test_binary_label_file_path_list(self, list_dir):
    return prepare_file_path_list(self.test_id_list, list_dir, self.cfg.lab_ext+'bin')
