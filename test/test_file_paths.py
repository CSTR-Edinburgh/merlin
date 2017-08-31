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

"""Tests FilePaths class.
"""

__author__ = 'pasindu@google.com (Pasindu De Silva)'

import logging.config  # pylint: disable=unused-import
import sys
# pylint: disable=g-import-not-at-top
sys.path.append('../src')
import configuration
from utils.file_paths import FilePaths


def test_file_paths(cfg):
  """Tests FilePaths constructor.

  Args:
    cfg: Merlin configuration
  """
  cfg.GenTestList = True
  file_paths = FilePaths(cfg)

  assert len(
      file_paths.file_id_list) == 100, 'Number of files in file list incorrect'
  assert len(
      file_paths.test_id_list) == 100, 'Number of test in file list incorrect'


def test_nn_out_in_data_sets(cfg):
  """Tests Train, Valid and Test filelists.

  Args:
    cfg: Merlin configuration
  """
  file_paths = FilePaths(cfg)
  file_paths.set_label_dir(0, 'ext', 0)
  file_paths.set_label_file_list()

  train_x_file_list, train_y_file_list = file_paths.get_train_list_x_y()
  valid_x_file_list, valid_y_file_list = file_paths.get_valid_list_x_y()
  test_x_file_list, test_y_file_list = file_paths.get_test_list_x_y()

  assert len(train_x_file_list
            ) == cfg.train_file_number, 'train set x axis dimension incorrect'
  assert len(valid_x_file_list
            ) == cfg.valid_file_number, 'valid set x axis dimension incorrect'
  assert len(test_x_file_list
            ) == cfg.test_file_number, 'test set x axis dimension incorrect'

  assert len(train_y_file_list
            ) == cfg.train_file_number, 'train set y axis dimension incorrect'
  assert len(valid_y_file_list
            ) == cfg.valid_file_number, 'valid set y axis dimension incorrect'
  assert len(test_y_file_list
            ) == cfg.test_file_number, 'test set y axis dimension incorrect'


def test_label_file_lists(cfg):
  """Tests label filelists.

  Args:
    cfg: Merlin configuration
  """
  file_paths = FilePaths(cfg)
  file_paths.set_label_dir(0, 'ext', 0)
  file_paths.set_label_file_list()

  # Case 1: GenTestList = False and test_synth_dir = None
  assert file_paths.in_label_align_file_list[
      0] == '/tmp/label_state_align/file1.lab'
  assert file_paths.binary_label_file_list[
      0] == '/tmp/inter_module/binary_label_0/file1.lab'
  assert file_paths.nn_label_file_list[
      0] == '/tmp/inter_module/nn_no_silence_lab_ext/file1.lab'
  assert file_paths.nn_label_norm_file_list[
      0] == '/tmp/inter_module/nn_no_silence_lab_norm_ext/file1.lab'

  # Case 2: GenTestList = True and test_synth_dir = None
  cfg.GenTestList = True
  file_paths = FilePaths(cfg)
  file_paths.set_label_dir(0, 'ext', 0)
  file_paths.set_label_file_list()
  assert file_paths.in_label_align_file_list[
      0] == '/tmp/label_state_align/test1.lab'
  assert file_paths.binary_label_file_list[
      0] == '/tmp/inter_module/binary_label_0/test1.lab'
  assert file_paths.nn_label_file_list[
      0] == '/tmp/inter_module/nn_no_silence_lab_ext/test1.lab'
  assert file_paths.nn_label_norm_file_list[
      0] == '/tmp/inter_module/nn_no_silence_lab_norm_ext/test1.lab'

  # Case 3: GenTestList = True and test_synth_dir = test_synth
  cfg.GenTestList = True
  cfg.test_synth_dir = 'test_synth'
  file_paths = FilePaths(cfg)
  file_paths.set_label_dir(0, 'ext', 0)
  file_paths.set_label_file_list()
  assert file_paths.in_label_align_file_list[
      0] == '/tmp/label_state_align/test1.lab'
  assert file_paths.binary_label_file_list[0] == 'test_synth/test1.lab'
  assert file_paths.nn_label_file_list[0] == 'test_synth/test1.lab'
  assert file_paths.nn_label_norm_file_list[0] == 'test_synth/test1.lab'


def _get_config_file():
  cfg = configuration.cfg
  cfg.configure('test_data/test.conf')
  return cfg


def main():
  test_file_paths(_get_config_file())
  test_nn_out_in_data_sets(_get_config_file())
  test_label_file_lists(_get_config_file())


if __name__ == '__main__':
  main()
