#! /usr/bin/python2 -u
# -*- coding: utf-8 -*-
#
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to validate Merlin setup.
"""

__author__ = 'pasindu@google.com (Pasindu De Silva)'

import logging
import logging.config
import os
import sys
import configuration
from utils.utils import read_file_list

logger = logging.getLogger('validation')


class Validation(object):
  """Runs Merlin validations
  """
  _is_valid = True

  def __init__(self, cfg):
    self.cfg = cfg

  def is_valid(self):
    """Returns whether the given configuration file is valid."""

    self.validate_label_settings()
    self.validate_acoustic_files()
    return self._is_valid;

  def validate_label_settings(self):
    if self.cfg.label_style != 'HTS':
      self._is_valid = False
      logging.error(
          'Only HTS-style labels are now supported as input to Merlin')

  def validate_acoustic_files(self):
    """Validates that acoustic features exists in given path.

    Args:
      cfg: Merlin configuration.
    """
    file_types_to_check = [
        {
            'name': 'mgc',
            'dir': self.cfg.in_mgc_dir,
            'ext': self.cfg.mgc_ext
        },
        {
            'name': 'bap',
            'dir': self.cfg.in_bap_dir,
            'ext': self.cfg.bap_ext
        },
        {
            'name': 'lf0',
            'dir': self.cfg.in_lf0_dir,
            'ext': self.cfg.lf0_ext
        },
        {
            'name': 'label_align',
            'dir': self.cfg.in_label_align_dir,
            'ext': self.cfg.lab_ext
        },
    ]

    file_ids = read_file_list(self.cfg.file_id_scp)
    actual_total = len(file_ids)

    expected_total = self.cfg.train_file_number + self.cfg.valid_file_number + self.cfg.test_file_number

    if expected_total > actual_total:
      logger.error('Expected %d files but found %d files', expected_total,
                   actual_total)

    for file_id in file_ids:
      for path_info in file_types_to_check:
        path = '%s/%s%s' % (path_info['dir'], file_id, path_info['ext'])
        if not os.path.exists(path):
          self._is_valid = False
          logger.error('File id %s missing feature %s at %s', file_id,
                       path_info['name'], path)


def main(args):
  if len(args) <= 1:
    sys.stderr.write('Usage - python src/validation path_to_conf1 path_to_conf2 ...\n')
    exit(1)

  for config_file in args[1:]:

    logging.info('Validating %s configuration.', config_file)

    cfg = configuration.cfg
    cfg.configure(config_file)
    validation = Validation(cfg)

    if validation.is_valid():
      logging.info('Configuration file %s passed validation checks.', config_file)
    else:
      logging.error('Configuration file %s contains errors.', config_file)

if __name__ == '__main__':
  main(sys.argv)
