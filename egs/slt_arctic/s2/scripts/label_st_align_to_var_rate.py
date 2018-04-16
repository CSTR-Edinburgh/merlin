#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: Felipe Espic

DESCRIPTION:
As Merlin works at a constant frame rate, but MagPhase runs at a variable frame
rate, it is needed to trick Merlin by warping the time durations in the label files.
This script converts the original constant-frame-rate state aligned labels to
variable-frame-rate labels, thus compensating for the frame rate missmatch.
This script acts as a workaround, so it should be removed when Merlin natively
support variable frame rates.

USE:
pyhthon <this_script_name.py> <merlin_acoustic_training_config_file.conf>

NOTES:
1.- This script needs ".shift" files extracted by MagPhase, even though they are not
    used for acoustic modelling (only .mag, .real, .imag, and .lf0 files are used in training/synthesis).

2.- The file crashlist_file stores the list of utterances that were not possible
    to convert. This could happen if for example some phonemes had no frames assigned.
    It rarelly occurs.
"""

import sys, os

this_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(this_dir + '/../../../../tools/magphase/src'))
import libutils as lu
import magphase as mp
import libaudio as la

def convert(file_id_list, in_lab_dir, in_feats_dir, fs, out_lab_dir, b_prevent_zeros=False):

    '''
    b_prevent_zeros: True if you want to ensure that all the phonemes have one frame at least.
    (not recommended, only useful when there are too many utterances crashed)
    '''

    # Conversion:
    lu.mkdir(out_lab_dir)
    v_filenames = lu.read_text_file2(file_id_list, dtype='string', comments='#')

    crashlist_file = lu.ins_pid('crash_file_list.scp')
    for filename in v_filenames:

        # Display:
        print('\nConverting lab file: ' + filename + '................................')

        # Current i/o files:
        in_lab_file   = os.path.join(in_lab_dir  , filename + '.lab')
        out_lab_file  = os.path.join(out_lab_dir , filename + '.lab')

        in_shift_file = os.path.join(in_feats_dir, filename + '.shift')


        # Debug:
        '''
        v_shift  = lu.read_binfile(in_shift_file, dim=1)
        v_n_frms = mp.get_num_of_frms_per_state(v_shift, in_lab_file, fs, b_prevent_zeros=b_prevent_zeros)
        la.convert_label_state_align_to_var_frame_rate(in_lab_file, v_n_frms, out_lab_file)
        #'''

        try:
            v_shift  = lu.read_binfile(in_shift_file, dim=1)
            v_n_frms = mp.get_num_of_frms_per_state(v_shift, in_lab_file, fs, b_prevent_zeros=b_prevent_zeros)

            la.convert_label_state_align_to_var_frame_rate(in_lab_file, v_n_frms, out_lab_file)

        except (KeyboardInterrupt, SystemExit):
            raise

        except:
            with open(crashlist_file, "a") as crashlistlog:
                crashlistlog.write(filename + '\n')

    print('Done!')


if __name__ == '__main__':

    # Parsing input arg:
    file_id_list = sys.argv[1]
    in_lab_dir   = sys.argv[2]
    in_feats_dir = sys.argv[3]
    fs           = int(sys.argv[4])
    out_lab_dir  = sys.argv[5]

    convert(file_id_list, in_lab_dir, in_feats_dir, fs, out_lab_dir, b_prevent_zeros=False)
