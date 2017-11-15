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

import sys
from os import path
from ConfigParser import SafeConfigParser

def parse_config_file(config_file):

    # Reading config file:
    config = SafeConfigParser()
    config.read(config_file)

    # Importing MagPhase:
    merlin = config.get('DEFAULT', 'Merlin')
    sys.path.append(path.join(merlin, 'tools', 'magphase', 'src'))
    global lu, la, mp
    import libutils as lu
    import libaudio as la
    import magphase as mp

    # Getting info:
    in_feats_dir = config.get('Paths' , 'in_acous_feats_dir')
    file_id_list = config.get('Paths' , 'file_id_list')
    in_lab_dir   = config.get('Labels', 'label_align_orig_const_rate')
    out_lab_dir  = config.get('Labels', 'label_align')
    fs = int(config.get('Waveform' , 'samplerate'))

    return file_id_list, in_lab_dir, in_feats_dir, fs, out_lab_dir


if __name__ == '__main__':

    # Parsing input arg:
    config_file = sys.argv[1]

    # Constants:
    b_prevent_zeros = False # True if you want to ensure that all the phonemes have one frame at least.
                            # (not recommended, only usful when there are too many utterances crashed)

    # Parsing config file:
    file_id_list, in_lab_dir, in_feats_dir, fs, out_lab_dir = parse_config_file(config_file)

    # Conversion:
    lu.mkdir(out_lab_dir)
    v_filenames = lu.read_text_file2(file_id_list, dtype='string', comments='#')
    n_files = len(v_filenames)
    
    crashlist_file = lu.ins_pid('crash_file_list.scp')

    for filename in v_filenames:

        # Display:
        print('\nConverting lab file: ' + filename + '................................')
        
        # Current i/o files:
        in_lab_file   = path.join(in_lab_dir  , filename + '.lab')
        out_lab_file  = path.join(out_lab_dir , filename + '.lab')
        in_shift_file = path.join(in_feats_dir, filename + '.shift')

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
