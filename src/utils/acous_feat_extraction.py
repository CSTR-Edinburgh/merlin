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

#/usr/bin/python -u

'''
This script assumes c-version STRAIGHT which is not available to public. Please use your
own vocoder to replace this script.
'''
import sys, os
#from utils import GlobalCfg
import logging


def feat_extraction_magphase(in_wav_dir, file_id_list, cfg, logger, b_multiproc=False):
    sys.path.append(cfg.magphase_bindir)
    import libutils as lu
    import magphase as mp

    def feat_extraction_magphase_one_file(in_wav_dir, file_name_token, acous_feats_dir, cfg, logger):

        # Logging:
        logger.info('Analysing waveform: %s.wav' % (file_name_token))

        # File setup:
        wav_file = os.path.join(in_wav_dir, file_name_token + '.wav')

        # Feat extraction:
        mp.analysis_for_acoustic_modelling(wav_file, out_dir=acous_feats_dir, mag_dim=cfg.mag_dim,
                                                            phase_dim=cfg.real_dim, b_const_rate=cfg.magphase_const_rate)

        return


    if b_multiproc:
        lu.run_multithreaded(feat_extraction_magphase_one_file, in_wav_dir, file_id_list, cfg.acous_feats_dir, cfg, logger)
    else:
        for file_name_token in file_id_list:
            feat_extraction_magphase_one_file(in_wav_dir, file_name_token, cfg.acous_feats_dir, cfg, logger)


    return


def acous_feat_extraction(in_wav_dir, file_id_list, cfg):

    logger = logging.getLogger("acous_feat_extraction")

    ## MagPhase Vocoder:
    if cfg.vocoder_type=='MAGPHASE':
        feat_extraction_magphase(in_wav_dir, file_id_list, cfg, logger)


    # TODO: Add WORLD and STRAIGHT

    # If vocoder is not supported:
    else:
        logger.critical('The vocoder %s is not supported for feature extraction yet!\n' % cfg.vocoder_type )
        raise

    return