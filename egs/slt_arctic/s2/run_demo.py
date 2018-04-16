#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: felipe
"""
import shutil
import scripts.label_st_align_to_var_rate as ltvr

import sys, os
this_dir = os.path.dirname(__file__)
sys.path.append(os.path.realpath(this_dir + '/../../../tools/magphase/src'))
import libutils as lu
import libaudio as la
import magphase as mp
import configparser # Install it with pip (it's not the same as 'ConfigParser' (old version))
import subprocess

def copytree(src_dir, l_items, dst_dir, symlinks=False, ignore=None):
    for item in l_items:
        s = os.path.join(src_dir, item)
        d = os.path.join(dst_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def feat_extraction(in_wav_dir, file_name_token, out_feats_dir, d_opts):

    # Display:
    print("\nAnalysing file: " + file_name_token + '.wav............................')

    # File setup:
    wav_file = os.path.join(in_wav_dir, file_name_token + '.wav')

    mp.analysis_compressed_type1_with_phase_comp_mcep(wav_file, out_dir=out_feats_dir, nbins_phase=d_opts['nbins_phase'], b_const_rate=d_opts['b_const_rate'])

    return


def open_config_file(configfile_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read([configfile_path])
    return parser

def save_config(parser, file_path):
    with open(file_path, 'wb') as file:
        parser.write(file)
    return

def mod_acoustic_config(parser, merlin_path, exper_path, exper_mode, d_mp_opts):
    parser['DEFAULT']['Merlin']   = merlin_path
    parser['DEFAULT']['TOPLEVEL'] = exper_path

    parser['Outputs']['real' ] = '%d' %  d_mp_opts['nbins_phase']
    parser['Outputs']['imag' ] = '%d' %  d_mp_opts['nbins_phase']
    parser['Outputs']['dreal'] = '%d' % (d_mp_opts['nbins_phase']*3)
    parser['Outputs']['dimag'] = '%d' % (d_mp_opts['nbins_phase']*3)

    parser = mod_number_of_utts(parser, exper_mode)

    return parser

def mod_duration_config(parser, merlin_path, exper_path, exper_mode):
    parser['DEFAULT']['Merlin']   = merlin_path
    parser['DEFAULT']['TOPLEVEL'] = exper_path

    parser = mod_number_of_utts(parser, exper_mode)

    return parser

def mod_number_of_utts(parser, exper_mode):

    if exper_mode=='full':
        parser['Paths']['file_id_list'] = '%(data)s/file_id_list_full.scp'
        parser['Data']['train_file_number'] = 1000
        parser['Data']['valid_file_number'] = 66
        parser['Data']['test_file_number' ] = 65

    elif exper_mode=='demo':
        pass

    return parser


if __name__ == '__main__':

    # INPUT:===================================================================================================

    # Name:
    exper_mode = 'demo' # 'full',

    # Setup:
    b_download_data  = 1
    b_setup_data     = 0
    b_config_merlin  = 0
    b_feat_extr      = 0
    b_conv_labs_rate = 0
    b_dur_train      = 0
    b_acous_train    = 0
    b_dur_syn        = 0
    b_acous_syn      = 0

    # Vocoder:
    b_feat_ext_multiproc = 1

    d_mp_opts = {'nbins_phase' : 10,
                 'b_const_rate': False,
                 'l_pf_type'   : [ 'no', 'magphase', 'merlin'] # 'magphase', 'merlin', 'no'
                 }


    exper_name = 'slt_arctic_magphase_' + exper_mode

    # PROCESS:===================================================================================================

    # Pre setup:
    exper_path  = os.path.join(this_dir, 'experiments' , exper_name)
    merlin_path = os.path.realpath(this_dir + '/../../..')

    # Build config parsers:

    # Duration training config file:
    pars_dur_train = open_config_file(os.path.join(this_dir, 'conf_base', 'dur_train_base.conf'))
    pars_dur_train = mod_duration_config(pars_dur_train, merlin_path, exper_path, exper_mode)

    # Duration synthesis:
    pars_dur_synth = open_config_file(os.path.join(this_dir, 'conf_base', 'dur_synth_base.conf'))
    pars_dur_synth = mod_duration_config(pars_dur_synth, merlin_path, exper_path, exper_mode)

    # Acoustic training:
    pars_acous_train = open_config_file(os.path.join(this_dir, 'conf_base', 'acous_train_base.conf'))
    pars_acous_train = mod_acoustic_config(pars_acous_train, merlin_path, exper_path, exper_mode, d_mp_opts)

    # Acoustic synth:
    pars_acous_synth = open_config_file(os.path.join(this_dir, 'conf_base', 'acous_synth_base.conf'))
    pars_acous_synth = mod_acoustic_config(pars_acous_synth, merlin_path, exper_path, exper_mode, d_mp_opts)

    #-----------------------------------------------------------------------------------------
    if b_download_data:
        print("\nDownloading data..............")
        subprocess.call(['wget', '-P', this_dir, 'http://felipeespic.com/depot/databases/merlin_demos/slt_arctic_%s_data.zip' % exper_mode])
        subprocess.call(['unzip', '-q', os.path.join(this_dir, 'slt_arctic_%s_data.zip' % exper_mode), '-d', this_dir])

    #-----------------------------------------------------------------------------------------

    if b_setup_data:
        print("\nMaking directories and copying data..............")
        shutil.copytree(os.path.join(this_dir, 'slt_arctic_' + exper_mode + '_data'), exper_path)
        shutil.copy2(__file__, os.path.join(exper_path, 'run_demo_backup.py'))


    #-----------------------------------------------------------------------------------------

    if b_config_merlin:
        save_config(pars_dur_train,   os.path.join(exper_path, 'duration_model', 'conf', 'dur_train.conf'))
        save_config(pars_dur_synth,   os.path.join(exper_path, 'duration_model', 'conf', 'dur_synth.conf'))
        save_config(pars_acous_train, os.path.join(exper_path, 'acoustic_model', 'conf', 'acous_train.conf'))
        save_config(pars_acous_synth, os.path.join(exper_path, 'acoustic_model', 'conf', 'acous_synth.conf'))

        shutil.copy2(os.path.join(this_dir, 'conf_base', 'logging_config.conf'), os.path.join(exper_path, 'acoustic_model', 'conf', 'logging_config.conf'))

    # Read file list:
    file_id_list = pars_acous_train['Paths']['file_id_list']
    l_file_tokns = lu.read_text_file2(os.path.join(exper_path, file_id_list), dtype='string', comments='#').tolist()
    acoustic_feats_path = os.path.join(exper_path, 'acoustic_model', 'data', 'acoustic_feats')

    #-----------------------------------------------------------------------------------------

    if b_feat_extr:
        # Extract features:
        lu.mkdir(acoustic_feats_path)

        if b_feat_ext_multiproc:
            lu.run_multithreaded(feat_extraction, os.path.join(exper_path, 'acoustic_model', 'data', 'wav'), l_file_tokns, acoustic_feats_path, d_mp_opts)
        else:
            for file_name_token in l_file_tokns:
                feat_extraction(os.path.join(exper_path, 'acoustic_model', 'data', 'wav'), file_name_token, acoustic_feats_path, d_mp_opts)

    #-----------------------------------------------------------------------------------------

    if b_conv_labs_rate:
        # NOTE: The script ./script/label_st_align_to_var_rate.py can be also called also directly from comand line.
        label_state_align = os.path.join(exper_path, 'acoustic_model', 'data', 'label_state_align')
        label_state_align_var_rate = pars_acous_train['Labels']['label_align']
        fs = int(pars_acous_train['Waveform']['samplerate'])
        ltvr.convert(file_id_list,label_state_align, acoustic_feats_path, fs, label_state_align_var_rate)


    # Run Merlin:
    submit_path     = os.path.join(this_dir, 'scripts', 'submit.sh')
    run_merlin_path = os.path.join(merlin_path, 'src', 'run_merlin.py')

    #-----------------------------------------------------------------------------------------

    # Run duration training:
    if b_dur_train:
        subprocess.call([submit_path, run_merlin_path, os.path.join(exper_path, 'duration_model', 'conf', 'dur_train.conf')])

    #-----------------------------------------------------------------------------------------

    # Run acoustic train:
    if b_acous_train:
        subprocess.call([submit_path,run_merlin_path, os.path.join(exper_path, 'acoustic_model', 'conf', 'acous_train.conf')])

    #-----------------------------------------------------------------------------------------

    # Run duration syn:
    if b_dur_syn:
        subprocess.call([submit_path, run_merlin_path, os.path.join(exper_path, 'duration_model', 'conf', 'dur_synth.conf')])

    #-----------------------------------------------------------------------------------------

    # Run acoustic synth:
    if b_acous_syn:
        subprocess.call([submit_path, run_merlin_path, os.path.join(exper_path, 'acoustic_model', 'conf', 'acous_synth.conf')])


    print("Done!")






