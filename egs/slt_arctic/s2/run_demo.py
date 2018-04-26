#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: Felipe Espic
"""
from shutil import copytree, copy2
import scripts.label_st_align_to_var_rate as ltvr
from os.path import join, dirname, realpath, isdir
import sys
this_dir = dirname(realpath(__file__))
sys.path.append(realpath(this_dir + '/../../../tools/magphase/src'))
import libutils as lu
import magphase as mp
import configparser # Install it with pip (it's not the same as 'ConfigParser' (old version))
from subprocess import call


def feat_extraction(in_wav_dir, file_name_token, out_feats_dir, d_opts):

    # Display:
    print("\nAnalysing file: " + file_name_token + '.wav............................')

    # File setup:
    wav_file = join(in_wav_dir, file_name_token + '.wav')

    mp.analysis_for_acoustic_modelling(wav_file, out_feats_dir,
                                        mag_dim=d_opts['mag_dim'],
                                        phase_dim=d_opts['phase_dim'],
                                        b_const_rate=d_opts['b_const_rate'])
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

def mod_acoustic_config(parser, merlin_path, exper_path, exper_type, d_mp_opts):
    parser['DEFAULT']['Merlin']   = merlin_path
    parser['DEFAULT']['TOPLEVEL'] = exper_path

    parser['Outputs']['mag' ] = '%d' %  d_mp_opts['mag_dim']
    parser['Outputs']['dmag'] = '%d' % (d_mp_opts['mag_dim']*3)

    parser['Outputs']['real' ] = '%d' %  d_mp_opts['phase_dim']
    parser['Outputs']['imag' ] = '%d' %  d_mp_opts['phase_dim']
    parser['Outputs']['dreal'] = '%d' % (d_mp_opts['phase_dim']*3)
    parser['Outputs']['dimag'] = '%d' % (d_mp_opts['phase_dim']*3)

    if exper_type=='full':
        parser['Architecture']['hidden_layer_size'] = "[1024, 1024, 1024, 1024, 1024, 1024]"
        parser['Architecture']['hidden_layer_type'] = "['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']"
        parser['Architecture']['model_file_name']   = "feed_forward_6_tanh"

    if d_mp_opts['b_const_rate']:
        parser['Labels']['label_align'] = '%(TOPLEVEL)s/acoustic_model/data/label_state_align'

    parser = mod_number_of_utts(parser, exper_type)

    return parser

def mod_duration_config(parser, merlin_path, exper_path, exper_type, d_mp_opts):
    parser['DEFAULT']['Merlin']   = merlin_path
    parser['DEFAULT']['TOPLEVEL'] = exper_path

    if exper_type=='full':
        parser['Architecture']['hidden_layer_size'] = "[1024, 1024, 1024, 1024, 1024, 1024]"
        parser['Architecture']['hidden_layer_type'] = "['TANH', 'TANH', 'TANH', 'TANH', 'TANH', 'TANH']"
        parser['Architecture']['model_file_name']   = "feed_forward_6_tanh"

    if d_mp_opts['b_const_rate']:
        parser['Labels']['label_align'] = '%(TOPLEVEL)s/acoustic_model/data/label_state_align'

    parser = mod_number_of_utts(parser, exper_type)

    return parser

def mod_number_of_utts(parser, exper_type):

    if exper_type=='full':
        parser['Paths']['file_id_list'] = '%(data)s/file_id_list_full.scp'
        parser['Data']['train_file_number'] = '%d' % 1000
        parser['Data']['valid_file_number'] = '%d' % 66
        parser['Data']['test_file_number' ] = '%d' % 65

    elif exper_type=='demo':
        pass

    return parser


if __name__ == '__main__':

    # INPUT:===================================================================================================

    # Experiment type:-----------------------------------------------------------------------
    exper_type = 'demo'  #  'demo' (50 training utts) or 'full' (1k training utts)

    # Steps:---------------------------------------------------------------------------------
    b_download_data  = 1 # Downloads wavs and label data.
    b_setup_data     = 1 # Copies downloaded data into the experiment directory. Plus, makes a backup copy of this script.
    b_config_merlin  = 1 # Saves new configuration files for Merlin.
    b_feat_extr      = 1 # Performs acoustic feature extraction using the MagPhase vocoder
    b_conv_labs_rate = 1 # Converts the state aligned labels to variable rate if running in variable frame rate mode (d_mp_opts['b_const_rate'] = False)
    b_dur_train      = 1 # Merlin: Training of duration model.
    b_acous_train    = 1 # Merlin: Training of acoustic model.
    b_dur_syn        = 1 # Merlin: Generation of state durations using the duration model.
    b_acous_syn      = 1 # Merlin: Waveform generation for the utterances provided in ./test_synthesis/prompt-lab

    # MagPhase Vocoder:-----------------------------------------------------------------------
    d_mp_opts = {}                     # Dictionary containing internal options for the MagPhase vocoder (mp).
    d_mp_opts['mag_dim'   ] = 100       # Number of coefficients (bins) for magnitude feature M.
    d_mp_opts['phase_dim' ] = 10       # Number of coefficients (bins) for phase features R and I.
    d_mp_opts['b_const_rate'] = False  # To work in constant frame rate mode.
    d_mp_opts['l_pf_type'   ] = [ 'no', 'magphase', 'merlin'] #  List containing the postfilters to apply during waveform generation.
    # You need to choose at least one: 'magphase' (magphase-tailored postfilter), 'merlin' (Merlin's style postfilter), 'no' (no postfilter)

    b_feat_ext_multiproc      = 1     # Acoustic feature extraction done in multiprocessing mode (faster).


    # PROCESS:===================================================================================================
    # Pre setup:-------------------------------------------------------------------------------
    exper_name  = 'slt_arctic_magphase_%s_mag_dim_%s_phase_dim_%d_const_rate_%d' % (exper_type, d_mp_opts['mag_dim'], d_mp_opts['phase_dim'], d_mp_opts['b_const_rate'])
    exper_path  = join(this_dir, 'experiments' , exper_name)
    merlin_path = realpath(this_dir + '/../../..')
    submit_path     = join(this_dir, 'scripts', 'submit.sh')
    run_merlin_path = join(merlin_path, 'src', 'run_merlin.py')
    dur_model_conf_path   = join(exper_path, 'duration_model', 'conf')
    acous_model_conf_path = join(exper_path, 'acoustic_model'   , 'conf')

    # Build config parsers:-------------------------------------------------------------------

    # Duration training config file:
    pars_dur_train = open_config_file(join(this_dir, 'conf_base', 'dur_train_base.conf'))
    pars_dur_train = mod_duration_config(pars_dur_train, merlin_path, exper_path, exper_type, d_mp_opts)

    # Duration synthesis:
    pars_dur_synth = open_config_file(join(this_dir, 'conf_base', 'dur_synth_base.conf'))
    pars_dur_synth = mod_duration_config(pars_dur_synth, merlin_path, exper_path, exper_type, d_mp_opts)

    # Acoustic training:
    pars_acous_train = open_config_file(join(this_dir, 'conf_base', 'acous_train_base.conf'))
    pars_acous_train = mod_acoustic_config(pars_acous_train, merlin_path, exper_path, exper_type, d_mp_opts)

    # Acoustic synth:
    pars_acous_synth = open_config_file(join(this_dir, 'conf_base', 'acous_synth_base.conf'))
    pars_acous_synth = mod_acoustic_config(pars_acous_synth, merlin_path, exper_path, exper_type, d_mp_opts)

    # Download Data:--------------------------------------------------------------------------
    if b_download_data:
        data_zip_file = join(this_dir, 'slt_arctic_%s_data.zip' % exper_type)
        call(['wget', 'http://felipeespic.com/depot/databases/merlin_demos/slt_arctic_%s_data.zip' % exper_type , '-O', data_zip_file])
        call(['unzip', '-o', '-q', data_zip_file, '-d', this_dir])

    # Setup Data:-----------------------------------------------------------------------------
    if b_setup_data:
        copytree(join(this_dir, 'slt_arctic_' + exper_type + '_data', 'exper'), exper_path)
        copy2(__file__, join(exper_path, 'run_demo_backup.py'))

    # Configure Merlin:-----------------------------------------------------------------------
    if b_config_merlin:
        save_config(pars_dur_train,   join(dur_model_conf_path  , 'dur_train.conf'))
        save_config(pars_dur_synth,   join(dur_model_conf_path  , 'dur_synth.conf'))
        save_config(pars_acous_train, join(acous_model_conf_path, 'acous_train.conf'))
        save_config(pars_acous_synth, join(acous_model_conf_path, 'acous_synth.conf'))

        copy2(join(this_dir, 'conf_base', 'logging_config.conf'), join(exper_path, 'acoustic_model', 'conf', 'logging_config.conf'))

    # Read file list:
    file_id_list = pars_acous_train['Paths']['file_id_list']
    l_file_tokns = lu.read_text_file2(file_id_list, dtype='string', comments='#').tolist()
    acoustic_feats_path = pars_acous_train['Paths']['in_acous_feats_dir']

    # Acoustic Feature Extraction:-------------------------------------------------------------
    if b_feat_extr:
        # Extract features:
        lu.mkdir(acoustic_feats_path)

        if b_feat_ext_multiproc:
            lu.run_multithreaded(feat_extraction, join(exper_path, 'acoustic_model', 'data', 'wav'), l_file_tokns, acoustic_feats_path, d_mp_opts)
        else:
            for file_name_token in l_file_tokns:
                feat_extraction(join(exper_path, 'acoustic_model', 'data', 'wav'), file_name_token, acoustic_feats_path, d_mp_opts)

    # Labels Conversion to Variable Frame Rate:------------------------------------------------
    if b_conv_labs_rate and not d_mp_opts['b_const_rate']: # NOTE: The script ./script/label_st_align_to_var_rate.py can be also called from comand line directly.
        label_state_align = join(exper_path, 'acoustic_model', 'data', 'label_state_align')
        label_state_align_var_rate = pars_acous_train['Labels']['label_align']
        fs = int(pars_acous_train['Waveform']['samplerate'])
        ltvr.convert(file_id_list,label_state_align, acoustic_feats_path, fs, label_state_align_var_rate)

    # Run duration training:-------------------------------------------------------------------
    if b_dur_train:
        call([submit_path, run_merlin_path, join(dur_model_conf_path, 'dur_train.conf')])

    # Run acoustic train:----------------------------------------------------------------------
    if b_acous_train:
        call([submit_path,run_merlin_path, join(acous_model_conf_path, 'acous_train.conf')])

    # Run duration syn:------------------------------------------------------------------------
    if b_dur_syn:
        call([submit_path, run_merlin_path, join(dur_model_conf_path, 'dur_synth.conf')])

    # Run acoustic synth:----------------------------------------------------------------------
    if b_acous_syn:
        call([submit_path, run_merlin_path, join(acous_model_conf_path, 'acous_synth.conf')])


    print("Done!")






