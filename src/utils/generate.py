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
import sys, os, subprocess, glob, commands
#from utils import GlobalCfg

import logging

#import configuration

# cannot have these outside a function - if you do that, they get executed as soon
# as this file is imported, but that can happen before the configuration is set up properly
# SPTK     = cfg.SPTK
# NND      = cfg.NND
# STRAIGHT = cfg.STRAIGHT


def run_process(args,log=True):

    logger = logging.getLogger("subprocess")
    
    # a convenience function instead of calling subprocess directly
    # this is so that we can do some logging and catch exceptions
    
    # we don't always want debug logging, even when logging level is DEBUG
    # especially if calling a lot of external functions
    # so we can disable it by force, where necessary
    if log:
        logger.debug('%s' % args)

    try:
        # the following is only available in later versions of Python
        # rval = subprocess.check_output(args)
        
        # bufsize=-1 enables buffering and may improve performance compared to the unbuffered case
        p = subprocess.Popen(args, bufsize=-1, shell=True, 
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        close_fds=True)
        # better to use communicate() than read() and write() - this avoids deadlocks
        (stdoutdata, stderrdata) = p.communicate()

        if p.returncode != 0:
            # for critical things, we always log, even if log==False
            logger.critical('exit status %d' % p.returncode )
            logger.critical(' for command: %s' % args )
            logger.critical('      stderr: %s' % stderrdata )
            logger.critical('      stdout: %s' % stdoutdata )
            raise OSError
            
        return (stdoutdata, stderrdata)
        
    except subprocess.CalledProcessError as e:
        # not sure under what circumstances this exception would be raised in Python 2.6
        logger.critical('exit status %d' % e.returncode )
        logger.critical(' for command: %s' % args )
        # not sure if there is an 'output' attribute under 2.6 ? still need to test this...
        logger.critical('  output: %s' % e.output )
        raise

    except ValueError:
        logger.critical('ValueError for %s' % args )
        raise
        
    except OSError:
        logger.critical('OSError for %s' % args )
        raise

    except KeyboardInterrupt:
        logger.critical('KeyboardInterrupt during %s' % args )
        try:
            # try to kill the subprocess, if it exists
            p.kill()
        except UnboundLocalError:
            # this means that p was undefined at the moment of the keyboard interrupt
            # (and we do nothing)
            pass

        raise KeyboardInterrupt
        


def generate_wav(gen_dir, file_id_list, cfg):
        
    logger = logging.getLogger("wav_generation")
    
    SPTK     = cfg.SPTK
#    NND      = cfg.NND
    STRAIGHT = cfg.STRAIGHT
    WORLD    = cfg.WORLD

    counter=1
    max_counter=len(file_id_list)

    ## to be moved
    pf_coef = cfg.pf_coef
    fw_coef = cfg.fw_alpha
    co_coef = cfg.co_coef
    fl_coef = cfg.fl

    counter=1
    #mgc_list = glob.glob(gen_dir + '/*' + cfg.mgc_ext)
    mgc_list = file_id_list
    max_counter = len(mgc_list)

    print   len(mgc_list)
    for mgc in mgc_list:

        logger.info('creating waveform for %4d of %4d: %s' % (counter,max_counter,mgc) )
        counter=counter+1
        mgc.strip()
        base   = os.path.basename(os.path.splitext(mgc)[0])
        files = {'sp'  : gen_dir + '/' + base + cfg.sp_ext,
                 'mgc' : gen_dir + '/' + base + cfg.mgc_ext,
                 'f0'  : gen_dir + '/' + base + '.f0',
                 'lf0' : gen_dir + '/' + base + cfg.lf0_ext,
                 'ap'  : gen_dir + '/' + base + '.ap',
                 'bap' : gen_dir + '/' + base + cfg.bap_ext,
                 'wav' : gen_dir + '/' + base + '.wav'}

        mgc_file_name = files['mgc']
        
        ### post-filtering
        if cfg.do_post_filtering:
            line = "echo 1 1 "
            for i in range(2, cfg.mgc_dim):
                line = line + str(pf_coef) + " "

            run_process('{line} | {x2x} +af > {weight}'
                        .format(line=line, x2x=SPTK['X2X'], weight=os.path.join(gen_dir, 'weight')))

            run_process('{freqt} -m {order} -a {fw} -M {co} -A 0 < {mgc} | {c2acr} -m {co} -M 0 -l {fl} > {base_r0}'
                        .format(freqt=SPTK['FREQT'], order=cfg.mgc_dim-1, fw=fw_coef, co=co_coef, mgc=files['mgc'], c2acr=SPTK['C2ACR'], fl=fl_coef, base_r0=files['mgc']+'_r0'))

            run_process('{vopr} -m -n {order} < {mgc} {weight} | {freqt} -m {order} -a {fw} -M {co} -A 0 | {c2acr} -m {co} -M 0 -l {fl} > {base_p_r0}'
                        .format(vopr=SPTK['VOPR'], order=cfg.mgc_dim-1, mgc=files['mgc'], weight=os.path.join(gen_dir, 'weight'),
                                freqt=SPTK['FREQT'], fw=fw_coef, co=co_coef, 
                                c2acr=SPTK['C2ACR'], fl=fl_coef, base_p_r0=files['mgc']+'_p_r0'))

            run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 0 -e 0 > {base_b0}'
                        .format(vopr=SPTK['VOPR'], order=cfg.mgc_dim-1, mgc=files['mgc'], weight=os.path.join(gen_dir, 'weight'),
                                mc2b=SPTK['MC2B'], fw=fw_coef, 
                                bcp=SPTK['BCP'], base_b0=files['mgc']+'_b0'))

            run_process('{vopr} -d < {base_r0} {base_p_r0} | {sopr} -LN -d 2 | {vopr} -a {base_b0} > {base_p_b0}'
                        .format(vopr=SPTK['VOPR'], base_r0=files['mgc']+'_r0', base_p_r0=files['mgc']+'_p_r0', 
                                sopr=SPTK['SOPR'], 
                                base_b0=files['mgc']+'_b0', base_p_b0=files['mgc']+'_p_b0'))
          
            run_process('{vopr} -m -n {order} < {mgc} {weight} | {mc2b} -m {order} -a {fw} | {bcp} -n {order} -s 1 -e {order} | {merge} -n {order2} -s 0 -N 0 {base_p_b0} | {b2mc} -m {order} -a {fw} > {base_p_mgc}'
                        .format(vopr=SPTK['VOPR'], order=cfg.mgc_dim-1, mgc=files['mgc'], weight=os.path.join(gen_dir, 'weight'),
                                mc2b=SPTK['MC2B'],  fw=fw_coef, 
                                bcp=SPTK['BCP'], 
                                merge=SPTK['MERGE'], order2=cfg.mgc_dim-2, base_p_b0=files['mgc']+'_p_b0',
                                b2mc=SPTK['B2MC'], base_p_mgc=files['mgc']+'_p_mgc'))

            mgc_file_name = files['mgc']+'_p_mgc'
            
        ###mgc to sp to wav
        if cfg.vocoder_type == 'STRAIGHT':
            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sp=files['sp']))
            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], f0=files['f0']))
            run_process('{x2x} +fa {f0} > {f0a}'.format(x2x=SPTK['X2X'], f0=files['f0'], f0a=files['f0'] + '.a'))

            if cfg.use_cep_ap:
                run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 0 {bap} > {ap}'
                            .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.bap_dim-1, fl=cfg.fl, bap=files['bap'], ap=files['ap']))
            else:
                run_process('{bndap2ap} {bap} > {ap}' 
                             .format(bndap2ap=STRAIGHT['BNDAP2AP'], bap=files['bap'], ap=files['ap']))

            run_process('{synfft} -f {sr} -spec -fftl {fl} -shift {shift} -sigp 0.0 -cornf 3500 -float -apfile {ap} {f0a} {sp} {wav}'
                        .format(synfft=STRAIGHT['SYNTHESIS_FFT'], sr=cfg.sr, fl=cfg.fl, shift=cfg.shift, ap=files['ap'], f0a=files['f0']+'.a', sp=files['sp'], wav=files['wav']))

            run_process('rm -f {sp} {f0} {f0a} {ap}'
                        .format(sp=files['sp'],f0=files['f0'],f0a=files['f0']+'.a',ap=files['ap']))
        elif cfg.vocoder_type == 'WORLD':        

            run_process('{sopr} -magic -1.0E+10 -EXP -MAGIC 0.0 {lf0} | {x2x} +fd > {f0}'.format(sopr=SPTK['SOPR'], lf0=files['lf0'], x2x=SPTK['X2X'], f0=files['f0']))        
            run_process('{x2x} +fd {bap} > {bapd}'.format(x2x=SPTK['X2X'], bap=files['bap'], bapd=files['bap'] + '.double'))
            
            run_process('{mgc2sp} -a {alpha} -g 0 -m {order} -l {fl} -o 2 {mgc} | {sopr} -d 32768.0 -P | {x2x} +fd > {sp}'
                        .format(mgc2sp=SPTK['MGC2SP'], alpha=cfg.fw_alpha, order=cfg.mgc_dim-1, fl=cfg.fl, mgc=mgc_file_name, sopr=SPTK['SOPR'], x2x=SPTK['X2X'], sp=files['sp']))

            run_process('{synworld} {fl} {sr} {f0} {sp} {bapd} {wav}'
                         .format(synworld=WORLD['SYNTHESIS'], fl=cfg.fl, sr=cfg.sr, f0=files['f0'], sp=files['sp'], bapd=files['bap'] + '.double', wav=files['wav']))
            
            run_process('rm -f {sp} {f0}'.format(sp=files['sp'],f0=files['f0']))

        else:
        
            logger.critical('The vocoder %s is not supported yet!\n' % cfg.vocoder_type )
        

