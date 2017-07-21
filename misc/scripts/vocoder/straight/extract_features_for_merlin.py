import os
import sys
import shutil
import glob
import time
import multiprocessing as mp

if len(sys.argv)!=5:
    print("Usage: ")
    print("python extract_features_for_merlin.py <path_to_merlin_dir> <path_to_wav_dir> <path_to_feat_dir> <sampling rate>")
    sys.exit(1)

# top merlin directory
merlin_dir = sys.argv[1]

# input audio directory
wav_dir = sys.argv[2]

# Output features directory
out_dir = sys.argv[3]

# initializations
fs = int(sys.argv[4])

# tools directory
straight = os.path.join(merlin_dir, "tools/bin/straight")
sptk     = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")

raw_dir = os.path.join(out_dir, 'raw' )
sp_dir  = os.path.join(out_dir, 'sp' )
mgc_dir = os.path.join(out_dir, 'mgc')
bap_dir = os.path.join(out_dir, 'bap')
ap_dir  = os.path.join(out_dir, 'ap')
f0_dir  = os.path.join(out_dir, 'f0' )
lf0_dir = os.path.join(out_dir, 'lf0')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(raw_dir):
    os.mkdir(raw_dir)

if not os.path.exists(sp_dir):
    os.mkdir(sp_dir)

if not os.path.exists(mgc_dir):
    os.mkdir(mgc_dir)

if not os.path.exists(bap_dir):
    os.mkdir(bap_dir)

if not os.path.exists(ap_dir):
    os.mkdir(ap_dir)

if not os.path.exists(f0_dir):
    os.mkdir(f0_dir)

if not os.path.exists(lf0_dir):
    os.mkdir(lf0_dir)

if fs == 16000:
    nFFT = 1024
    alpha = 0.58

elif fs == 48000:
    nFFT = 4096
    alpha = 0.77

else:
    print("As of now, we don't support %d Hz sampling rate." %(fs))
    print("Please consider either downsampling to 16000 Hz or upsampling to 48000 Hz")
    sys.exit(1)

mcsize = 59
order = 24
nFFTHalf = 1 + nFFT / 2
fshift = 5

def get_wav_filelist(wav_dir):
    wav_files = []
    for file in os.listdir(wav_dir):
        whole_filepath = os.path.join(wav_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".wav"):
            wav_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            wav_files += get_wav_filelist(whole_filepath)

    wav_files.sort()

    return wav_files

def process(filename):
    '''
    The function decomposes a wav file into F0, mel-cepstral coefficients, and band aperiodicity
    :param filename: path to wav file
    :return: .lf0, .mgc and .bap files
    '''
    file_id = os.path.basename(filename).split(".")[0]
    print(file_id)

    sox_wav_2_raw_cmd = 'sox %s -b 16 -c 1 -r %s -t raw %s' % (filename,\
                                                               fs,\
                                                               os.path.join(raw_dir, file_id + '.raw'))
    os.system(sox_wav_2_raw_cmd)

    ### STRAIGHT ANALYSIS -- extract vocoder parameters ###
    ### extract f0, sp, ap ###

    straight_f0_analysis_cmd = "%s -nmsg -maxf0 400 -uf0 400 -minf0 50 -lf0 50 -f0shift %s -f %s -raw %s %s" % (os.path.join(straight, 'tempo'), \
                                                                                                        fshift, fs, \
                                                                                                        os.path.join(raw_dir, file_id + '.raw'), \
                                                                                                        os.path.join(f0_dir, file_id + '.f0'))
    os.system(straight_f0_analysis_cmd)

    straight_ap_analysis_cmd = "%s -nmsg -f %s -fftl %s -apord %s -shift %s -f0shift %s -float -f0file %s -raw %s %s" % (os.path.join(straight, 'straight_bndap'),\
                                                                                                                          fs, nFFT, nFFTHalf, fshift, fshift,\
                                                                                                                          os.path.join(f0_dir, file_id + '.f0'), \
                                                                                                                          os.path.join(raw_dir, file_id + '.raw'), \
                                                                                                                          os.path.join(ap_dir, file_id + '.ap'))
    os.system(straight_ap_analysis_cmd)

    straight_sp_analysis_cmd = "%s -nmsg -f %s -fftl %s -apord %s -shift %s -f0shift %s -order %s -f0file %s -pow -float -raw %s %s" % (os.path.join(straight, 'straight_mcep'),\
                                                                  fs, nFFT, nFFTHalf, fshift, fshift, mcsize, \
    os.path.join(f0_dir,file_id + '.f0'), \
    os.path.join(raw_dir,file_id + '.raw'), \
    os.path.join(sp_dir,file_id + '.sp'))

    os.system(straight_sp_analysis_cmd)

    ### convert f0 to lf0 ###

    sptk_x2x_af_cmd = "%s +af %s | %s > %s " % (os.path.join(sptk, 'x2x'), \
                                                os.path.join(f0_dir, file_id + '.f0'), \
                                                os.path.join(sptk, 'sopr') + ' -magic 0.0 -LN -MAGIC -1.0E+10', \
                                                os.path.join(lf0_dir, file_id + '.lf0'))
    os.system(sptk_x2x_af_cmd)

    ### convert sp to mgc ###
    sptk_mcep = "%s -a %s -m %s -l %s -e 1.0E-8 -j 0 -f 0.0 -q 3 %s > %s" % (os.path.join(sptk, 'mcep'),\
                                       alpha, mcsize, nFFT,\
                                                                             os.path.join(sp_dir, file_id+'.sp'),\
                                                                             os.path.join(mgc_dir, file_id+'.mgc'))
    os.system(sptk_mcep)

    ### convert ap to bap ###
    sptk_mcep = "%s -a %s -m %s -l %s -e 1.0E-8 -j 0 -f 0.0 -q 1 %s > %s" % (os.path.join(sptk, 'mcep'),\
                                       alpha, order, nFFT,\
                                                                             os.path.join(ap_dir, file_id+'.ap'),\
                                                                             os.path.join(bap_dir, file_id+'.bap'))

    os.system(sptk_mcep)

print("--- Feature extraction started ---")
start_time = time.time()

# get wav files list
wav_files = get_wav_filelist(wav_dir)

# do multi-processing
pool = mp.Pool(mp.cpu_count())
pool.map(process, wav_files)

# clean temporal files
shutil.rmtree(raw_dir, ignore_errors=True)
shutil.rmtree(sp_dir, ignore_errors=True)
shutil.rmtree(f0_dir, ignore_errors=True)
shutil.rmtree(ap_dir, ignore_errors=True)

print("You should have your features ready in: "+out_dir)    

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- Feature extraction completion time: %d min. %d sec ---" % (m, s)))

