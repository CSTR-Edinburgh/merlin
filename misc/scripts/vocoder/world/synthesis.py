import os
import sys

if len(sys.argv)!=6:
    print("Usage: ")
    print("python synthesis.py <path_to_merlin_dir> <path_to_feat_dir> <path_to_out_wav_dir> <sampling rate> <file_id_list>")
    sys.exit(1)

# top merlin directory
merlin_dir = sys.argv[1]

# input feat directory
feat_dir = sys.argv[2]

# Output audio directory
out_dir = sys.argv[3]

# initializations
fs = int(sys.argv[4])

# file ID list
file_id_scp = sys.argv[5]

# feat directories
mgc_dir = os.path.join(feat_dir, 'mgc')
bap_dir = os.path.join(feat_dir, 'bap')
lf0_dir = os.path.join(feat_dir, 'lf0')

if not os.path.exists(mgc_dir):
    mgc_dir = feat_dir

if not os.path.exists(bap_dir):
    bap_dir = feat_dir

if not os.path.exists(lf0_dir):
    lf0_dir = feat_dir

# tools directory
world = os.path.join(merlin_dir, "tools/bin/WORLD")
sptk  = os.path.join(merlin_dir, "tools/bin/SPTK-3.9")

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if fs == 16000:
    nFFTHalf = 1024
    alpha = 0.58

elif fs == 22050:
    nFFTHalf = 1024
    alpha = 0.65

elif fs == 44100:
    nFFTHalf = 2048
    alpha = 0.76

elif fs == 48000:
    nFFTHalf = 2048
    alpha = 0.77

else:
    print("As of now, we don't support %d Hz sampling rate." %(fs))
    print("Please consider either downsampling to 16000 Hz or upsampling to 48000 Hz")
    sys.exit(1)

mcsize=59

# set to True if synthesizing generated files
post_filtering = False

# this coefficient depends on voice
pf_coef = 1.07 

f = open(file_id_scp)
file_id_list = [newline.strip() for newline in f.readlines()]
f.close()

for file_id in file_id_list:
    ### WORLD Re-synthesis -- reconstruction of parameters ###
    print(file_id)

    ### convert lf0 to f0 ###
    sopr_cmd = "%s -magic -1.0E+10 -EXP -MAGIC 0.0 %s | %s +fa > %s" %  (os.path.join(sptk, 'sopr'), \
                                                                             os.path.join(lf0_dir, file_id+".lf0"), \
                                                                             os.path.join(sptk, "x2x"), \
                                                                             os.path.join(out_dir, file_id+".f0a"))
    os.system(sopr_cmd)

    x2x_cmd1 = "%s +ad %s > %s" % (os.path.join(sptk, "x2x"), \
                                      os.path.join(out_dir, file_id+".f0a"), \
                                      os.path.join(out_dir, file_id+".f0"))
    os.system(x2x_cmd1)

    mgc_file = os.path.join(mgc_dir, file_id+".mgc")
    if post_filtering:
        ### post-filtering mgc ###
        mgc_file = os.path.join(out_dir, file_id+".mgc_p")
        mcpf_cmd = "%s -m %d -b %f %s > %s" % (os.path.join(sptk, "mcpf"), 
                                                mcsize, pf_coef, \
                                                os.path.join(mgc_dir, file_id+".mgc"), \
                                                os.path.join(out_dir, file_id+".mgc_p"))
        os.system(mcpf_cmd)


    ### convert mgc to sp ###
    mgc2sp_cmd = "%s -a %f -g 0 -m %d -l %d -o 2 %s | %s -d 32768.0 -P | %s +fd > %s" % (os.path.join(sptk, "mgc2sp"), 
                                                                                            alpha, mcsize, nFFTHalf, \
                                                                                            mgc_file, \
                                                                                            os.path.join(sptk, "sopr"), \
                                                                                            os.path.join(sptk, "x2x"), \
                                                                                            os.path.join(out_dir, file_id+".sp"))
    os.system(mgc2sp_cmd)

    ### convert bapd to bap ###
    x2x_cmd2 = "%s +fd %s > %s" % (os.path.join(sptk, "x2x"), \
                                     os.path.join(bap_dir, file_id+".bap"), \
                                     os.path.join(out_dir, file_id+".bapd"))
    os.system(x2x_cmd2)

    # Final synthesis using WORLD
    synth_cmd = "%s %d %d %s %s %s %s" % (os.path.join(world, "synth"), \
                                            nFFTHalf, fs, \
                                            os.path.join(out_dir, file_id+".f0"), \
                                            os.path.join(out_dir, file_id+".sp"), \
                                            os.path.join(out_dir, file_id+".bapd"), \
                                            os.path.join(out_dir, file_id+".wav"))
    os.system(synth_cmd)

