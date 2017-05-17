import os

# top merlin directory
merlin_dir="/afs/inf.ed.ac.uk/group/cstr/projects/phd/s1432486/work/test/merlin"

# tools directory
world=merlin_dir+"/tools/bin/WORLD"
sptk=merlin_dir+"/tools/bin/SPTK-3.9"

# input audio directory
wav_dir=merlin_dir+"/egs/slt_arctic/s1/slt_arctic_full_data/wav"

# Output features directory
out_dir=merlin_dir+"/egs/slt_arctic/s1/slt_arctic_full_data/feat"

sp_dir=os.path.join(out_dir,'sp')
mgc_dir=os.path.join(out_dir,'mgc')
ap_dir=os.path.join(out_dir,'ap')
bap_dir=os.path.join(out_dir,'bap')
f0_dir= os.path.join(out_dir,'f0')
lf0_dir= os.path.join(out_dir,'lf0')

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

if not os.path.exists(sp_dir):
    os.mkdir(sp_dir)

if not os.path.exists(mgc_dir):
    os.mkdir(mgc_dir)

if not os.path.exists(bap_dir):
    os.mkdir(bap_dir)

if not os.path.exists(f0_dir):
    os.mkdir(f0_dir)

if not os.path.exists(lf0_dir):
    os.mkdir(lf0_dir)

# initializations
fs=16000

if fs == 16000:
    nFFTHalf = 1024
    alpha = 0.58

if fs == 16000:
    nFFTHalf = 2048
    alpha = 0.77

wav_files = []
def get_wav_filelist(wav_dir):
    for file in os.listdir(wav_dir):
        whole_filepath = os.path.join(wav_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".wav"):
            wav_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            wav_files.append(get_wav_filelist(whole_filepath))

get_wav_filelist(wav_dir)
#bap order depends on sampling freq.
mcsize=59

for file in wav_files:
    if file is not None:
        file_id = os.path.basename(file).split(".")[0]
        ### WORLD ANALYSIS -- extract vocoder parameters ###
        ### extract f0, sp, ap ###
        world_analysis_cmd = "%s %s %s %s %s" %(os.path.join(world,'analysis'), \
                                                file,
                                             os.path.join(f0_dir,file_id+'.f0'), \
                                             os.path.join(sp_dir, file_id + '.sp'), \
                                             os.path.join(bap_dir, file_id + '.bapd'))
        os.system(world_analysis_cmd)

        ### convert f0 to lf0 ###
        sptk_x2x_da_cmd = "%s +da %s > %s"%(os.path.join(sptk,'x2x'),\
                                          os.path.join(f0_dir,file_id+'.f0'), \
                                          os.path.join(f0_dir, file_id + '.f0a'))
        os.system(sptk_x2x_da_cmd)

        sptk_x2x_af_cmd = "%s +af %s | %s > %s " % (os.path.join(sptk, 'x2x'), \
                                              os.path.join(f0_dir, file_id + '.f0a'), \
                                              os.path.join(sptk,'sopr')+' -magic 0.0 -LN -MAGIC -1.0E+10',\
                                              os.path.join(lf0_dir,file_id+'.lf0'))
        os.system(sptk_x2x_af_cmd)

        ### convert sp to mgc ###
        sptk_x2x_df_cmd1 = "%s +df %s | %s | %s >%s" % (os.path.join(sptk, 'x2x'), \
                                                    os.path.join(sp_dir, file_id + '.sp'), \
                                                    os.path.join(sptk, 'sopr') + ' -R -m 32768.0', \
                                                    os.path.join(sptk, 'mcep')+' -a '+str(alpha)+' -m '+ str(mcsize)+' -l '+str(nFFTHalf)+' -e 1.0E-8 -j 0 -f 0.0 -q 3 ',\
                                                    os.path.join(mgc_dir,file_id+'.mgc'))
        os.system(sptk_x2x_df_cmd1)

        ### convert bapd to bap ###
        sptk_x2x_df_cmd2 = "%s +df %s > %s "%(os.path.join(sptk,"x2x"),\
                                            os.path.join(bap_dir,file_id+".bapd"),\
                                            os.path.join(bap_dir,file_id+'.bap'))
        os.system(sptk_x2x_df_cmd2)
    
