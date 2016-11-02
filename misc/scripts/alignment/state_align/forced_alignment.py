import os, sys
import time

from sys import argv, stderr
from subprocess import check_call, Popen, CalledProcessError, PIPE
from mean_variance_norm import MeanVarianceNorm

# string constants for various shell calls
STATE_NUM=5
F = str(0.01)
SFAC = str(5.0)
PRUNING = [str(i) for i in (250., 150., 2000.)]

MACROS = 'macros'
HMMDEFS = 'hmmdefs'
VFLOORS = 'vFloors'

##
HTKDIR = path/to/tools/htk
HCompV = os.path.join(HTKDIR, 'HCompV')
HCopy  = os.path.join(HTKDIR, 'HCopy' )
HERest = os.path.join(HTKDIR, 'HERest')
HHEd   = os.path.join(HTKDIR, 'HHEd'  )
HVite  = os.path.join(HTKDIR, 'HVite' )

class ForcedAlignment(object):

    def __init__(self):
        self.proto = None
        self.phoneme_mlf = None

    def _make_proto(self):
        ## make proto
        fid = open(self.proto, 'w')
        means = ' '.join(['0.0' for _ in xrange(39)])
        varg  = ' '.join(['1.0' for _ in xrange(39)])
        print >> fid, """~o <VECSIZE> 39 <USER>
~h "proto"
<BEGINHMM>
<NUMSTATES> 7"""
        for i in xrange(2, STATE_NUM+2):
            print >> fid, '<STATE> {0}\n<MEAN> 39\n{1}'.format(i, means)
            print >> fid, '<VARIANCE> 39\n{0}'.format(varg)
        print >> fid, """<TRANSP> 7
 0.0 1.0 0.0 0.0 0.0 0.0 0.0
 0.0 0.6 0.4 0.0 0.0 0.0 0.0
 0.0 0.0 0.6 0.4 0.0 0.0 0.0
 0.0 0.0 0.0 0.6 0.4 0.0 0.0
 0.0 0.0 0.0 0.0 0.6 0.4 0.0
 0.0 0.0 0.0 0.0 0.0 0.7 0.3
 0.0 0.0 0.0 0.0 0.0 0.0 0.0
<ENDHMM>"""
        fid.close()

        ## make vFloors
        check_call([HCompV, '-f', F, '-C', self.cfg,
                              '-S', self.train_scp,
                              '-M', self.cur_dir, self.proto])
        ## make local macro
        # get first three lines from local proto
        fid = open(os.path.join(self.cur_dir, MACROS), 'w')
        source = open(os.path.join(self.cur_dir,
                      os.path.split(self.proto)[1]), 'r')
        for _ in xrange(3):
            print >> fid, source.readline(),
        source.close()
        # get remaining lines from vFloors
        fid.writelines(open(os.path.join(self.cur_dir,
                                          VFLOORS), 'r').readlines())
        fid.close()
        ## make hmmdefs
        fid = open(os.path.join(self.cur_dir, HMMDEFS), 'w')
        for phone in open(self.phonemes, 'r'):
            source = open(self.proto, 'r')
            # ignore
            source.readline()
            source.readline()
            # the header
            print >> fid, '~h "{0}"'.format(phone.rstrip())
            # the rest
            fid.writelines(source.readlines())
            source.close()
        fid.close()

    def _read_file_list(self, file_name):

        file_lists = []
        fid = open(file_name)
        for line in fid.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            file_lists.append(line)
        fid.close()

        return  file_lists

    def _full_to_mono(self, full_file_name, mono_file_name, phoneme_dict):
        fre = open(full_file_name, 'r')
        fwe = open(mono_file_name, 'w')
        for line in fre.readlines():
            line = line.strip()
            if len(line) < 1:
                continue
            tmp_list = line.split('-')
            tmp_list = tmp_list[1].split('+')
            mono_phone = tmp_list[0]
            print >> fwe, '{0}'.format(mono_phone)
            if not phoneme_dict.has_key(mono_phone):
                phoneme_dict[mono_phone] = 1
            phoneme_dict[mono_phone] += 1
        fwe.close()
        fre.close()

    def _check_data(self, file_id_list, multiple_speaker):

        copy_scp = open(self.copy_scp, 'w')
        check_scp = open(self.train_scp, 'w')
        i = 0

        phoneme_dict = {}
        speaker_utt_dict = {}

        for file_id in file_id_list:
            wav_file = os.path.join(self.wav_dir, file_id + '.wav')
            lab_file = os.path.join(self.lab_dir, file_id + '.lab')
            mfc_file = os.path.join(self.mfc_dir, file_id + '.mfc')
            mono_lab_file = os.path.join(self.mono_lab_dir, file_id + '.lab')

            mfc_sub_dir = os.path.dirname(mfc_file)
            if os.path.exists(wav_file) and os.path.exists(lab_file):
                if not os.path.exists(mfc_sub_dir):
                    os.makedirs(mfc_sub_dir)

                print >> copy_scp, '{0} {1}'.format(wav_file, mfc_file)
                print >> check_scp, '{0}'.format(mfc_file)


                if multiple_speaker:
                    tmp_list = file_id.split('/')
                    speaker_name = tmp_list[0]
                    if not speaker_utt_dict.has_key(speaker_name):
                        speaker_utt_dict[speaker_name] = []
                    speaker_utt_dict[speaker_name].append(mfc_file)
                else:
                    if not speaker_utt_dict.has_key('only_one'):
                        speaker_utt_dict['only_one'] = []
                    speaker_utt_dict['only_one'].append(mfc_file)


                self._full_to_mono(lab_file, mono_lab_file, phoneme_dict)
        copy_scp.close()
        check_scp.close()

        fid = open(self.phonemes, 'w')
        fmap = open(self.phoneme_map, 'w')
        for phoneme in phoneme_dict.keys():
#            print   phoneme, phoneme_dict[phoneme]
            print >> fid, '{0}'.format(phoneme)
            print >> fmap, '{0} {0}'.format(phoneme)
        fmap.close()
        fid.close()

        self.phoneme_mlf = os.path.join(self.cfg_dir, 'mono_phone.mlf')
        fid = open(self.phoneme_mlf, 'w')
        print >> fid, '#!MLF!#'
        print >> fid, '"*/*.lab" -> "' + self.mono_lab_dir + '"'
        fid.close()

        return  speaker_utt_dict

    def _HCopy(self):
        """
        Compute MFCCs
        """
        # write a CFG for extracting MFCCs
        print >> open(self.cfg, 'w'), """SOURCEKIND = WAVEFORM
SOURCEFORMAT = WAVE
TARGETRATE = 50000.0
TARGETKIND = MFCC_D_A_0
WINDOWSIZE = 250000.0
PREEMCOEF = 0.97
USEHAMMING = T
ENORMALIZE = T
CEPLIFTER = 22
NUMCHANS = 20
NUMCEPS = 12"""
        check_call([HCopy, '-C', self.cfg, '-S', self.copy_scp])
        # write a CFG for what we just built
        print >> open(self.cfg, 'w'), """TARGETRATE = 50000.0
TARGETKIND = USER
WINDOWSIZE = 250000.0
PREEMCOEF = 0.97
USEHAMMING = T
ENORMALIZE = T
CEPLIFTER = 22
NUMCHANS = 20
NUMCEPS = 12"""

    def _nxt_dir(self):
        """
        Get the next HMM directory
        """
        # pass on the previously new one to the old one
        self.cur_dir = self.nxt_dir
        # increment
        self.n += 1
        # compute the path for the new one
        self.nxt_dir = os.path.join(self.hmm_dir, str(self.n).zfill(3))
        # make the new directory
        os.mkdir(self.nxt_dir)

    def prepare_training(self, file_id_list_name, wav_dir, lab_dir, work_dir, multiple_speaker):

        print  '---preparing enverionment'
        self.cfg_dir = os.path.join(work_dir, 'config')
        self.model_dir = os.path.join(work_dir, 'model')
        self.cur_dir = os.path.join(self.model_dir, 'hmm0')
        if not os.path.exists(self.cfg_dir):
            os.makedirs(self.cfg_dir)
        if not os.path.exists(self.cur_dir):
            os.makedirs(self.cur_dir)

        self.phonemes = os.path.join(work_dir, 'mono_phone.list')
        self.phoneme_map = os.path.join(work_dir, 'phoneme_map.dict')
        # HMMs
        self.proto = os.path.join(self.cfg_dir, 'proto')
        # SCP files
        self.copy_scp = os.path.join(self.cfg_dir, 'copy.scp')
        self.test_scp = os.path.join(self.cfg_dir, 'test.scp')
        self.train_scp = os.path.join(self.cfg_dir, 'train.scp')
        # CFG
        self.cfg = os.path.join(self.cfg_dir, 'cfg')
    
        self.wav_dir = wav_dir
        self.lab_dir = lab_dir
        self.mfc_dir = os.path.join(work_dir, 'mfc')
        if not os.path.exists(self.mfc_dir):
            os.makedirs(self.mfc_dir)

        self.mono_lab_dir = os.path.join(work_dir, 'mono_no_align')
        if not os.path.exists(self.mono_lab_dir):
            os.makedirs(self.mono_lab_dir)

        file_id_list = self._read_file_list(file_id_list_name)
        print  '---checking data'
        speaker_utt_dict = self._check_data(file_id_list, multiple_speaker)
        
        print  '---extracting features'
        self._HCopy()
        print time.strftime("%c")

        print  '---feature_normalisation'
        normaliser = MeanVarianceNorm(39)
        for key_name in speaker_utt_dict.keys():
            normaliser.feature_normalisation(speaker_utt_dict[key_name], speaker_utt_dict[key_name])  ## save to itself
        print time.strftime("%c")
        
        print  '---making proto'
        self._make_proto()
        
    def train_hmm(self, niter, num_mix):
        """
        Perform one or more rounds of estimation
        """

        print time.strftime("%c")
        print  '---training HMM models'
        done = 0
        mix = 1
        while mix <= num_mix and done == 0:
            for i in xrange(niter):
                next_dir = os.path.join(self.model_dir, 'hmm_mix_' + str(mix) + '_iter_' + str(i+1))
                if not os.path.exists(next_dir):
                    os.makedirs(next_dir)
                check_call([HERest, '-C', self.cfg, '-S', self.train_scp,
                            '-I', self.phoneme_mlf,
                            '-M', next_dir,
                            '-H', os.path.join(self.cur_dir, MACROS),
                            '-H', os.path.join(self.cur_dir, HMMDEFS),
                            '-t'] + PRUNING + [self.phonemes],
                           stdout=PIPE)
                self.cur_dir = next_dir

            if mix * 2 <= num_mix:
                ##increase mixture number
                hed_file = os.path.join(self.cfg_dir, 'mix_' + str(mix * 2) + '.hed')
                fid = open(hed_file, 'w')
                print  >> fid, 'MU ' + str(mix * 2) + ' {*.state[2-'+str(STATE_NUM+2)+'].mix}\n'
                fid.close()

                next_dir = os.path.join(self.model_dir, 'hmm_mix_' + str(mix * 2) + '_iter_0')
                if not os.path.exists(next_dir):
                    os.makedirs(next_dir)
                
                check_call( [HHEd, '-A', 
                             '-H', os.path.join(self.cur_dir, MACROS),
                             '-H', os.path.join(self.cur_dir, HMMDEFS),
                             '-M', next_dir] + [hed_file] + [self.phonemes])

                self.cur_dir = next_dir
                mix = mix * 2
            else:
                done = 1

    def align(self, work_dir, lab_align_dir):
        """
        Align using the models in self.cur_dir and MLF to path
        """
        print  '---aligning data'
        print time.strftime("%c")
        self.align_mlf = os.path.join(work_dir, 'mono_align.mlf')

        check_call([HVite, '-a', '-f', '-m', '-y', 'lab', '-o', 'SM', 
                    '-i', self.align_mlf, '-L', self.mono_lab_dir,
                    '-C', self.cfg, '-S', self.train_scp,
                    '-H', os.path.join(self.cur_dir, MACROS),
                    '-H', os.path.join(self.cur_dir, HMMDEFS),
                    '-I', self.phoneme_mlf, '-t'] + PRUNING +
                   ['-s', SFAC, self.phoneme_map, self.phonemes])

        self._postprocess(self.align_mlf, lab_align_dir)

    def _postprocess(self, mlf, lab_align_dir):
        if not os.path.exists(lab_align_dir):
            os.makedirs(lab_align_dir)

        state_num = STATE_NUM
        fid = open(mlf, 'r')
        line = fid.readline()
        while True:
            line = fid.readline()
            line = line.strip()
            if len(line) < 1:
                break
            line = line.replace('"', '')
            file_base = os.path.basename(line)
            flab = open(os.path.join(self.lab_dir, file_base), 'r')
            fw   = open(os.path.join(lab_align_dir, file_base), 'w')
            for full_lab in flab.readlines():
                full_lab = full_lab.strip()
                for i in xrange(state_num):
                    line = fid.readline()
                    line = line.strip()
                    tmp_list = line.split()
                    print >> fw,'{0} {1} {2}[{3}]'.format(tmp_list[0], tmp_list[1], full_lab, i+2)

            fw.close()
            flab.close()
            line = fid.readline()
            line = line.strip()
            if line != '.':
                print 'The two files are not matched!\n'
                sys.exit(1)
        fid.close()


if __name__ == '__main__':

    work_dir = os.getcwd()  

    wav_dir = os.path.join(work_dir, 'slt_wav')
    lab_dir = os.path.join(work_dir, 'label_no_align')
    lab_align_dir = os.path.join(work_dir, 'label_state_align')

    file_id_list_name = os.path.join(work_dir, 'file_id_list.scp')
    
    ## if multiple_speaker is tuned on. the file_id_list.scp has to reflact this
    ## for example
    ## speaker_1/0001
    ## speaker_2/0001
    ## This is to do speaker-dependent normalisation
    multiple_speaker = False    

    aligner = ForcedAlignment()
    aligner.prepare_training(file_id_list_name, wav_dir, lab_dir, work_dir, multiple_speaker)

    aligner.train_hmm(7, 32)
    aligner.align(work_dir, lab_align_dir)
    print   '---done!'


    
