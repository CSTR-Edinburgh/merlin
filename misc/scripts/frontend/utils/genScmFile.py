import os,sys,glob
import collections

def readtext(fname):
    f = open(fname, 'r')
    data = f.read()
    data = data.strip(' \n')
    f.close()
    return data

def create_dictionary_from_txt_dir(txt_dir):
    utt_text = {}
    textfiles = glob.glob(in_txt_dir + '/*.txt')

    num_of_files = len(textfiles)

    for i in xrange(num_of_files):
        textfile = textfiles[i]
        junk,filename = os.path.split(textfile)
        filename = filename.split('.')[0]
        
        text = readtext(textfile)
        utt_text[filename] = text

    return utt_text

def create_dictionary_from_txt_file(txt_file):
    utt_text = {}
    in_f = open(txt_file, 'r')
    for newline in in_f.readlines():
        newline = newline.strip()
        newline = newline.replace('(', '')
        newline = newline.replace(')', '')

        text_parts = newline.split()
        filename = text_parts[0]

        text = ' '.join(text_parts[1:])
        text = text[1:-1] ## remove begining and end double quotes

        utt_text[filename] = text

    return utt_text

if __name__ == "__main__":

    if len(sys.argv)!=5:
        print 'Usage: python genScmFile.py <in_txt_dir/in_txt_file> <out_utt_dir> <out_scm_file> <out_file_id_list>'
        sys.exit(1)

    out_utt_dir  = sys.argv[2]
    out_scm_file = sys.argv[3]
    out_id_file  = sys.argv[4]

    if not os.path.exists(out_utt_dir):
        os.makedirs(out_utt_dir)
    
    if os.path.isdir(sys.argv[1]):
        print "creating a scheme file from text directory"
        in_txt_dir = sys.argv[1]
        utt_text   = create_dictionary_from_txt_dir(in_txt_dir)
    
    elif os.path.isfile(sys.argv[1]):
        print "creating a scheme file from text file"
        in_txt_file = sys.argv[1]
        utt_text    = create_dictionary_from_txt_file(in_txt_file)

    sorted_utt_text = collections.OrderedDict(sorted(utt_text.items()))

    out_f1 = open(out_scm_file, 'w')
    out_f2 = open(out_id_file, 'w')

    ### if you want to use a particular voice
    #out_f1.write("(voice_cstr_edi_fls_multisyn)\n")

    for utt_name, sentence in sorted_utt_text.iteritems():
        out_file_name = os.path.join(out_utt_dir, utt_name+'.utt')
        sentence = sentence.replace('"', '\\"')
        out_f1.write("(utt.save (utt.synth (Utterance Text \""+sentence+"\" )) \""+out_file_name+"\")\n")
        out_f2.write(utt_name+"\n")

    out_f1.close()
    out_f2.close()
