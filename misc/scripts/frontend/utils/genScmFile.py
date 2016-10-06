import os,sys,glob
import collections

def readtext(fname):
    f = open(fname, 'r')
    data = f.read()
    data = data.strip(' \n')
    f.close()
    return data

def create_utt_text_from_dir(txt_dir):
    utt_text = {}
    textfiles = glob.glob(in_txt_dir + '/*.txt')

    num_of_files = len(textfiles)

    for i in xrange(num_of_files):
        textfile = textfiles[i]
        junk,filename = os.path.split(textfile)
        
        text = readtext(textfile)
        utt_text[filename] = text

    return utt_text

def create_utt_text_from_scheme_file(txt_file):
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

    if len(sys.argv)!=4:
        print 'Usage: python genScmFile.py <in_txt_dir/in_txt_file> <out_scm_file> <out_utt_dir>'
        sys.exit(1)

    out_scm_file = sys.argv[2]
    out_utt_dir  = sys.argv[3]

    if not os.path.exists(out_utt_dir):
        os.makedirs(out_utt_dir)
    
    if os.path.isdir(sys.argv[1]):
        in_txt_dir = sys.argv[1]
        utt_text   = create_utt_text_from_dir(in_txt_dir)
    
    elif os.path.isfile(sys.argv[1]):
        in_txt_file = sys.argv[1]
        utt_text    = create_utt_text_from_scheme_file(in_txt_file)

    print utt_text
    
    sorted_utt_text = collections.OrderedDict(sorted(utt_text.items()))

    out_f = open(out_scm_file, 'w')

    ### if you want to use a particular voice
    #out_f.write("(voice_cstr_edi_fls_multisyn)\n")

    for utt_name, sentence in sorted_utt_text.iteritems():
        out_file_name = os.path.join(out_utt_dir, utt_name+'.utt')
        sentence = sentence.replace('"', '\\"')
        out_f.write("(utt.save (utt.synth (Utterance Text \""+sentence+"\" )) \""+out_file_name+"\")\n")

    out_f.close()
