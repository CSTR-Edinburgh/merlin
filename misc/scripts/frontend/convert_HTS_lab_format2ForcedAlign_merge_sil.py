import sys,os
import scipy.io.wavfile as sciwav
import numpy as np


def changeFormat(in_lab_file,out_lab_file):
    ip2 = open(in_lab_file,'r')
    op1 = open(out_lab_file,'w')
    data = ip2.readlines()
    ph_arr=[]
    for i in data:
        fstr = i.strip().split()
        ftag = fstr[2]
        ph = ftag[ftag.index('-')+1:ftag.index('+')]
        if(ph=='#' or ph=='pau'):
            continue;
        ph_arr.append(ph)
    count=0;prev_ph=''
    for i in data:
        fstr = i.strip().split()
        ftag = fstr[2]
        mid_indx = ftag.index(':')
        p1 = ftag[0:mid_indx]
        p2 = ftag[mid_indx:]
        ph = ftag[ftag.index('-')+1:ftag.index('+')]
        #print ph
        if(ph!='#' and ph!='pau'):
            count=count+1
        if(prev_ph=='pau' and ph=='pau'):
            continue;
        if(count<=2 and 'pau' in p1) or (count>len(ph_arr)-2 and 'pau' in p1):
            p1 = p1.replace('pau','sil')
            ftag = p1+p2
        if(count>=1 and count<len(ph_arr)):
            if '-sil+' in ftag:
                print in_lab_file
                ftag = ftag.replace('-sil+','-pau+')
        #if(prev_ph=='#' and ph=='#')
        #    continue;
        #if(count>2 and count<len(ph_arr)-1 and '#' in p1) or (count>=1 and count<len(ph_arr) and ph=='#'):
        #    p1 = p1.replace('#','pau')
        #    ftag = p1+p2
        op1.write(ftag+'\n')
        prev_ph=ph

    ip2.close()
    op1.close()

if __name__ == "__main__":

    if len(sys.argv)<3:
        print 'Usage: python convert_HTS_lab_format2ForcedAlign.py <input_lab_dir> <output_lab_dir> <file_id_list_scp>\n'
        sys.exit(0)

    in_lab_dir = sys.argv[1]
    out_lab_dir = sys.argv[2]

    os.system('mkdir -p '+out_lab_dir)

    if len(sys.argv)==3:
        os.system('ls '+sys.argv[1]+' > temp')
        os.system("sed -i \'s/\.lab//g\' temp")
        ip1 = open("temp",'r')
    else:
        ip1 = open(sys.argv[3],'r')

    for i in ip1.readlines():
        fname = i.strip()
        in_lab_file = in_lab_dir+'/'+fname+'.lab'
        out_lab_file = out_lab_dir+'/'+fname+'.lab'
        changeFormat(in_lab_file,out_lab_file)
        #print fname
        #break;

    ip1.close()
    if len(sys.argv)==3:
        os.system('rm temp')
