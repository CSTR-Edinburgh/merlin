import sys,os

def normalize_for_merlin(in_lab_file,out_lab_file):
    ip2 = open(in_lab_file,'r')
    op1 = open(out_lab_file,'w')
    data = ip2.readlines()
    ph_arr=[]
    for i in data:
        fstr = i.strip().split()
        ftag = fstr[2]
        ph = ftag[ftag.index('-')+1:ftag.index('+')]
        if(ph=='pau'):
            continue;
        ph_arr.append(ph)
    count=0;prev_ph=''
    merged_data = [[],[],[]]
    for i in data:
        fstr = i.strip().split()
        start_time = fstr[0]
        end_time   = fstr[1]
        ftag = fstr[2]
        mid_indx = ftag.index(':')
        p1 = ftag[0:mid_indx]
        p2 = ftag[mid_indx:]
        ph = ftag[ftag.index('-')+1:ftag.index('+')]
        #print ph
        if(ph!='pau'):
            count=count+1
        if(prev_ph=='pau' and ph=='pau'):
            continue;
        if(count<=2 and 'pau' in p1) or (count>len(ph_arr)-2 and 'pau' in p1):
            p1 = p1.replace('pau','sil')
            ftag = p1+p2
        if(count>=1 and count<len(ph_arr)):
            if '-sil+' in ftag:
                ftag = ftag.replace('-sil+','-pau+')
        merged_data[0].append(start_time)
        merged_data[1].append(end_time)
        merged_data[2].append(ftag)
        prev_ph=ph

    tot_num_ph = len(merged_data[0])
    for j in xrange(tot_num_ph):
        if j<tot_num_ph-1:
            ph_end=int(merged_data[0][j+1])
            rem_t = ph_end%50000
            if rem_t<=25000:
                ph_end = ph_end - rem_t
            else:
                ph_end = ph_end + (50000-rem_t)
            merged_data[0][j+1]=str(ph_end)    
            merged_data[1][j]=merged_data[0][j+1]
        else:
            end_time = int(end_time)
            rem_t = end_time%50000
            if rem_t<=25000:
                end_time = end_time - rem_t
            else:
                end_time = end_time + (50000-rem_t)
            merged_data[1][j]=str(end_time)

        if (int(merged_data[1][j])-int(merged_data[0][j]))==0:
            print 'zero duration for this phone'
            raise 
        op1.write(merged_data[0][j]+' '+merged_data[1][j]+' '+merged_data[2][j]+'\n')

    ip2.close()
    op1.close()

if __name__ == "__main__":

    if len(sys.argv)<3:
        print 'Usage: python normalize_clustergen_lab_for_merlin.py <input_lab_dir> <output_lab_dir> <optional:file_id_list_scp>\n'
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
        normalize_for_merlin(in_lab_file,out_lab_file)
        print fname
        #break;

    ip1.close()
    if len(sys.argv)==3:
        os.system('rm temp')
