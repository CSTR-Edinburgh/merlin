import sys,os
import numpy as np

def divide_into_states(st_dur, fn_dur, num_states):
    state_dur = np.zeros((2, num_states), np.int64)

    state_dur[0][0] = st_dur
    state_dur[1][num_states-1] = fn_dur

    num_of_frames  = (fn_dur-st_dur)/50000
    nof_each_state = num_of_frames/num_states

    #if nof_each_state<1:
    #    print 'warning: some states are with zero duration'

    for k in xrange(num_states-1):
        state_dur[1][k]   = state_dur[0][k]+(nof_each_state*50000)
        state_dur[0][k+1] = state_dur[1][k]

    return state_dur

def normalize_dur(dur):
    rem_t = dur%50000
    
    if rem_t<=25000:
        dur = dur - rem_t
    else:
        dur = dur + (50000-rem_t)

    return dur

def normalize_label_files(in_lab_file, out_lab_file, label_style, write_time_stamps):
    out_f = open(out_lab_file,'w')
    
    in_f = open(in_lab_file,'r')
    data = in_f.readlines()
    in_f.close()

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

    num_states = 5
    tot_num_ph = len(merged_data[0])
    for j in xrange(tot_num_ph):
        if j<tot_num_ph-1:
            ph_end = normalize_dur(int(merged_data[0][j+1]))
            merged_data[0][j+1] = str(ph_end)    
            merged_data[1][j]   = merged_data[0][j+1]
        else:
            end_time = normalize_dur(int(end_time))
            merged_data[1][j]=str(end_time)

        if (int(merged_data[1][j])-int(merged_data[0][j]))==0:
            print 'Error: zero duration for this phone'
            raise

        if label_style == "phone_align":
            if write_time_stamps:
                out_f.write(merged_data[0][j]+' '+merged_data[1][j]+' '+merged_data[2][j]+'\n')
            else:
                out_f.write(merged_data[2][j]+'\n')
        elif label_style == "state_align":
            for k in xrange(num_states):
                state_dur = divide_into_states(int(merged_data[0][j]), int(merged_data[1][j]), num_states) 
                out_f.write(str(state_dur[0][k])+' '+str(state_dur[1][k])+' '+merged_data[2][j]+'['+str(k+2)+']\n')

    out_f.close()

if __name__ == "__main__":

    if len(sys.argv)<5:
        print 'Usage: python normalize_lab_for_merlin.py <input_lab_dir> <output_lab_dir> <label_style> <file_id_list_scp> <optional: write_time_stamps (1/0)>\n'
        sys.exit(0)

    in_lab_dir   = sys.argv[1]
    out_lab_dir  = sys.argv[2]
    label_style  = sys.argv[3]
    file_id_list = sys.argv[4]

    write_time_stamps = True
    if len(sys.argv)==6:
        if int(sys.argv[5])==0:
            write_time_stamps = False 

    if label_style!="phone_align" and label_style!="state_align":
        print "These labels %s are not supported as of now...please use state_align or phone_align!!" % (label_style)
        sys.exit(0)

    if not os.path.exists(out_lab_dir):
        os.makedirs(out_lab_dir)

    in_f = open(file_id_list,'r')

    for i in in_f.readlines():
        filename = i.strip()+'.lab'
        print filename
        in_lab_file  = os.path.join(in_lab_dir, filename)
        out_lab_file = os.path.join(out_lab_dir, filename)
        normalize_label_files(in_lab_file, out_lab_file, label_style, write_time_stamps)
        #break;

    in_f.close()
