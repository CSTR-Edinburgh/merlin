import os
import sys
import re

def change_label_format(inp_label_file_list, out_label_file_list, label_style="state_align"):

    utt_len = len(inp_label_file_list) 
        
    ### read file by file ###
    for i in range(utt_len):    
        inp_label_file_name = inp_label_file_list[i]
        out_label_file_name = out_label_file_list[i]

        label_info = convert_hts_lab_to_festival_lab(inp_label_file_name, out_label_file_name, label_style)
     
        print_status(i, utt_len)

    sys.stdout.write("\n")
    
def convert_hts_lab_to_festival_lab(inp_label_file_name, out_label_file_name, label_style):
    ### read label file ###
    fid = open(inp_label_file_name)
    utt_labels = fid.readlines()
    fid.close()
    
    dur = 0.0
    lab_info = [[],[]]   
    
    ### process label file ###
    for line in utt_labels:
        line = line.strip()

        if len(line) < 1:
            continue
        temp_list = re.split('\s+', line)
        full_label = temp_list[2]

        if label_style == "state_align":
            full_label_length = len(full_label) - 3  # remove state information [k]
            state_index = full_label[full_label_length + 1]

            state_index = int(state_index) - 1
            if state_index == 1:
                ph_start_time = temp_list[0]
            if state_index == 5:
                ph_end_time   = temp_list[1]
                full_label    = full_label[0:full_label_length]
                current_phone = full_label[full_label.index('-') + 1:full_label.index('+')]
                dur = dur + ((float(ph_end_time)-float(ph_start_time))*(10**-7))
                lab_info[0].append(dur)
                lab_info[1].append(current_phone)
        elif label_style == "phone_align":
            ph_start_time = temp_list[0]
            ph_end_time   = temp_list[1]
            current_phone = full_label[full_label.index('-') + 1:full_label.index('+')]
            dur = dur + ((float(ph_end_time)-float(ph_start_time))*(10**-7))
            lab_info[0].append(dur)
            lab_info[1].append(current_phone)
             
    out_f = open(out_label_file_name, 'w')
    out_f.write('#\n')
    for j in xrange(len(lab_info[0])):
        dur = lab_info[0][j]
        ph  = lab_info[1][j]
        out_f.write(str(dur)+' 125 '+ph+'\n')
    out_f.close()

    return lab_info

def print_status(i, length): 
    pr = int(float(i+1)/float(length)*100)
    st = int(float(pr)/7)
    sys.stdout.write(("\r%d/%d ")%(i+1,length)+("[ %d"%pr+"% ] <<< ")+('='*st)+(''*(100-st)))
    sys.stdout.flush()

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

def read_file_list(file_name):
    file_lists = []
    fid = open(file_name)
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        file_lists.append(line)
    fid.close()

    return  file_lists

if __name__=='__main__':
    
    if len(sys.argv)!=5:
        print 'Usage: python convert_hts_label_format_to_festival.py <input_folder> <output_folder> <file_list> <label_style: state_align/phone_align>\n'
        sys.exit(1)

    inp_lab_dir  = sys.argv[1]
    out_lab_dir  = sys.argv[2]

    file_id_scp  = sys.argv[3]
    file_id_list = read_file_list(file_id_scp)
    
    label_style  = sys.argv[4]
      
    inp_label_file_list = prepare_file_path_list(file_id_list, inp_lab_dir, '.lab')
    out_label_file_list = prepare_file_path_list(file_id_list, out_lab_dir, '.lab')
   
    print 'changing HTS label format to festival...' 
    change_label_format(inp_label_file_list, out_label_file_list, label_style)
