import os,sys
import numpy
import itertools    

from binary_io import BinaryIOCollection

io_funcs = BinaryIOCollection()

def compute_mean_and_std(lf0_file_list):
    all_files_lf0_arr = numpy.zeros(200000)
    
    current_index = 0
    for lf0_file in lf0_file_list:
        lf0_arr, frame_number = io_funcs.load_binary_file_frame(lf0_file, 1)
        for lf0_value in lf0_arr:
            all_files_lf0_arr[current_index] = numpy.exp(lf0_value)
            current_index+=1

    all_files_lf0_arr = all_files_lf0_arr[all_files_lf0_arr>0]
    all_files_lf0_arr = numpy.log(all_files_lf0_arr)
    
    mean_f0 = numpy.mean(all_files_lf0_arr)
    std_f0  = numpy.std(all_files_lf0_arr)

    return mean_f0, std_f0

def get_lf0_filelist(lf0_dir):
    lf0_files = []
    for file in os.listdir(lf0_dir):
        whole_filepath = os.path.join(lf0_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".lf0"):
            lf0_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            lf0_files += get_lf0_filelist(whole_filepath)

    lf0_files.sort()

    return lf0_files


if __name__ == "__main__":
    # parse the arguments
    lf0_dir = sys.argv[1]
    lf0_stats_file = sys.argv[2]

    lf0_file_list   = get_lf0_filelist(lf0_dir)
    mean_f0, std_f0 = compute_mean_and_std(lf0_file_list)
   
    out_f = open(lf0_stats_file, 'w')
    out_f.write('%f %f\n' %(mean_f0, std_f0))
    out_f.close()
    

