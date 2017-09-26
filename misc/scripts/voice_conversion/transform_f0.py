import os,sys
import numpy
import argparse
    
from binary_io import BinaryIOCollection

io_funcs = BinaryIOCollection()
 
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

def prepare_file_path_list(file_id_list, file_dir, file_extension, new_dir_switch=True):
    if not os.path.exists(file_dir) and new_dir_switch:
        os.makedirs(file_dir)
    file_name_list = []
    for file_id in file_id_list:
        file_name = file_dir + '/' + file_id + file_extension
        file_name_list.append(file_name)

    return  file_name_list

def transform_f0(src_lf0_arr, stats_dict):
    mu_src = stats_dict['mu_src']
    mu_tgt = stats_dict['mu_tgt']

    std_src = stats_dict['std_src']
    std_tgt = stats_dict['std_tgt']

    tgt_lf0_arr = numpy.zeros(len(src_lf0_arr))
    for i in range(len(src_lf0_arr)):
        lf0_src = src_lf0_arr[i]
        f0_src  = numpy.exp(lf0_src)
        if f0_src <= 0:
            tgt_lf0_arr[i] = lf0_src
        else:
            tgt_lf0_arr[i] = (mu_tgt + (std_tgt/std_src)*(lf0_src - mu_src)) 
    
    return tgt_lf0_arr

def transform_lf0_dir(src_lf0_file_list, tgt_lf0_file_list, stats_dict):
    for i in range(len(src_lf0_file_list)):
        src_lf0_file = src_lf0_file_list[i]
        tgt_lf0_file = tgt_lf0_file_list[i]
        transform_lf0_file(src_lf0_file, tgt_lf0_file, stats_dict)

def transform_lf0_file(src_lf0_file, tgt_lf0_file, stats_dict):
    src_lf0_arr, frame_number = io_funcs.load_binary_file_frame(src_lf0_file, 1)
    tgt_lf0_arr = transform_f0(src_lf0_arr, stats_dict)
    io_funcs.array_to_binary_file(tgt_lf0_arr, tgt_lf0_file)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--srcstatsfile', required=True, help='path to source lf0 stats file')
    parser.add_argument('--tgtstatsfile', required=True, help='path to target lf0 stats file')
    parser.add_argument('--srcdir', type=str, help='path to source lf0 data directory')
    parser.add_argument('--tgtdir', type=str, help='path to target lf0 data directory')
    parser.add_argument('--filelist', type=str, help='path to file ID list')
    parser.add_argument('--srcfile', type=str, help='path to source lf0 data file')
    parser.add_argument('--tgtfile', type=str, help='path to target lf0 data file')
    opt = parser.parse_args()
    
    if opt.srcdir is None and opt.srcfile is None:
        print("at least one of --srcdir and --srcfile is required")
        sys.exit(1)

    if opt.tgtdir is None and opt.tgtfile is None:
        print("at least one of --tgtdir and --tgtfile is required")
        sys.exit(1)
    
    if opt.srcdir is not None and opt.filelist is None:
        print("file ID list is required")
        sys.exit(1)
    
    src_lf0_stats_file = opt.srcstatsfile
    tgt_lf0_stats_file = opt.tgtstatsfile
    
    if os.path.isfile(src_lf0_stats_file):
        in_f = open(src_lf0_stats_file, 'r')
        data = in_f.readlines()
        in_f.close()

        [src_mean_f0, src_std_f0] = map(float, data[0].strip().split())
    else:
        print("File doesn't exist!! Please check path: %s" %(src_lf0_stats_file))
        
    if os.path.isfile(tgt_lf0_stats_file):
        in_f = open(tgt_lf0_stats_file, 'r')
        data = in_f.readlines()
        in_f.close()

        [tgt_mean_f0, tgt_std_f0] = map(float, data[0].strip().split())
    else:
        print("File doesn't exist!! Please check path: %s" %(tgt_lf0_stats_file))

    #print(src_mean_f0, src_std_f0)
    #print(tgt_mean_f0, tgt_std_f0)
   
    stats_dict = {}

    stats_dict['mu_src'] = src_mean_f0
    stats_dict['mu_tgt'] = tgt_mean_f0

    stats_dict['std_src'] = src_std_f0
    stats_dict['std_tgt'] = tgt_std_f0

    if opt.srcdir is not None and opt.tgtdir is not None:
        file_id_list = read_file_list(opt.filelist)
        src_lf0_file_list = prepare_file_path_list(file_id_list, opt.srcdir, '.lf0')
        tgt_lf0_file_list = prepare_file_path_list(file_id_list, opt.tgtdir, '.lf0')
        
        transform_lf0_dir(src_lf0_file_list, tgt_lf0_file_list, stats_dict)

    elif opt.srcfile is not None and opt.tgtfile is not None:

        transform_lf0_file(opt.srcfile, opt.tgtfile, stats_dict)
        
