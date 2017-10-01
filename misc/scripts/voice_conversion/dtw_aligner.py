#!/usr/bin/env python
import os
import sys
import time
import shutil
import multiprocessing as mp

import fastdtw

from binary_io import BinaryIOCollection
from align_feats import AlignFeats

if len(sys.argv)!=6:
    print("Usage: python dtw_aligner.py <path_to_tools_dir> <path_to_src_feat_dir> <path_to_tgt_feat_dir> <path_to_src_aligned_feat_dir> <bap_dim>")
    sys.exit(1)

### Arguments

# tools directory
tools_dir = sys.argv[1]

# Source features directory
src_feat_dir = sys.argv[2]

# Target features directory
tgt_feat_dir = sys.argv[3]

# Source-aligned features directory
src_aligned_feat_dir = sys.argv[4]

# bap dimension
bap_dim = int(sys.argv[5])

if not os.path.exists(src_aligned_feat_dir):
    os.makedirs(src_aligned_feat_dir)

### Define variables
mgc_dim = 60
lf0_dim = 1

src_mgc_dir = os.path.join(src_feat_dir, "mgc")
tgt_mgc_dir = os.path.join(tgt_feat_dir, "mgc")

src_bap_dir = os.path.join(src_feat_dir, "bap")
tgt_bap_dir = os.path.join(tgt_feat_dir, "bap")

src_lf0_dir = os.path.join(src_feat_dir, "lf0")
tgt_lf0_dir = os.path.join(tgt_feat_dir, "lf0")

### create outut directories
src_aligned_mgc_dir = os.path.join(src_aligned_feat_dir, "mgc")
src_aligned_bap_dir = os.path.join(src_aligned_feat_dir, "bap")
src_aligned_lf0_dir = os.path.join(src_aligned_feat_dir, "lf0")

if not os.path.exists(src_aligned_mgc_dir):
    os.mkdir(src_aligned_mgc_dir)

if not os.path.exists(src_aligned_bap_dir):
    os.mkdir(src_aligned_bap_dir)

if not os.path.exists(src_aligned_lf0_dir):
    os.mkdir(src_aligned_lf0_dir)

#################################################################
######## align source feats with target feats using dtw ## ######
#################################################################

io_funcs = BinaryIOCollection()
aligner  = AlignFeats()

def get_mgc_filelist(mgc_dir):
    mgc_files = []
    for file in os.listdir(mgc_dir):
        whole_filepath = os.path.join(mgc_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".mgc"):
            mgc_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            mgc_files += get_mgc_filelist(whole_filepath)

    mgc_files.sort()

    return mgc_files

def load_dtw_path(dtw_path):
    dtw_path_dict = {}
    nframes = len(dtw_path)

    for item,i in zip(dtw_path, range(nframes)):
        if item[1] not in dtw_path_dict:
            dtw_path_dict[item[1]] = item[0]

    return dtw_path_dict

def process(filename):
    '''
    The function derives dtw alignment path given source mgc and target mgc
    :param filename: path to src mgc file
    '''
    file_id = os.path.basename(filename).split(".")[0]
    print(file_id)

    ### DTW alignment -- align source with target parameters ###
    src_mgc_file = os.path.join(src_mgc_dir, file_id+ ".mgc")
    tgt_mgc_file = os.path.join(tgt_mgc_dir, file_id+ ".mgc")

    src_features, src_frame_number = io_funcs.load_binary_file_frame(src_mgc_file, mgc_dim)
    tgt_features, tgt_frame_number = io_funcs.load_binary_file_frame(tgt_mgc_file, mgc_dim)

    ### dtw align src with tgt ###
    distance, dtw_path = fastdtw.fastdtw(src_features, tgt_features)

    ### load dtw path
    dtw_path_dict = load_dtw_path(dtw_path)
    assert len(dtw_path_dict)==tgt_frame_number   # dtw length not matched

    ### align features
    aligner.align_src_feats(os.path.join(src_mgc_dir, file_id+ ".mgc"), os.path.join(src_aligned_mgc_dir, file_id+ ".mgc"), mgc_dim, dtw_path_dict)
    aligner.align_src_feats(os.path.join(src_bap_dir, file_id+ ".bap"), os.path.join(src_aligned_bap_dir, file_id+ ".bap"), bap_dim, dtw_path_dict)
    aligner.align_src_feats(os.path.join(src_lf0_dir, file_id+ ".lf0"), os.path.join(src_aligned_lf0_dir, file_id+ ".lf0"), lf0_dim, dtw_path_dict)

print("--- DTW alignment started ---")
start_time = time.time()

# get mgc files list
mgc_files = get_mgc_filelist(src_mgc_dir)

# do multi-processing
pool = mp.Pool(mp.cpu_count())
pool.map(process, mgc_files)

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- DTW alignment completion time: %d min. %d sec ---" % (m, s)))

if not os.path.exists(src_aligned_mgc_dir):
    print("DTW alignment unsucessful!!")
else:
    print("You should have your src feats(aligned with target) ready in: %s" % (src_aligned_feat_dir))    

