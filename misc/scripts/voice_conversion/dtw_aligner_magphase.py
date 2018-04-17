#!/usr/bin/env python
import os
import sys
import time
#import shutil
import multiprocessing as mp

import fastdtw

from binary_io import BinaryIOCollection
from align_feats import AlignFeats

if len(sys.argv)!=5:
    print("Usage: python dtw_aligner_magphase.py <path_to_tools_dir> <path_to_src_feat_dir> <path_to_tgt_feat_dir> <path_to_src_aligned_feat_dir>")
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

if not os.path.exists(src_aligned_feat_dir):
    os.makedirs(src_aligned_feat_dir)

### Define variables
mag_dim  = 60 # TODO: Change this (avoid hardcoded)
real_dim = 10
imag_dim = 10
lf0_dim  = 1

#src_mag_dir = src_feat_dir
#tgt_mag_dir = tgt_feat_dir

#src_lf0_dir = os.path.join(src_feat_dir, "lf0")
#tgt_lf0_dir = os.path.join(tgt_feat_dir, "lf0")

### create outut directories
#src_aligned_mag_dir = os.path.join(src_aligned_feat_dir, "mag")
#src_aligned_bap_dir = os.path.join(src_aligned_feat_dir, "bap")
#src_aligned_lf0_dir = os.path.join(src_aligned_feat_dir, "lf0")

#if not os.path.exists(src_aligned_mag_dir):
#    os.mkdir(src_aligned_mag_dir)

#if not os.path.exists(src_aligned_bap_dir):
#    os.mkdir(src_aligned_bap_dir)

#if not os.path.exists(src_aligned_lf0_dir):
#    os.mkdir(src_aligned_lf0_dir)

#################################################################
######## align source feats with target feats using dtw ## ######
#################################################################

io_funcs = BinaryIOCollection()
aligner  = AlignFeats()

def get_mag_filelist(mag_dir):
    mag_files = []
    for file in os.listdir(mag_dir):
        whole_filepath = os.path.join(mag_dir,file)
        if os.path.isfile(whole_filepath) and str(whole_filepath).endswith(".mag"):
            mag_files.append(whole_filepath)
        elif os.path.isdir(whole_filepath):
            mag_files += get_mag_filelist(whole_filepath)

    mag_files.sort()

    return mag_files

def load_dtw_path(dtw_path):
    dtw_path_dict = {}
    nframes = len(dtw_path)

    for item,i in zip(dtw_path, range(nframes)):
        if item[1] not in dtw_path_dict:
            dtw_path_dict[item[1]] = item[0]

    return dtw_path_dict

def process(filename):
    '''
    The function derives dtw alignment path given source mag and target mag
    :param filename: path to src mag file
    '''
    file_id = os.path.basename(filename).split(".")[0]
    print(file_id)

    ### DTW alignment -- align source with target parameters ###
    src_mag_file = os.path.join(src_feat_dir, file_id + ".mag")
    tgt_mag_file = os.path.join(tgt_feat_dir, file_id + ".mag")

    src_features, src_frame_number = io_funcs.load_binary_file_frame(src_mag_file, mag_dim)
    tgt_features, tgt_frame_number = io_funcs.load_binary_file_frame(tgt_mag_file, mag_dim)

    ### dtw align src with tgt ###
    distance, dtw_path = fastdtw.fastdtw(src_features, tgt_features)

    ### load dtw path
    dtw_path_dict = load_dtw_path(dtw_path)
    assert len(dtw_path_dict)==tgt_frame_number   # dtw length not matched

    ### align features
    aligner.align_src_feats(os.path.join(src_feat_dir, file_id + ".mag") , os.path.join(src_aligned_feat_dir, file_id + ".mag") , mag_dim , dtw_path_dict)
    aligner.align_src_feats(os.path.join(src_feat_dir, file_id + ".real"), os.path.join(src_aligned_feat_dir, file_id + ".real"), real_dim, dtw_path_dict)
    aligner.align_src_feats(os.path.join(src_feat_dir, file_id + ".imag"), os.path.join(src_aligned_feat_dir, file_id + ".imag"), imag_dim, dtw_path_dict)
    aligner.align_src_feats(os.path.join(src_feat_dir, file_id + ".lf0") , os.path.join(src_aligned_feat_dir, file_id + ".lf0") , lf0_dim , dtw_path_dict)

print("--- DTW alignment started ---")
start_time = time.time()

# get mag files list
mag_files = get_mag_filelist(src_feat_dir)

# do multi-processing
pool = mp.Pool(mp.cpu_count())
pool.map(process, mag_files)

(m, s) = divmod(int(time.time() - start_time), 60)
print(("--- DTW alignment completion time: %d min. %d sec ---" % (m, s)))

if not os.path.exists(src_aligned_feat_dir):
    print("DTW alignment unsucessful!!")
else:
    print("You should have your src feats(aligned with target) ready in: %s" % (src_aligned_feat_dir))    


