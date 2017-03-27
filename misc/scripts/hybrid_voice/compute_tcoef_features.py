#!/usr/bin/env python

import os, sys
import re
import numpy

import processHybridInfo

mstohtk        = 10000
sectoms        = 1000
frameshift     = 5
numHybridSec   = 4

def findHybridParamRichContexts(file_id_list, feat_dict, vfloor, data_dir, tcoef_dir, lab_dir, sil_identifier, ignoreSilence=True):
    ### create tcoef dir if not exists ###
    if not os.path.isdir(tcoef_dir):
        os.makedirs(tcoef_dir)

    #print "vfloor: {0}".format(vfloor)
    for file_index in xrange(len(file_id_list)):
        file_name  = file_id_list[file_index]
        label_file = os.path.join(lab_dir, file_name+'.lab')
        tcoef_file = os.path.join(tcoef_dir, file_name+'.tcoef')

        label_info = processHybridInfo.readHybridLabelFile(label_file, file_index, sil_identifier, ignoreSilence)
        hybridInfo = processHybridInfo.convertToHybridLabel(label_info, numHybridSec)

        feat_index = 0
        tempFeats  = [[] for x in range(len(feat_dict))]
        for feat_ext, feat_dim in feat_dict.iteritems():
            in_feat_dir = os.path.join(data_dir, feat_ext)
            feat_file   = os.path.join(in_feat_dir, file_name+'.'+feat_ext)
            
            tempFeats[feat_index] = processHybridInfo.readBottleneckFeatures(feat_file, feat_dim)
            if feat_ext == 'lf0':
                tempFeats[feat_index] = numpy.exp(tempFeats[feat_index])
            feat_index = feat_index + 1

        features = numpy.hstack(tempFeats)
    	 	
        outf = open(tcoef_file, 'w')
    	outf.write('EST_File Track\nDataType ascii\nNumFrames {0}\nNumChannels {1}\nNumAuxChannels 0\nfile_type 14\nEST_Header_End\n'.format(len(hybridInfo[1]),len(features[0])*2))
    	
        temp     = [[] for x in range(len(hybridInfo[1]))]
    	silMeans = numpy.zeros(len(features[0]))
    	silVars  = numpy.ones(len(features[0]))
    	
        for x in xrange(len(hybridInfo[1])):
            outf.write('{0}'.format(float(hybridInfo[4][x])/sectoms))
            if sil_identifier in hybridInfo[3][x]:
                tempMeans = silMeans
                tempVars  = silVars
            else:
                if int(hybridInfo[1][x])==int(hybridInfo[2][x]):
                    #set to the frame value if there is no range!
                	temp[x] = features[hybridInfo[1][x]:hybridInfo[2][x]+1]
                else:
                    temp[x] = features[hybridInfo[1][x]:hybridInfo[2][x]]
                
                tempContext = processHybridInfo.ContextInfo(float(hybridInfo[0][x]), hybridInfo[1][x], hybridInfo[2][x], hybridInfo[3][x], temp[x])
                tempDist    = tempContext.getFeatsDistribution()
                
                tempDist.enforceVFloor(vfloor)
                tempMeans = tempDist.getArrayMeans()
                tempVars  = tempDist.getArrayVariances()
            
            for y in tempMeans:
                outf.write('\t{0}'.format(y))
            for y in tempVars:
                outf.write('\t{0}'.format(y))
            outf.write('\n')
        
        outf.close()
        print_status(file_index, len(file_id_list))

    sys.stdout.write("\n")
	
    return tempFeats

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
        file_name = file_dir + '/' + file_id + '.' + file_extension
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
    
    #### User configurable variables #### 
    
    merlin_dir = "/work/smg/v-srikanth/merlin"
    data_dir   = os.path.join(merlin_dir, "egs/slt_arctic/s1/experiments/slt_arctic_demo/acoustic_model/data")
   
    feat_dict = {'mgc':60, 'lf0':1, 'bap':1}
    
    sil_identifier = 'sil' 
    in_lab_dir = os.path.join(data_dir, 'lab') 
    
    out_dir    = os.path.join(data_dir, 'hybrid_voice_data')
    vfloor_dir = os.path.join(out_dir,  'vfloor') 
    tcoef_dir  = os.path.join(out_dir,  'tcoef')
    
    if not os.path.isdir(vfloor_dir):
        os.makedirs(vfloor_dir)
    
    if not os.path.isdir(tcoef_dir):
        os.makedirs(tcoef_dir)
    
    tcoef_train_dir = os.path.join(tcoef_dir, 'train')
    tcoef_test_dir  = os.path.join(tcoef_dir, 'test')
    
    #### Train and test file lists #### 

    train_id_scp  = os.path.join(data_dir, 'train_id_list.scp')
    train_id_list = read_file_list(train_id_scp)
    
    test_id_scp   = os.path.join(data_dir, 'test_id_list.scp')
    test_id_list  = read_file_list(test_id_scp)
   
    #### calculate variance flooring for each feature (from only training files) ####

    feat_index = 0
    vf=[[] for x in range(len(feat_dict))]

    for feat_ext, feat_dim in feat_dict.iteritems():
        filename = feat_ext+'_'+str(feat_dim)+'_vfloor'
        var_file = os.path.join(vfloor_dir, filename) 

        if not os.path.isfile(var_file):
            print 'Calculating variance flooring for '+feat_ext+'...'
            in_feat_dir    = os.path.join(data_dir, feat_ext)
            feat_file_list = prepare_file_path_list(train_id_list, in_feat_dir, feat_ext)

            vf[feat_index] = processHybridInfo.calculateParamGV(feat_file_list, feat_dim)
            vf[feat_index] = vf[feat_index]*0.01

            numpy.savetxt(var_file, vf[feat_index])
        else:
            vf[feat_index] = numpy.loadtxt(var_file)

        feat_index = feat_index + 1

    vfloor = numpy.hstack(vf)

    #### calculate tcoef features ####
    
    print 'computing tcoef features for training data...'
    tempFeats = findHybridParamRichContexts(train_id_list, feat_dict, vfloor, data_dir, tcoef_train_dir, in_lab_dir, sil_identifier)
    
    print 'computing tcoef features for test data...'
    tempFeats = findHybridParamRichContexts(test_id_list , feat_dict, vfloor, data_dir, tcoef_test_dir,  in_lab_dir, sil_identifier)

