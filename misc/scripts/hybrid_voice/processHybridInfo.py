#!/usr/bin/env python

import os,numpy,re

mstohtk=10000
sectoms=1000
frameshift=5

class ContextInfo:
    def __init__(self, fileID, frameStart, frameEnd, context, btlnkFeats=None):
        self.fid       = numpy.array([int(fileID)])
        self.sframe    = numpy.array([int(frameStart)])
        self.eframe    = numpy.array([int(frameEnd)])
        self.context   = context
        self.feats     = numpy.array(btlnkFeats)
        self.featsDist = DistributionInfo(1) # one stream
	
        if btlnkFeats.any():
		    self.featsDist.setMean(numpy.mean(self.feats,0))
		    self.featsDist.setVariance(numpy.var(self.feats,0))
        
    def getContext(self):
        return self.context

    def getFeats(self):
        return self.feats

    def getFeatsDistribution(self):
	    return self.featsDist

    def getId(self):
	    return self.fid

    def getStartFrame(self):
	    return self.sframe

    def getEndFrame(self):
	    return self.eframe
    
    def setStartFrame(self, frameStart):
	    self.sframe = frameStart

    def setEndFrame(self, frameEnd):
	    self.eframe = frameEnd
    
    def sameContext(self, altContext):
        if self.context==altContext:
            print 'match found: {0}'.format(altContext)
        return self.context==altContext
    
    def addContextInstance(self, fileID, frameStart, frameEnd, btlnkFeats):
        self.fid    = numpy.hstack([self.fid,int(fileID)])
        self.sframe = numpy.hstack([self.sframe,int(frameStart)])
        self.eframe = numpy.hstack([self.eframe,int(frameEnd)])
        self.feats  = numpy.vstack([self.feats,btlnkFeats])
        
        self.featsDist.setMean(numpy.mean(self.feats,0))
        self.featsDist.setVariance(numpy.var(self.feats,0))
    
    def contextMatch(self, expr):
	    res = expr.search(self.context)
	    return res.group(1)

class DistributionInfo:
    def __init__(self, mixNum=1): 
        self.mean      = [None for x in xrange(mixNum)]
        self.var       = [None for x in xrange(mixNum)]
        self.mixWeight = [None for x in xrange(mixNum)]
    
    def setVariance(self, variance, index=0):
        self.var[index] = numpy.array(variance, dtype=numpy.float)
    
    def setMean(self, mean, index=0):
        self.mean[index] = numpy.array(mean, dtype=numpy.float)
    
    def setMixWeight(self, weight, index=0):
        self.mixWeight[index] = weight
    
    def getCovariance(self, index=0):
        covariance = numpy.zeros((len(self.var[index]), len(self.var[index])))
        for i in xrange(len(self.var[index])):
            covariance[i, i] = self.var[index][i]
        return covariance
    
    def getInverseCovariance(self, index=0):
        covariance = numpy.zeros((len(self.var[index]), len(self.var[index])))
        for i in xrange(len(self.var[index])):
            covariance[i,i] = 1.0/self.var[index][i]
        return covariance
    
    def getDimensionality(self, index=0):
        return len(self.var[index])
    
    def getMeans(self, index=0):
        meanMatrix = numpy.transpose(numpy.matrix(self.mean[index]))
        return meanMatrix

    def getArrayVariances(self, index=0):
        return self.var[index]

    def getArrayMeans(self, index=0):
        return self.mean[index]

    def getMixWeight(self, index=0):
        return self.mixWeight[index]

    def enforceVFloor(self, varFloor, index=0):
        count=0
        for x in xrange(len(self.var[index])):
            if self.var[index][x]<varFloor[x]:
                self.var[index][x] = varFloor[x]
                count = count+1
        return count


def readBottleneckFeatures(fname, featNum=32):
    data = numpy.fromfile(fname, 'float32')
    data = data.reshape(-1,featNum)
    return data

def calculateParamGV(feat_file_list, feat_dim=32):
    data = numpy.empty((1, feat_dim))
    for file_index in xrange(len(feat_file_list)):
        file_name   = feat_file_list[file_index]
        (junk, ext) = feat_file_list[file_index].split('.')
        
        features    = readBottleneckFeatures(file_name, feat_dim)

        if ext == 'lf0': #remove unvoiced values
            features = features[numpy.where(features != -1.*(10**(10)))[0]]
            features = numpy.exp(features) #convert to linear scale

        if file_index==0:
            data=features
        else:
            data=numpy.concatenate((data,features),0)

    gv = numpy.var(data, 0)
    return gv

def readHybridLabelFile(fname, idnum, sil_identifier='#', ignoreSilence=True):
    fid  = open(fname, 'r')
    data = fid.readlines()
    fid.close()

    lines = [[data[x].split()[0], data[x].split()[2]] for x in xrange(1,len(data))] #exclude first line!

    columns = [[] for x in xrange(len(lines[0]))]
    for line in lines:
        for i, item in enumerate(line):
            columns[i].append(item)

    idarr   = numpy.ones(len(columns[0]))*idnum
    stime   = numpy.hstack((0,numpy.array(columns[0][:-1],dtype=numpy.float64)))
    columns = numpy.vstack((idarr,stime,columns))

    if ignoreSilence:
        keep = [not(bool(re.search(sil_identifier,x))) for x in columns[3]]
    else:
        keep = [bool(1) for x in xrange(len(columns[3]))]

    toInc  = numpy.where(keep)[0]
    gap    = numpy.array(columns[2][toInc], dtype=numpy.float64)-numpy.array(columns[1][toInc], dtype=numpy.float64)
    frames = (gap*sectoms)/frameshift

    frameEnd   = numpy.cumsum(frames)
    frameEnd   = numpy.round(frameEnd,0)
    frameStart = numpy.append(0,frameEnd[:-1])

    allFrameStart = numpy.ones(len(columns[2]))*-1
    allFrameEnd   = numpy.ones(len(columns[2]))*-1

    for point in xrange(len(toInc)):
        allFrameEnd[toInc[point]]   = frameEnd[point]
        allFrameStart[toInc[point]] = frameStart[point]

    data = [columns[0],allFrameStart,allFrameEnd,columns[3],numpy.array(columns[1], dtype=numpy.float64)*sectoms,numpy.array(columns[2], dtype=numpy.float64)*sectoms]
    
    return data

def convertToHybridLabel(labData, numHybridSec):
    hybridData   = [[] for x in xrange(len(labData))]
    labDurations = labData[2]-labData[1]
    
    tDur = labData[5]-labData[4]
    for i in xrange(len(labData[0])):
        #keep as frames or convert to time?! Currently kept in frames
        sectionLen = float(labDurations[i])/numHybridSec
        
        tLen = float(tDur[i])/numHybridSec
        for j in xrange(numHybridSec):
            hybridData[0].append(labData[0][0])
            hybridData[1].append(int(labData[1][i]+numpy.floor((j)*sectionLen)))
            hybridData[2].append(int(labData[1][i]+numpy.floor((j+1)*sectionLen)))
            hybridData[3].append(labData[3][i]+'[{0}]'.format(j))
            hybridData[5].append(int(labData[4][i]+numpy.floor((j+1)*tLen)))
    
    hybridData[1] = numpy.array(hybridData[1])
    hybridData[2] = numpy.array(hybridData[2])
    hybridData[3] = numpy.array(hybridData[3])
    hybridData[4] = numpy.append(labData[4][0],hybridData[5][0:len(hybridData[3])-1])
    hybridData[5] = numpy.array(hybridData[5])
    
    return hybridData

