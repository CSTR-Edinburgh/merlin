'''
Copyright 2011-2013 Pawel Swietojanski

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
MERCHANTABLITY OR NON-INFRINGEMENT.
See the Apache 2 License for the specific language governing permissions and
limitations under the License.

Not fully implemented [28 OCT 2011] 
TODO: support for options: _C, H_IREFC

'''

import io, os, sys, numpy, struct, logging

class HTK_Parm_IO(object):
    '''
    For details look at the HTK book, Chapter 5.10 Storage of Parameter Files
    '''
        
    # HTK datatybes
    H_WAVEFORM  = 0
    H_LPC       = 1
    H_LPREFC    = 2
    H_LPCEPSTRA = 3
    H_LPDELCEP  = 4
    H_IREFC     = 5
    H_MFCC      = 6
    H_FBANK     = 7
    H_MELSPEC   = 8
    H_USER      = 9
    H_DISCRETE  = 10
    H_PLP       = 11
    H_ANON      = 12
    
    # Additional 'param kind' options
    _E = 0x0001 #has energy
    _N = 0x0002 #absolute energy suppressed
    _D = 0x0004 #has delta coefficients
    _A = 0x0008 #has acceleration coefficients
    _C = 0x0010 #is compressed
    _Z = 0x0020 #has zero mean static coef.
    _K = 0x0040 #has CRC checksum
    _O = 0x0080 #has 0th cepstral coef.
    _V = 0x0100 #has VQ data
    _T = 0x0200 #has third differential coef.
    
    MASK_H_DATATYPE = 0x003f # the first 6 bits contain datatype
    
    def __init__(self, n_samples=0, samp_period=0, samp_size=0, param_kind = 0, data=None):
        '''
        '''
        
        # HTK header
        self.n_samples = n_samples # number of samples in file (4-byte integer)
        self.samp_period = samp_period # sample period in 100ns units (4-byte integer)
        self.samp_size = samp_size   # number of bytes per sample (2-byte integer)
        self.param_kind = param_kind  # a code indicating the sample kind (2-byte integer)   
        
        self.data = data
        
        return None

    def htk_datatype(self):
        return (self.param_kind & self.MASK_H_DATATYPE)
    
    
    def set_htk_datatype(self, value):
        self.param_kind = value | ~self.MASK_H_DATATYPE
    
    
    def htk_datatype_has_option(self, option):
        """Return True/False if the given options are set
        
        :type option: int
        :param option: one of the _E _N _D etc. flags
        
        """
        return (((self.param_kind>>6) & option)>0)
    
    
    def set_htk_datatype_option(self, value):
        self.param_kind = (value<<6) | self.param_kind
    
    def read_htk(self, filename, reshape_to_matrix=True):
        '''
        '''
        try:     
            
            f = open(filename, 'rb')
            
            self.n_samples = struct.unpack('<I', f.read(4))[0]
            self.samp_period = struct.unpack('<I', f.read(4))[0]    
            self.samp_size = struct.unpack('<H', f.read(2))[0]
            self.param_kind = struct.unpack('<H', f.read(2))[0]
             
            if (self.htk_datatype_has_option(self._C)):
                #TODO compression
                #self.A = struct.unpack('>H', f.read(2))[0]
                #self.B = struct.unpack('>H', f.read(2))[0]
                raise Exception("Compressed files not supported yet!")
            
            if (self.htk_datatype() == self.H_WAVEFORM):
                self.data = numpy.fromfile(f, numpy.int16)
            else:
                self.data = numpy.fromfile(f, numpy.float32)
#                print   "world"
                if reshape_to_matrix:
                    self.data = self.data.reshape( (self.n_samples, -1) )
            
#            if(sys.byteorder=='little'):
#                print   "hello"
#                self.data.byteswap(True) # forces big-endian byte ordering
            
            f.close()
        except IOError as e:
            logging.error(e)
            raise Exception(e)
    
        return None
    
    def write_htk(self, filename):
        '''
        '''
        try:
            
            file = open(filename, 'wb')
            
            file.write(struct.pack('<I', self.n_samples))
            file.write(struct.pack('<I', self.samp_period))
            file.write(struct.pack('<H', self.samp_size))
            file.write(struct.pack('<H', self.param_kind))
            
            #if(sys.byteorder=='little'):
            #    self.data.byteswap(True) # force big-endian byte ordering
            
            self.data.tofile(file)

        except IOError as e:
            raise Exception(e) 
                
        return None
    
    def print_info(self):
        
        print "Samples number: ", self.n_samples
        print "Sample period: [100ns]", self.samp_period
        print "Bytes/sample:", self.samp_size
        print "ParamKind - datatype: ", self.htk_datatype()
        print "ParamKind - options: _E(%i), _D(%i), A(%i)", self.htk_datatype_has_option(self._E), self.htk_datatype_has_option(self._D), self.htk_datatype_has_option(self._A)
        print "Features matrix shape", self.data.shape
        print "Features", self.data
        
        return None
    
    def get_data_size(self):
        return self.data.size*self.data.itemsize

def test_HTK_Parm_IO():
    
    #filename_src = "../data/GE001_1.feat"
    filename_src = "../data/tr1.mfc"
    filename_dst = "../data/tr1_dst.mfc" 
    
    htk = HTK_Parm_IO()
    
    try:   
        print 'SOURCE FILE : '
        htk.read_htk(filename_src)
        htk.print_info()
        #print "t", htk.dupa, sys.byteorder
        
        htk.writeHTK(filename_dst)
        
        print 'TARGET FILE : '
        htk2 = HTK_Parm_IO()
        htk2.read_htk(filename_dst)
        htk2.print_info()
        
    except Exception as e:
        print e
    
    return None
    
    
if __name__ == "__main__":
    test_HTK_Parm_IO()
    
