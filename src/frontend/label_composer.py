################################################################################
#           The Neural Network (NN) based Speech Synthesis System
#                https://svn.ecdf.ed.ac.uk/repo/inf/dnn_tts/
#                
#                Centre for Speech Technology Research                 
#                     University of Edinburgh, UK                       
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.                           
#                                                                       
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute  
#  this software and its documentation without restriction, including   
#  without limitation the rights to use, copy, modify, merge, publish,  
#  distribute, sublicense, and/or sell copies of this work, and to      
#  permit persons to whom this work is furnished to do so, subject to   
#  the following conditions:
#  
#   - Redistributions of source code must retain the above copyright  
#     notice, this list of conditions and the following disclaimer.   
#   - Redistributions in binary form must reproduce the above         
#     copyright notice, this list of conditions and the following     
#     disclaimer in the documentation and/or other materials provided 
#     with the distribution.                                          
#   - The authors' names may not be used to endorse or promote products derived 
#     from this software without specific prior written permission.   
#                                  
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK        
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING      
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT   
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE     
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN   
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,          
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF       
#  THIS SOFTWARE.
################################################################################


import logging
import imp
import numpy
from io_funcs.binary_io import BinaryIOCollection

from lxml import etree
    
from frontend.label_normalisation import HTSLabelNormalisation, XMLLabelNormalisation


# context-dependent printing format for Numpy - should move this out to a utility file somewhere
import contextlib
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield 
    numpy.set_printoptions(**original)


class LabelComposer(object):

    # a class that can compose input labels according to the user's specification, and convert them to numerical vectors
    
    def __init__(self):

        self.logger = logging.getLogger("labels")
        self.configuration = None
        self.label_dimension=None

        # what label styles we find in the feature specification
        # e.g., 'xpath' , 'hts'
        self.label_styles={}
        
        self.use_precompiled_xpaths = False ## will be set True if xpaths are compiled
                
    def load_label_configuration(self, filename):

        # load in a label specification, provided by the user
        try:
            self.configuration = imp.load_source('label_config',filename)
        except IOError:
            self.logger.critical('failed to open label configuration file %s' % filename)
            raise
        except:
            self.logger.critical('error loading label configuration from %s' % filename)
            raise
            
        # perform some sanity checks on it
        #
        # make sure 'labels' is defined
        try:
            assert self.configuration.labels
        except AssertionError:
            logger.critical('loaded label configuration file %s, but it did not define "labels" !' % filename)
            
            
    def compute_label_dimension(self):

        self.label_dimension=0

        try:
            assert self.configuration
        except AssertionError:
            self.logger.critical('no label configuration loaded, so cannot compute dimension')
            raise
            
        for feature_specification in self.configuration.labels:
            #osw# self.logger.debug('looking at feature %s' % feature_specification )
            # feature is a dictionary specifying how to construct this part of the input feature vector
            if feature_specification.has_key('xpath'):
                # xpath and hts are mutually exclusive label styles
                assert not feature_specification.has_key('hts')

                # if there is a mapper, then we will use that to convert the features to numbers
                # we need to look at the mapper to deduce the dimensionality of vectors that it will produce
                if feature_specification.has_key('mapper'):

                    # get an arbitrary item as the reference and measure its dimensionality
                    try:
                        l = len(feature_specification['mapper'].itervalues().next())
                    except:
                        logger.critical('Empty mapper for feature %s' % feature_specification)
                        
                    for k,v in feature_specification['mapper'].iteritems():
                        # make sure all other entries have the same dimension
                        try:
                            assert len(v) == l
                        except AssertionError:
                            logger.critical('Inconsistent dimensionality in mapper for feature %s' % feature_specification)
                    self.label_dimension = self.label_dimension + l
                    #print '   add %s    cum: %s'%( str(l), self.label_dimension)
                    
                            
                else:
                    # without a mapper, features will be single numerical values
                    self.label_dimension = self.label_dimension + 1
                    #print '   add 1    cum: %s'%( self.label_dimension)

                # we have seen at least one feature that will required xpath label files to be loaded
                self.label_styles['xpath'] = True
                
                    
            if feature_specification.has_key('hts'):
                assert not feature_specification.has_key('xpath')
                self.label_styles['hts'] = False # will become True once implemented
                # not yet implemented !
                self.logger.warning('HTS features not implemented - ignoring them!')
                
        self.label_dimension += 1 ## for frame features -- TODO: decide how to handle this properly
        #print '   add 3   cum: %s'%(  self.label_dimension)

        return self.label_dimension



    def precompile_xpaths(self):
        '''
        Add compiled versions of xpaths to items of self.configuration.labels.
        This avoids compilation each time each xpath is applied at each node, and saves
        a lot of time.
        '''
        try:
            assert self.configuration
        except AssertionError:
            self.logger.critical('no label configuration loaded, so cannot precompile xpaths')
            raise

        new_labels = []
        
        for feature_specification in self.configuration.labels:
            
            if feature_specification.has_key('xpath'):
                #osw# self.logger.debug('precompiling xpath %s' % feature_specification['xpath'] )
                compiled_xpath = etree.XPath(feature_specification['xpath'])
                ## overwrite the original string:
                feature_specification['xpath'] = compiled_xpath
                ## Note that it can be retrieved via path attribute: <COMPILEDXPATH>.path
            new_labels.append(feature_specification)

            
        ## set flag to use these instead of string xpaths when labels are made: 
        self.use_precompiled_xpaths = True

        self.configuration.labels = new_labels
        

    
    def make_labels(self,input_file_descriptors,out_file_name=None,\
                                    fill_missing_values=False,iterate_over_frames=False):

        ## input_file_descriptors is e.g. {'xpath': <open XML file for reading>}

        # file_descriptors is a dictionary of open label files all for the same utterance
        # currently supports XPATH or HTS file formats only
        # keys should be 'xpath' or 'hts'
        
        # an array in which to assemble all the features
        all_labels = None
        
        try:
            assert self.configuration
        except AssertionError:
            self.logger.critical('no label configuration loaded, so cannot make labels')
            raise
            
            
        # now iterate through the features, and create the features from the appropriate open label file
        
        xpath_list = []  ## gather all here and extact all features in one pass
        mapper_list = []
        
        for (item_number, feature_specification) in enumerate(self.configuration.labels):
        
            #osw# self.logger.debug('constructing feature %.80s ...' % feature_specification )
                        
            ## osw -- we'll append frame features to the data for the *LAST* 
            ##        feature_specification in our list 
            add_frame_features = False
            if item_number+1 == len(self.configuration.labels):
                add_frame_features = True
                #osw# self.logger.debug('append frame features')
                        
            # which label file should we use?
            if feature_specification.has_key('xpath'):
                # xpath and hts are mutually exclusive label styles
                assert not feature_specification.has_key('hts')
                #osw# self.logger.debug(' feature style: xpath ; XPATH: %s' % feature_specification['xpath']  )
            
                # actually make the features from this open file and the current XPATH

                try:
                    assert self.configuration.target_nodes
                except:
                    self.logger.critical('When using XPATH features, "target_nodes" must be defined in the label config file')
                    raise

                try:
                    xpath_list.append(feature_specification['xpath'])
                    if feature_specification.has_key('mapper'):
                        mapper_list.append(feature_specification['mapper'])
                    else:
                        mapper_list.append(None)
                except:
                    self.logger.critical('error creating XMLLabelNormalisation object for feature %s' % feature_specification )
                    raise
                    
                    
            if feature_specification.has_key('hts'):
                assert not feature_specification.has_key('xpath')
                # not yet implemented !
                self.logger.warning('HTS features not implemented - ignoring them!')
                #these_labels=None
                # to do, with implementation: deal with fill_missing_values correctly                     
                          
                          
        ## Now extract all feats in one go -- go straight to all_labels -- don't compose from 'these_labels':
        label_normaliser = XMLLabelNormalisation(xpath=xpath_list,mapper=mapper_list,fill_missing_values=fill_missing_values,target_nodes=self.configuration.target_nodes,use_compiled_xpath=self.use_precompiled_xpaths,iterate_over_frames=iterate_over_frames)
                            
        try:
            all_labels = label_normaliser.extract_linguistic_features(input_file_descriptors['xpath'], add_frame_features=add_frame_features)
        except KeyError:
            self.logger.critical('no open xpath label file available to create feature %s' % feature_specification )
            raise
        
            
        
#             # add these_features as additional columns of all_features
#             if (these_labels != None):
#                 if all_labels != None:
#                     all_labels = numpy.hstack((all_labels,these_labels))
#                 else:
#                     all_labels= these_labels

        if all_labels != None:
            self.logger.debug(' composed features now have dimension %d' % all_labels.shape[1])
            
        #osw# self.logger.debug( 'first line of labels: ' + str(all_labels[0,:]))
                
        
        # finally, save the labels
        if out_file_name:
            io_funcs = BinaryIOCollection()
            io_funcs.array_to_binary_file(all_labels, out_file_name)
            
            ## osw: useful for debugging:
            ##numpy.savetxt(out_file_name + '.TXT', all_labels, delimiter='\t')
            
            
            # debug
            # with printoptions(threshold=3000, linewidth=1000, edgeitems=1000, precision=1, suppress=True):
            #     # print all_labels
            #     print all_labels.sum(axis=1)
            
            
            self.logger.info('saved numerical features of shape %s to %s' % (all_labels.shape,out_file_name) )
        else:
            return all_features



if __name__ == '__main__':

    logger = logging.getLogger("labels")
    logger.setLevel(logging.DEBUG)
    # a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    label_composer = LabelComposer()
    label_composer.load_label_configuration('configuration/labelconfigfile.conf')
    
    print 'Loaded configuration, which is:'
    print label_composer.configuration.labels

    d=label_composer.compute_label_dimension()
    print "label dimension will be",d
    
    # not written test code for actual label processing - too complex and relies on config files
    
