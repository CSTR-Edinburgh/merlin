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

from frontend.label_normalisation import HTSLabelNormalisation


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
            if 'xpath' in feature_specification:
                # xpath and hts are mutually exclusive label styles
                assert 'hts' not in feature_specification

                # if there is a mapper, then we will use that to convert the features to numbers
                # we need to look at the mapper to deduce the dimensionality of vectors that it will produce
                if 'mapper' in feature_specification:

                    # get an arbitrary item as the reference and measure its dimensionality
                    try:
                        l = len(next(iter(feature_specification['mapper'].values())))
                    except:
                        logger.critical('Empty mapper for feature %s' % feature_specification)

                    for k,v in feature_specification['mapper'].items():
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


            if 'hts' in feature_specification:
                assert 'xpath' not in feature_specification
                self.label_styles['hts'] = False # will become True once implemented
                # not yet implemented !
                self.logger.warning('HTS features not implemented - ignoring them!')

        self.label_dimension += 1 ## for frame features -- TODO: decide how to handle this properly
        #print '   add 3   cum: %s'%(  self.label_dimension)

        return self.label_dimension


if __name__ == '__main__':

    logger = logging.getLogger("labels")
    logger.setLevel(logging.DEBUG)
    # a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    label_composer = LabelComposer()
    label_composer.load_label_configuration('configuration/labelconfigfile.conf')

    print('Loaded configuration, which is:')
    print(label_composer.configuration.labels)

    d=label_composer.compute_label_dimension()
    print("label dimension will be",d)

    # not written test code for actual label processing - too complex and relies on config files
