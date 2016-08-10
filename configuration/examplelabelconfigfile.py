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

# configuration for the input labels (features) for the DNN
# 
# this currently supports 
# * input labels can be any combination of HTS and XML style input labels
# * output features are numerical *only* (all strings are fully expanded into 1-of-n encodings, etc)
# 
# 
# 
# this is all executable python code
#  so we need to define things before using them
#  that means the description is bottom-up

import logging
logger = logging.getLogger("labels")

# we need to specify how any non-numerical (e.g., unicode string) features will be converted (mapped) into numerical feature vectors
# (just some examples for now)
maps = {

'cplace_to_binary':{
    '_UNSEEN_' : [0,0,0,0,0,0,0],
          'NA' : [0,0,0,0,0,0,0],
        '_NA_' : [0,0,0,0,0,0,0],
    'alveolar' : [1,0,0,0,0,0,0],
      'dental' : [0,1,0,0,0,0,0],
     'glottal' : [0,0,1,0,0,0,0],
      'labial' : [0,0,0,1,0,0,0],
 'labiodental' : [0,0,0,0,1,0,0],
     'palatal' : [0,0,0,0,0,1,0],
       'velar' : [0,0,0,0,0,0,1]
    },

'cmanner_to_binary':{
    '_UNSEEN_' : [0,0,0,0,0,0],
          'NA' : [0,0,0,0,0,0],
        '_NA_' : [0,0,0,0,0,0],
   'affricate' : [1,0,0,0,0,0],
 'approximant' : [0,1,0,0,0,0],
   'fricative' : [0,0,1,0,0,0],
      'liquid' : [0,0,0,1,0,0],
       'nasal' : [0,0,0,0,1,0],
        'stop' : [0,0,0,0,0,1]
    },

'cvoiced_to_binary':{
 '_UNSEEN_' : [0,0],
       'NA' : [0,0],
     '_NA_' : [0,0],
      'yes' : [1,0],
       'no' : [0,1]
    },

'vfront_to_binary':{
 '_UNSEEN_' : [0,0,0],
       'NA' : [0,0,0],
     '_NA_' : [0,0,0],
     'back' : [1,0,0],
      'mid' : [0,1,0],
    'front' : [0,0,1]
    },
    
'vheight_to_binary':{
 '_UNSEEN_' : [0,0,0],
       'NA' : [0,0,0],
     '_NA_' : [0,0,0],
     'high' : [1,0,0],
      'mid' : [0,1,0],
      'low' : [0,0,1]
    },
    
'vlength_to_binary':{
  '_UNSEEN_' : [0,0,0,0],
        'NA' : [0,0,0,0],
      '_NA_' : [0,0,0,0],
 'diphthong' : [1,0,0,0],
      'long' : [0,1,0,0],
     'schwa' : [0,0,1,0],
     'short' : [0,0,0,1]
    },
    
'vround_to_binary':{
 '_UNSEEN_' : [0,0],
       'NA' : [0,0],
     '_NA_' : [0,0],
      'yes' : [1,0],
       'no' : [0,1]
    },
    
'vowel_cons_to_binary':{
 '_UNSEEN_' : [0,0],
       'NA' : [0,0],
     '_NA_' : [0,0],
    'vowel' : [1,0],
     'cons' : [0,1]
    }
}

# read additional maps from external files and add them to the 'maps' dictionary
#  each such file must define a dictionary of dictionaries called maps, in the same format as above
#  TO DO - avoid full paths here - import them from the main config file
external_map_files=['/Users/simonk/data/dnn_tts/data/ossian/maps/segment_map.py']

import imp
for fname in external_map_files:
    # not sure this will work second time around - may not be able to import under the same module name ??
    external_maps = imp.load_source('external_maps',fname)
    for k,v in external_maps.maps.iteritems():
        if maps.has_key(k):
            logger.warning('Redefined map %s and over-wrote the previous map with the same name' % k)
        maps[k] = v

# how to extract features
# (just a few examples for now)
# 
# each feature is a dictionary with various possible entries:
#   xpath: an XPATH that will extract the required feature from a segment target node of an Ossian XML utterance tree
#   hts:   a (list of) HTS pseudo regular expression(s) that match(es) part of an HTS label, resulting in a single boolean feature
#   mapper:   an optional function or dictionary which converts the feature value (e.g., a string) to a (vector of) numerical value(s)
# 
# the dictionary describes how to compute that feature
# first, either xpath or hts describes how to extract the feature from a tree or label name
# then, an optional mapping converts the feature via a lookup table (also a dictionary) into a numerical value or vector
# 
# if no mapper is provided, then the feature must already be a single numerical or boolean value
# 
# some XPATH-based features

# in a future version, we could be more fleixble and allow more than one target_node type at once, 
# with a set of XPATHs for each target_node - it would not be very hard to modify the code to do this

# the target nodes within the XML trees that the XPATH expressions apply to
target_nodes = "//segment"
# target_nodes = "//state" ???

        # <segment pronunciation="t" cmanner="stop" cplace="alveolar" cvoiced="no" vfront="NA" vheight="NA" vlength="NA" vowel_cons="cons" vround="NA" start="1040" end="1090" has_silence="no">


# and the XPATH expressions to apply

ll_segment =      {'xpath':'preceding::segment[2]/attribute::pronunciation',   'mapper':maps['segment_to_binary'] }
l_segment  =      {'xpath':'preceding::segment[1]/attribute::pronunciation',   'mapper':maps['segment_to_binary'] }
c_segment  =      {'xpath':                    './attribute::pronunciation',   'mapper':maps['segment_to_binary'] }
r_segment  =      {'xpath':'following::segment[1]/attribute::pronunciation',   'mapper':maps['segment_to_binary'] }
rr_segment =      {'xpath':'following::segment[2]/attribute::pronunciation',   'mapper':maps['segment_to_binary'] }

cmanner    =      {'xpath':                    './attribute::cmanner',          'mapper':maps['cmanner_to_binary'] }
cplace     =      {'xpath':                    './attribute::cplace',           'mapper':maps['cplace_to_binary'] }
cvoiced    =      {'xpath':                    './attribute::cvoiced',          'mapper':maps['cvoiced_to_binary'] }

vfront     =      {'xpath':                    './attribute::vfront',           'mapper':maps['vfront_to_binary'] }
vheight    =      {'xpath':                    './attribute::vheight',          'mapper':maps['vheight_to_binary'] }
vlength    =      {'xpath':                    './attribute::vlength',          'mapper':maps['vlength_to_binary'] }
vround     =      {'xpath':                    './attribute::vround',           'mapper':maps['vround_to_binary'] }

vowel_cons =      {'xpath':                    './@vowel_cons',                'mapper':maps['vowel_cons_to_binary'] }


# a composite "vector" of XPATH features
#  this is just an ordered list of features, each of which is a dictionary describing how to compute this feature
#  each feature may be a single numerical value or a vector of numerical values
xpath_labels =[ 

ll_segment,
 l_segment,
 c_segment,
 r_segment,
rr_segment,

cmanner,
cplace,
cvoiced,

vfront,
vheight,
vlength,
vround,

vowel_cons
]


# some HTS pseudo regular expression-based features
# all of these evaluate to a single boolean value, which will be eventually represented numerically 
# note: names of features will need modifying to valid Python variable names (cannot contain "-", for example)
C_Dental_Fricative = {'hts':'{*-T+*,*-D+*}'}
C_Rounded_End      = {'hts':'{*-9^+*,*-aU+*,*-o^+*,*-Or+*,*-QO+*,*-Q+*,*-@Ur+*,*-@U+*,*-O+*,*-u+*,*-U+*}'}
C_OI               = {'hts':'{*-OI+*}'}

# a composite "vector" of HTS features
hts_labels = [C_Dental_Fricative, C_Rounded_End, C_OI]




# the full feature vector
labels = xpath_labels # + hts_labels

