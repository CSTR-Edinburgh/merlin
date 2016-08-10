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

import numpy

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


## osw -- also make some maps automatically, only specifying list of values for brevity:

def make_1_of_k_map(values):
    ## strip special null values:
    nulls = ['_UNSEEN_','NA','_NA_']
    values = [val for val in values if val not in nulls]
    map = {}
    for (i, value) in enumerate(values):
        vector = numpy.zeros(len(values))
        vector[i] = 1
        map[value] = vector.tolist()
    for value in nulls:
        map[value] = numpy.zeros(len(values)).tolist()
    return map
    
    
phone_names = ['@', '@@', '@U', 'A', 'D', 'E', 'E@', 'I', 'I@', 'N', 'O', 'OI', 'Q', 'S', 'T', 'U', 'U@', 'V', 'Z', 'a', 'aI', 'aU', 'b', 'd', 'dZ', 'eI', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'l!', 'lw', 'm', 'm!', 'n', 'n!', 'p', 'r', 's', 'sil', 't', 'tS', 'u', 'v', 'w', 'z']

fine_POS_inventory = ['_COMMA_', '_FULLSTOP_', '_SPACE_', 'cc', 'cd', 'dt', 'dt_VERTICALLINE_vbz', 'ex', 'ex_VERTICALLINE_vbz', 'in', 'jj', 'jjr', 'jjs', 'md', 'md_VERTICALLINE_rb', 'nn', 'nn_VERTICALLINE_pos', 'nnp', 'nnp_VERTICALLINE_pos', 'nnps', 'nns', 'pdt', 'prp', 'prp_DOLLARSIGN_', 'prp_VERTICALLINE_md', 'prp_VERTICALLINE_vbp', 'prp_VERTICALLINE_vbz', 'rb', 'rbr', 'rbs', 'rp', 'to', 'vb', 'vb_VERTICALLINE_pos', 'vb_VERTICALLINE_prp', 'vbd', 'vbd_VERTICALLINE_rb', 'vbg', 'vbn', 'vbp', 'vbp_VERTICALLINE_rb', 'vbz', 'vbz_VERTICALLINE_rb', 'wdt', 'wp', 'wp_VERTICALLINE_vbz', 'wrb']

coarse_POS_inventory = ['adj', 'adv', 'function', 'noun', 'punc', 'space', 'verb']

stress_inventory = ['stress_0', 'stress_1', 'stress_2']

maps['phone_to_binary'] = make_1_of_k_map(phone_names)
maps['fine_POS_to_binary'] = make_1_of_k_map(fine_POS_inventory)
maps['coarse_POS_to_binary'] = make_1_of_k_map(coarse_POS_inventory)
maps['stress_to_binary'] = make_1_of_k_map(stress_inventory)





# read additional maps from external files and add them to the 'maps' dictionary
#  each such file must define a dictionary of dictionaries called maps, in the same format as above
#  TO DO - avoid full paths here - import them from the main config file
external_map_files=[]

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
target_nodes = "//state"

      
# and the XPATH expressions to apply

xpath_labels = []

## NB: first feature is for silence trimming only:
xpath_labels.append({'xpath': "./ancestor::segment/attribute::pronunciation = 'sil'"})

for xpath in [

    "./ancestor::segment/preceding::segment[2]/attribute::pronunciation",
    "./ancestor::segment/preceding::segment[1]/attribute::pronunciation",
    "./ancestor::segment/attribute::pronunciation",
    "./ancestor::segment/following::segment[1]/attribute::pronunciation",
    "./ancestor::segment/following::segment[2]/attribute::pronunciation"]:
    
    xpath_labels.append({'xpath': xpath, 'mapper':maps['phone_to_binary']})


for xpath in [

    "./ancestor::segment/preceding::segment[2]/attribute::vowel_cons",
    "./ancestor::segment/preceding::segment[2]/attribute::vfront",
    "./ancestor::segment/preceding::segment[2]/attribute::vheight",
    "./ancestor::segment/preceding::segment[2]/attribute::vlength",
    "./ancestor::segment/preceding::segment[2]/attribute::vround",
    "./ancestor::segment/preceding::segment[2]/attribute::cmanner",
    "./ancestor::segment/preceding::segment[2]/attribute::cplace",
    "./ancestor::segment/preceding::segment[2]/attribute::cvoiced",

    "./ancestor::segment/preceding::segment[1]/attribute::vowel_cons",
    "./ancestor::segment/preceding::segment[1]/attribute::vfront",
    "./ancestor::segment/preceding::segment[1]/attribute::vheight",
    "./ancestor::segment/preceding::segment[1]/attribute::vlength",
    "./ancestor::segment/preceding::segment[1]/attribute::vround",
    "./ancestor::segment/preceding::segment[1]/attribute::cmanner",
    "./ancestor::segment/preceding::segment[1]/attribute::cplace",
    "./ancestor::segment/preceding::segment[1]/attribute::cvoiced",

    "./ancestor::segment/attribute::vowel_cons",
    "./ancestor::segment/attribute::vfront",
    "./ancestor::segment/attribute::vheight",
    "./ancestor::segment/attribute::vlength",
    "./ancestor::segment/attribute::vround",
    "./ancestor::segment/attribute::cmanner",
    "./ancestor::segment/attribute::cplace",
    "./ancestor::segment/attribute::cvoiced",

    "./ancestor::segment/following::segment[1]/attribute::vowel_cons",
    "./ancestor::segment/following::segment[1]/attribute::vfront",
    "./ancestor::segment/following::segment[1]/attribute::vheight",
    "./ancestor::segment/following::segment[1]/attribute::vlength",
    "./ancestor::segment/following::segment[1]/attribute::vround",
    "./ancestor::segment/following::segment[1]/attribute::cmanner",
    "./ancestor::segment/following::segment[1]/attribute::cplace",
    "./ancestor::segment/following::segment[1]/attribute::cvoiced",

    "./ancestor::segment/following::segment[2]/attribute::vowel_cons",
    "./ancestor::segment/following::segment[2]/attribute::vfront",
    "./ancestor::segment/following::segment[2]/attribute::vheight",
    "./ancestor::segment/following::segment[2]/attribute::vlength",
    "./ancestor::segment/following::segment[2]/attribute::vround",
    "./ancestor::segment/following::segment[2]/attribute::cmanner",
    "./ancestor::segment/following::segment[2]/attribute::cplace",
    "./ancestor::segment/following::segment[2]/attribute::cvoiced"]:
    
    feature = xpath.split(':')[-1]
    xpath_labels.append({'xpath': xpath, 'mapper':maps[feature + '_to_binary']})


## syll stress
for xpath in [
        "ancestor::syllable/preceding::syllable[1]/attribute::stress",
        "ancestor::syllable/attribute::stress",
        "ancestor::syllable/following::syllable[1]/attribute::stress"]:
    xpath_labels.append({'xpath': xpath, 'mapper': maps['stress_to_binary']})


## fine & coarse POS -- 3 word window
for xpath in [
        "ancestor::token/preceding::token[@token_class='word'][1]/attribute::safe_pos",
        "ancestor::token/attribute::safe_pos",
        "ancestor::token/following::token[@token_class='word'][1]/attribute::safe_pos"]:
    xpath_labels.append({'xpath': xpath, 'mapper': maps['fine_POS_to_binary']})

for xpath in [
        "ancestor::token/preceding::token[@token_class='word'][1]/attribute::coarse_pos",
        "ancestor::token/attribute::coarse_pos",
        "ancestor::token/following::token[@token_class='word'][1]/attribute::coarse_pos"]:
    xpath_labels.append({'xpath': xpath, 'mapper': maps['coarse_POS_to_binary']})
    

## === SIZES and DISTANCES till start/end -- these are numeric and not mapped:

for xpath in [

    ## state in segment -- number states is fixed, so exclude size and only count in 1 direction
    "count(./preceding-sibling::state)",

    ## segments in syll
    "count(ancestor::syllable/preceding::syllable[1]/descendant::segment)",
    "count(ancestor::syllable/descendant::segment)",
    "count(ancestor::syllable/following::syllable[1]/descendant::segment)",
    "count(./ancestor::segment/preceding-sibling::segment)",
    "count(./ancestor::segment/following-sibling::segment)",

    ## segments in word
    "count(ancestor::token/preceding::token[@token_class='word'][1]/descendant::segment)",
    "count(ancestor::token/descendant::segment)",
    "count(ancestor::token/following::token[@token_class='word'][1]/descendant::segment)",
    "count(./ancestor::syllable/preceding-sibling::syllable/descendant::segment)",
    "count(./ancestor::syllable/following-sibling::syllable/descendant::segment)",

    ## syll in word
    "count(ancestor::token/preceding::token[@token_class='word'][1]/descendant::syllable)",
    "count(ancestor::token/descendant::syllable)",
    "count(ancestor::token/following::token[@token_class='word'][1]/descendant::syllable)",
    "count(./ancestor::syllable/preceding-sibling::syllable)",
    "count(./ancestor::syllable/following-sibling::syllable)",

    ## word in phrase
    "count(ancestor::phrase/preceding::phrase[1]/descendant::token[@token_class='word'])",
    "count(ancestor::phrase/descendant::token[@token_class='word'])",
    "count(ancestor::phrase/following::phrase[1]/descendant::token[@token_class='word'])",
    "count(ancestor::token/preceding-sibling::token[@token_class='word'])",
    "count(ancestor::token/following-sibling::token[@token_class='word'])",

    ## syll in phrase
    "count(ancestor::phrase/preceding::phrase[1]/descendant::syllable)",
    "count(ancestor::phrase/descendant::syllable)",
    "count(ancestor::phrase/following::phrase[1]/descendant::syllable)",
    "count(ancestor::token/preceding-sibling::token/descendant::syllable)",
    "count(ancestor::token/following-sibling::token/descendant::syllable)",

    ## segment in phrase
    "count(ancestor::phrase/preceding::phrase[1]/descendant::segment)",
    "count(ancestor::phrase/descendant::segment)",
    "count(ancestor::phrase/following::phrase[1]/descendant::segment)",
    "count(ancestor::token/preceding-sibling::token/descendant::segment)",
    "count(ancestor::token/following-sibling::token/descendant::segment)",

    ## X in utterance
    "count(preceding::segment)",
    "count(preceding::syllable)",
    "count(preceding::token[@token_class='word'])",
    "count(preceding::phrase)",

    "count(following::segment)",
    "count(following::syllable)",
    "count(following::token[@token_class='word'])",
    "count(following::phrase)",

    "count(ancestor::utt/descendant::segment)",
    "count(ancestor::utt/descendant::syllable)",
    "count(ancestor::utt/descendant::token[@token_class='word'])",
    "count(ancestor::utt/descendant::phrase)"
    ]:
        xpath_labels.append({'xpath': xpath})








# 
# # a composite "vector" of XPATH features
# #  this is just an ordered list of features, each of which is a dictionary describing how to compute this feature
# #  each feature may be a single numerical value or a vector of numerical values
# xpath_labels =[ 
# 
# # ll_segment,
# #  l_segment,
# #  c_segment,
# #  r_segment,
# # rr_segment,
# 
# cmanner,
# cplace,
# cvoiced,
# 
# vfront,
# vheight,
# vlength,
# vround,
# 
# vowel_cons
# ]
# 

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




