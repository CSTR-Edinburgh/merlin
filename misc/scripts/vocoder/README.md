Vocoders
--------

a) STRAIGHT - extracts 60-dim MGC, 25-dim BAP, 1-dim LF0 

b) WORLD - extracts 60-dim MGC, variable-dim BAP, 1-dim LF0 <br/>
(BAP dim: 1 for 16Khz, 5 for 48Khz) 

c) MAGPHASE - extracts 60-dim mag, 45-dim real, 45-dim imag, 1-dim LF0 <br/>
(Dimensions of mag, real and imag can be fine-tuned)

d) WORLD_v2 - extracts 60-dim MGC, 5-dim BAP, 1-dim LF0 <br/>
(dimensions of MGC and BAP can be fine-tuned)

We recommend using either STRAIGHT or WORLD. 

WORLD_v2 and MAGPHASE are still under development and require more testing. 
