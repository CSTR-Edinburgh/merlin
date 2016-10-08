Create training labels for merlin with HTK tools
------------------------------------------------

Step 0: Install [festival] (http://festvox.org/packed/festival/2.4/festival-2.4-release.tar.gz), 
and [HTK] (http://htk.eng.cam.ac.uk/download.shtml).

Step 1: Run setup.sh -- to download slt data and to create the config file. <br/>
./setup.sh

Step 2: Please configure paths to all tools in config.cfg

Step 3: Run aligner.sh -- which uses HVite (from HTK tools) to do forced-alignment. <br/>
./run_aligner.sh config.cfg

If all the above steps performed successfully, you should have your labels ready !! :)

HTK Aligner
-----------

The steps to perform HTK alignment are adapted from HTS source code: <br/>
[HMM-based Speech Synthesis System (HTS)] (http://hts.sp.nitech.ac.jp/)


