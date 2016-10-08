Create training labels for merlin with festvox tools
----------------------------------------------------

Step 0: Install [speech_tools] (http://festvox.org/packed/festival/2.4/speech_tools-2.4-release.tar.gz), [festival] (http://festvox.org/packed/festival/2.4/festival-2.4-release.tar.gz), 
and [festvox] (http://festvox.org/download.html).

Step 1: Run setup.sh -- to download slt data and to create the config file. <br/>
./setup.sh

Step 2: Please configure paths to all tools in config.cfg

Step 3: Run aligner.sh -- which uses ehmm in clustergen setup (from festvox tools) to do forced-alignment. <br/>
./run_aligner.sh config.cfg

If all the above steps performed successfully, you should have your labels ready !! :)

EHMM Aligner
------------

The steps to perform ehmm alignment are adapted from: <br/>
[http://festvox.org/bsv/c3174.html] (http://festvox.org/bsv/c3174.html)


