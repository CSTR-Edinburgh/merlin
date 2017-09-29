Download Merlin
---------------

```bash
git clone https://github.com/CSTR-Edinburgh/merlin.git 
```

Install tools
-------------

```bash
bash merlin/tools/compile_tools.sh
```

Merlin benchmark for Blizzard 2017
--------------

To run full voice, please follow below steps:

```bash
cd merlin/egs/fls_blizzard2017/s1
./run_merlin_benchmark.sh
```

Merlin benchmark for Blizzard 2017 made use of WORLD vocoder and Unilex phoneset, training on 5866 utterances. The training of the voice approximately takes 4 to 6 hours. 

Compare the results in log files to baseline results from [RESULTS.md](https://github.com/CSTR-Edinburgh/merlin/blob/master/egs/fls_blizzard2017/s1/RESULTS.md)

Generate new sentences
----------------------

To generate new sentences, please follow below steps:

```bash
./merlin_synthesis.sh
```

