## Merlin: The Neural Network (NN) based Speech Synthesis System

This repository contains the Neural Network (NN) based Speech Synthesis System  
developed at the Centre for Speech Technology Research (CSTR), University of 
Edinburgh. 

Merlin is a toolkit for building Deep Neural Network models for statistical parametric speech synthesis. 
It must be used in combination with a front-end text processor (e.g., Festival) and a vocoder (e.g., STRAIGHT or WORLD).

The system is written in Python and relies on the Theano numerical computation library.

Merlin comes with recipes (in the spirit of the [Kaldi](https://github.com/kaldi-asr/kaldi) automatic speech recognition toolkit) to show you how to build state-of-the art systems.

Merlin is free software, distributed under an Apache License Version 2.0, allowing unrestricted commercial and non-commercial use alike.

Read the documentation at [cstr-edinburgh.github.io/merlin](https://cstr-edinburgh.github.io/merlin/).

Merlin is compatible with: __Python 2.7__.

Installation
------------

Merlin uses the following dependencies:

- numpy, scipy
- matplotlib
- bandmat
- theano
- sklearn, keras (optional, required if you use keras models)

To install Merlin, `cd` merlin and run the below steps:

- Install some tools in Merlin
```sh
bash tools/compile_tools.sh
```
- Install python dependencies
```sh
pip install -r requirements.txt
```

For detailed instructions, to build the toolkit: see [`INSTALL`](https://github.com/CSTR-Edinburgh/merlin/blob/master/INSTALL.md).  
These instructions are valid for UNIX
systems including various flavors of Linux;


Getting started with Merlin
---------------------------

To run the example system builds, see `egs/README.txt`

As a first demo, please follow the scripts in `egs/slt_arctic`

Now, you can also follow Josh Meyer [blog post](http://jrmeyer.github.io/merlin/2017/02/14/Installing-Merlin.html) for detailed instructions <br/> on how to install Merlin and build SLT demo voice.

For a more in-depth tutorial about building voices with Merlin, you can check out:

- [Arctic voices](https://cstr-edinburgh.github.io/merlin/slt-arctic-voice)
- [Build your own voice](https://cstr-edinburgh.github.io/merlin/build-own-voice)


Synthetic speech samples
------------------------

Listen to [synthetic speech samples](https://cstr-edinburgh.github.io/merlin/demo.html) from our SLT arctic voice.

Development pattern for contributors
------------------------------------

1. [Create a personal fork](https://help.github.com/articles/fork-a-repo/)
of the [main Merlin repository](https://github.com/CSTR-Edinburgh/merlin) in GitHub.
2. Make your changes in a named branch different from `master`, e.g. you create
a branch `my-new-feature`.
3. [Generate a pull request](https://help.github.com/articles/creating-a-pull-request/)
through the Web interface of GitHub.

Contact Us
----------

Post your questions, suggestions, and discussions to [GitHub Issues](https://github.com/CSTR-Edinburgh/merlin/issues).

Citation
--------

If you publish work based on Merlin, please cite: 

Zhizheng Wu, Oliver Watts, Simon King, "[Merlin: An Open Source Neural Network Speech Synthesis System](http://ssw9.net/papers/ssw9_PS2-13_Wu.pdf)" in Proc. 9th ISCA Speech Synthesis Workshop (SSW9), September 2016, Sunnyvale, CA, USA.

