# INSTALL

To install Merlin, `cd` merlin and run the below steps:

- Install some basic tools in Merlin
```sh
bash tools/compile_tools.sh
```
- Install python dependencies
```sh
pip install -r requirements.txt
```

## More advanced instructions

1. go to `tools/`  and follow INSTALL instructions there.
2. Merlin is coded in python and need third-party python libraries such as:

#### numpy, scipy, matplotlib, lxml 

- Usually shipped with your python packages 
- Available in Ubuntu packages

#### theano

- Can be found on pip
- Need version 0.7 and above
- http://deeplearning.net/software/theano

#### bandmat

- Can be found on pip
- https://pypi.python.org/pypi/bandmat

#### For running on NVIDIA GPU, you will need also CUDA

- https://developer.nvidia.com/cuda-zone

#### and you might want also CUDNN [optionnal]

- https://developer.nvidia.com/cudnn

### Computational efficiency
    
- Computationnal efficiency is obviously greatly improved using GPU.
- It is also improved using the latest versions of theano and numpy.

## Some Linux Instructions

#### For Ubuntu: 
```sh
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libc6-dev-i386 glibc-devel.i686 csh
```

#### For Fedora: 
```sh
sudo yum install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git libc6-dev-i386 glibc-devel.i686 csh python-lxml libxslt-devel
```

#### Common libraries for both Ubuntu and Fedora:
```sh
sudo env "PATH=$PATH" pip install Theano
sudo env "PATH=$PATH" pip install matplotlib
sudo env "PATH=$PATH" pip install bandmat
sudo env "PATH=$PATH" pip install lxml
```

#### For all stand-alone machines:
- If you are not a sudo user, this [post](https://cstr-edinburgh.github.io/install-merlin/) may help you install Merlin.

