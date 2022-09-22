# Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 14.04/16.04/18.04/20.04/21.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.1, 1,3, 1,5~1.10)
* CUDA 9.0 or higher (PyTorch 1.3+ needs CUDA 9.2+)
* [`spconv v1.x`](https://github.com/traveller59/spconv/tree/v1.2.1) or [`spconv v2.x`](https://github.com/traveller59/spconv)

Notice that all results in the paper is running in with Spconv 1.2.1, but the flops calculation for spconv-based network 
(i.e. CenterPoint-Voxel or SECOND) is running with Spconv 2.x.

### Install `pcdet v0.5.2`
NOTE: Please re-install `pcdet v0.5.2` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.
```shell
git clone https://github.com/jihanyang/SparseKD.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries:

```
pip install -r requirements.txt
```

* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 
    * If you use PyTorch 1.1, then make sure you install the `spconv v1.0` with ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
    * If you use PyTorch 1.3+, then you need to install the `spconv v1.2.1`. As mentioned by the author of [`spconv`](https://github.com/traveller59/spconv/tree/v1.2.1), you need to use their docker if you use PyTorch 1.4+. 
    * You could also install latest `spconv v2.x` with pip, see the official documents of [spconv](https://github.com/traveller59/spconv).
  
c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

### Install Thop
(If you don't need to calculate **Flops**, **Parameters** and **Acts**, just skip this part)
To calculate **Flops** and **Acts**, we leverage the popular [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter).

Note: that the **Flops** in our paper actually means **Macs** in [pytorch-OpCounter](https://github.com/Lyken17/pytorch-OpCounter).
We use the term `Flops` in order to follow the same formulation as [RegNetX](https://github.com/facebookresearch/pycls), we name it as flops in 
the paper.

a. Clone our customized Thop.
```shell
git clone https://github.com/jihanyang/pytorch-OpCounter.git
```

b. Install.
```shell
cd pytorch-OpCounter && pip install .
```
