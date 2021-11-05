# Topological Data Analysis of Vasculature

***This package is maintained by Dr Bernadette J. Stolz [here](https://github.com/stolzbernadette/TDA-Tumour-Vasculature)***

This software package was used to quantity tumour vasculature in mesoscopic photoacoustic imaging. See here for the corresponding research article. The package was original written by [Dr Bernadette J. Stolz](https://www.maths.ox.ac.uk/people/bernadette.stolz) and applied by [Dr Paul W. Sweeney](www.psweeney.co.uk) to analyse 3D images of tumour vasculature obtained using raster-scanning optoacoustic mesoscopy (RSOM).

## References 
> [Multiscale Topology Characterises Dynamic Tumour Vascular Networks](https://arxiv.org/abs/2008.08667)<br>
> Bernadette J, Stolz et al.

## Prerequisites
The 3D CNN was trained using:
* Python 3.6.
* Keras 2.3.1.
* Tensorflow-GPU 1.14.0.

A package list for a Python environment has been provided and can be installed using the method described below.

## Installation
The package is compatible with Python3, and has been tested on Ubuntu 18.04 LTS. 
Other distributions of Linux, macOS, Windows should work as well.

To install the package from source, download zip file on GitHub page or run the following in a terminal:
```bash
git clone https://github.com/psweens/Vascular-TDA.git
```

The required Python packages can be found [here](https://github.com/psweens/Vascular-TDA/blob/main/REQUIREMENTS.txt). The package list can be installed, for example, using creating a Conda environment by running:
```bash
conda create --name <env> --file REQUIREMENTS.txt
```
