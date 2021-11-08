# Topological Data Analysis of Vasculature

***This repository is NOT maintained. A current version is maintained by Dr Bernadette J. Stolz [here](https://github.com/stolzbernadette/TDA-Tumour-Vasculature).***

This software package was used to quantity tumour vasculature in mesoscopic photoacoustic imaging here. The package was originally written by [Dr Bernadette J. Stolz](https://www.maths.ox.ac.uk/people/bernadette.stolz) and applied by [Dr Paul W. Sweeney](www.psweeney.co.uk) to analyse 3D images of tumour vasculature obtained using raster-scanning optoacoustic mesoscopy (RSOM).

In summary, the package is split into four parts:
1. Data Preprocessing - segmented tiff stacks are initially converted to a .nii format.
2. Data Extraction - segmentations are skeletonised.
3. Data Analysis - calculates vascular descriptors.
4. Void Analysis - computes and analyses voids.

***TDA_main_script.m*** acts as a wrapper to perform the analysis detailed in in Brown, Sweeney & Lefebvre et al. (2021). Note, (4) was not used.

## References 
> [Multiscale Topology Characterises Dynamic Tumour Vascular Networks](https://arxiv.org/abs/2008.08667)<br>
> Bernadette J, Stolz et al.

## Prerequisites
The following softwares are the minimal requirements:
* Matlab 2021b.
* Python 3.6.
* NetworkX 2.4.
* Tensorflow 2.3.1
* GUDHI 3.2.0

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
This also contains the [Spyder IDE](https://www.spyder-ide.org/) to run the Python script.
