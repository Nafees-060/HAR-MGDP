# HAR-MGDP:  Human Activity Recognition on multiple GPUs using Data Parallelism 
## Dependencies
### Software
- Python version: 3.6.9
- TensorFlow : 2.2.0
- Keras : 2.3.1
### Hardware 
- Multi-GPU cluset setup 
# Installation
## General User
1.	Requirement of Multi-GPU setup
2.	Python 3 or higher Required to install
3.	Tensorflow and keras
4.	Install **Horovod** with the supported packages. All installation guidance is available on [here](https://github.com/horovod/horovod). Generally, you should have setup of Horovod with NCCL, TensorFlow, Openmpi and Gloo support.
## The Chinese University of Hong Kong (CUHK) User
 CUHK students can access CSE cluster environment. 
-	Horovod with NCCL, tensorflow, openmpi and gloo support is available on **gpu40-gpu53 (Ubunt18 with RTX2080Ti)**. 
-	Request the gpu node with the feature (rtx2080,ubuntu18).
-	For CSE cluster setup and its detail you may visit [official page](https://www.cuhk.edu.hk/itsc/hpc/slurm.html).
## Data set
The dataset (PAMP2) is included here in the repository with folder name **input**
## Usage 
### General User
  1. git clone https://bitbucket.org/udayb/polymage.git -b ppopp-2018-ae
