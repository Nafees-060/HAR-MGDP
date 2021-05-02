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
  1. `cd Go into your directory`
  2.  `git clone https://github.com/Nafees-060/HAR-MGDP.git`
  3. `cd HAR-MGDP`
  5.  **To run on a machine with 2 GPUs:** `horovodrun -np 2 -H localhost:2 python HAR-CNN-Horovod.py 256` `# 256 is the Batch size. By default it is set 500' 
### CUHK User 
 1. Need to connect CUHK cse VPN first ( for Outside CUHK students)
 2. Run the putty software   (For Windows user)
 3. Enter the Hostname or IP address in the field of putty. It will make you on to the command line 
 4. `login:  (Enter your Login)`
 5. `Password (Enter your password)`
 6. `export SLURM_CONF=/opt1/slurm/gpu-slurm.conf`
 7.	`srun --qos=gpu --gres=gpu:1 -n 4 -c 4 -C rtx2080,ubuntu18 --pty bash`  #(gpu: 1, 2, 3, 4, 5)
 8. `srun --qos=gpu --gres=gpu:1 -n 4 -c 4 -C rtx2080,ubuntu18 --pty bash`  #(gpu: 1, 2, 3, 4, 5)
 9. `Go into your directory` `cd /research/Abc/xyz`
 10. `git clone https://github.com/Nafees-060/HAR-MGDP.git`
 11. `cd HAR-MGDP`
 12. `/usr/local/bin/horovodrun -np 2 -H localhost:2 /usr/bin/python3 HAR-CNN-Horovod.py 256`   # 256 is the Batch size. By default it is set 500

## Experimental Workflow 
1.	Please repeat the step of Usage -> CUHK User -> 7 (with the changes of --gres=gpu:1 or 2 or 3 or 4 or 5) to see the results across all number of GPUs. Such as 
    * srun --qos=gpu --gres=gpu:1, 2,3, 4, 5 -n 4 -c 4 -C rtx2080,ubuntu18 --pty
2.	In case of all GPUs repeat the Usage -> CUHK User -> 11 (with the changes of batch size 256, 512, 1024, 2048 and 4096). Such as 
    *	/usr/local/bin/horovodrun -np 2 -H localhost:2 /usr/bin/python3 HAR-CNN-Horovod.py 256, 512, 1024, 2048 and 4096

3.	In summary, Experiments were carried out with different batch sizes (powers of 2) on all considered GPUs to see the effect of batch size.

## Note 
In case of any difficulties in using the described system, please raise an issue here in repository: https://github.com/Nafees-060/HAR-MGDP/issues





