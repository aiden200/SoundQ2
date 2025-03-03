# Supplimental Installation steps for Grounded SAM2

Reminder, this is not the official guidelines to install Grounded Sam2. Please follow the [official instructions](https://github.com/IDEA-Research/Grounded-SAM-2/tree/dd4c5141b75e4838dd486c64f773c43b4db3a07b?tab=readme-ov-file#installation) first, then check in with this doc if it does not work.

Go into the SAM2 directory
```bash
cd Grounded-SAM-2
```

### Download the Pretrained checkpoints

Download the pretrained `SAM 2` checkpoints:

```bash
cd checkpoints
bash download_ckpts.sh
```

Download the pretrained `Grounding DINO` checkpoints:

```bash
cd ../gdino_checkpoints
bash download_ckpts.sh
cd ..
```


### Installing CUDA on conda environment:

Use the command 

```bash
conda install nvidia::cuda
```
to install CUDA on conda. Specific CUDA versions also could be installed. For example, use 
```bash
conda install nvidia/label/cuda-12.4.1::cuda
```

to install version 12.4.1

### Set the environment variables:
Use the following commands to set the environment variables within the conda environment. After executing these commands, you may need to reactivate the conda environment.

```bash
conda env config vars set CUDA_HOME="<miniconda OR anaconda PATH>/envs/soundq2/"
conda env config vars set LD_LIBRARY_PATH="<miniconda OR anaconda PATH>/envs/soundq2/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
conda env config vars set CPATH="<miniconda OR anaconda PATH>/envs/soundq2/targets/x86_64-linux/include/:$CPATH"
```

*Note:* To find the correct path for `CUDA_HOME` use `which nvcc`. In my case, the output of the command was:

```bash
/home/user/anaconda3/envs/soundq2/bin/nvcc
```
Therefor, I set my `CUDA_HOME` as `/home/user/anaconda3/envs/soundq2/`.

*Note:* To find the correct path for `LD_LIBRARY_PATH` use `find ~ -name cuda_runtime_api.h`. In my case, the output of the command was:
```bash
/home/user/anaconda3/envs/soundq2/targets/x86_64-linux/include/cuda_runtime_api.h
```
So I set the `LD_LIBRARY_PATH` as `/home/user/anaconda3/envs/soundq2/targets/x86_64-linux/lib/` and CPATH as `/home/user/anaconda3/envs/soundq2/targets/x86_64-linux/include/`. 

### Install the project
Install SAM2 
```bash
pip install -e .
``` 
Install required Packages:
```bash
pip install ninja 
```
Install Grounding Dino:
```bash
pip install --no-build-isolation -e grounding_dino
```

*Note:* I ran into a `DeprecatedTypeProperties`, `c10::ScalarType` exists error, I found that installing this exact pytorch and torchvision version fixed it.

- `Python==3.10`
- `torch == 2.5.1`
- `torchvision==0.20.1`
- `cuda12.4`


```
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu124
```