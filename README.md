# Bidirectional Consistency Models (PyTorch)
## Official code and model checkpoints
[![](http://img.shields.io/badge/arXiv-2403.18035-B31B1B.svg)](https://arxiv.org/abs/2403.18035)

This repo contains:
- official PyTorch **code** and model **weights** of [Bidirectional Consistency Models (BCM)](https://arxiv.org/abs/2403.18035) on ImageNet-64.
- PyTorch **code** and model **weights** of our reproduced [Improved Consistency Training (iCT)](https://arxiv.org/abs/2310.14189) on ImageNet-64.

BCM learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Our repository is based on [openai/consistency_models](https://github.com/openai/consistency_models), which was initially released under the MIT license.

We will also provide our code for BCM and iCT on CIFAR-10 with JAX. Please stay tuned for updates!

## TL;DR
BCM learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. BCM offers diverse sampling options and has great potential in downstream tasks.

## Model Weights
We provide checkpoints for BCM and our reproduced iCT on ImageNet-64:
- [BCM-ImageNet-64](https://figshare.com/ndownloader/articles/27134694/versions/1?folder_path=bcf_imagenet64_no32_qkv_4096)
- [iCT-ImageNet-64](https://figshare.com/ndownloader/articles/27134694/versions/1?folder_path=ict_imagenet64_no32_qkv_4096)
- [BCM-deep-ImageNet-64](https://figshare.com/ndownloader/articles/27134694/versions/1?folder_path=bcf_imagenet64_no32_qkv_deep_4096)
- [iCT-deep-ImageNet-64](https://figshare.com/ndownloader/articles/27134694/versions/2?folder_path=ict_imagenet64_no32_qkv_deep_4096)

Their FIDs are as follows:
|   Name   | NFE | FID |
|:--------:|:---:|:---:|
|    BCM / BCM-deep           |  1  |  4.18 / 3.14   |
|                          |  2  |  2.88 / 2.45   |
|                          |  3  |  2.78 / 2.61   |
|                          |  4  |  2.68 / 2.35   |
|   reproduced iCT / iCT-deep|  1  |  4.60 / 3.94  |
|                          |  2  |  3.40 / 3.14   |


## Dependencies

To install all packages in this codebase along with their dependencies, run
```
cd iCT
pip install -e .
```

To install with Docker, run the following commands:
```
cd docker && make build && make run
```

Please note that flash-attn==0.2.8, which cannot be substituted with the latest version and could be hard to install, is fortunately optional and not used in our best models.

We also suggest manually install mpi4py using Anaconda instead of pip, with the following command:
```
conda install -c conda-forge mpi4py=3.1.4 mpich=3.2.3
```

## Training

As we described in our paper, for complex dataset like ImageNet-64, we propose to finetune BCM from pretrained iCT model.
We, therefore, first provide code for iCT and then for BCM Finetuning.
Regarding the code to train BCM from scratch, please check out our JAX implementation on CIFAR-10.


### iCT 

The code for our reproduced iCT is located in the ```iCT``` folder.
As we described in our paper, we found the original iCT suffers from instability on ImageNet-64. 
In our experiments, it diverges after ~620k iterations and the best one-step generation FID we got is ~6.20, largely falling behind the reported 4.02 in the iCT paper.
We are open to any discussions on solutions to the instability issue and possible ways to reproduce the officially reported results.

We suspect this instability comes from the architecture of ADM. Therefore, as a remedy, we proposed *removing the attention at the resolution of 32* and applying *normalization to QKV matrices*, following EDM2. We found this helpful in improving the performance and yielding a one-step FID of 4.60.
We also apply *early stop* and save the checkpoint with the best one-step generation FID. 

Without modifications to the code, it is expected to start the training scripts with MPI for DDP training. For the commonly used SLURM, we provide the following starting script as an example:
```
srun -p YOUR_SLURM_PARTITION \
    --job-name=ict_no32_qkv \
    -n 64 --gres=gpu:8 --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --quotatype=reserved \
    --mpi=pmi2 \
    sh WORKSPACE_DIR/iCT/scripts/ict_imagenet64_no32_qkv_4096.sh
```
The above script starts an iCT experiment with our architecture modifications, using 8 computing nodes (64 GPUs in total). 

To run the original iCT, please first switch back to the original network architecture. 
If you have flash-attn==0.2.8 installed, this can be done by simply setting ```attention_type="flash"``` at https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/iCT/cm/unet.py#L282. 
If not, just keep ```attention_type="default"``` and set ```cosine=False``` at https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/iCT/cm/unet.py#L412.
Then run the following script:
```
srun -p YOUR_SLURM_PARTITION \
    --job-name=ict \
    -n 64 --gres=gpu:8 --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --quotatype=reserved \
    --mpi=pmi2 \
    sh WORKSPACE_DIR/iCT/scripts/ict_imagenet64.sh
```


### BCM Funetuning

The code for BCM is located in the ```BCM``` folder.
For ImageNet-64, we finetune BCM from pretrained iCT model to increase scalability, so please specify the location of the pretrained checkpoint in ```BCM/scripts/bcf_imagenet64_no32_qkv_4096.sh```.
We carefully initialize the model to ensure that newly added ```t_end``` will not influence the iCT prediction. Please find the details in our paper.

To perform BCF with, e.g., 64 GPUs, please run the following script:
```
srun -p YOUR_SLURM_PARTITION \
    --job-name=bcm \
    -n 64 --gres=gpu:8 --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --quotatype=reserved \
    --mpi=pmi2 \
    sh WORKSPACE_DIR/BCM/scripts/bcf_imagenet64_no32_qkv_4096.sh
```

Regarding the code to train BCM from scratch, please check out our JAX implementation on CIFAR-10.


### FP32 Training

Our implementation also support training with fp32 by setting ```fp16=False``` in the training script, **which is actually *not* supported by [the official CM implementation](https://github.com/openai/consistency_models).**
Please note that training with higher numerical accuracy doubles the computing budget and GPU memory and, according to our early experiments, may lead to different model behaviors during training.
We hope our code and observation could help future studies on the influence of numerical issues on CMs.

## Evaluations

### Sampling

Since BCM supports very flexible ways of sampling (ancestral, zigzag, mixture; see details in our paper), we think it would be overly verbose and less straightforward to pass arguments to the sampling script.
Instead, we provide just one simple script (```BCM/scripts/image_sample.py``` or ```iCT/scripts/image_sample.py``` for BCM/iCT), and allow users to modify the code for all sampling methods. 
We provide detailed examples in the script, around https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/iCT/scripts/image_sample.py#L70 for iCT and around https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/BCM/scripts/image_sample.py#L116 for BCM. 
We believe these examples are simple and straightforward enough as each of them only requires to modify numbers in a few lines.

To do distributed sampling on 4 GPUs (e.g., for iCT), please run:
```
srun -p YOUR_SLURM_PARTITION \
    --job-name=ict_sampling \
    -n 4 --gres=gpu:4 --ntasks-per-node=4 \
    --cpus-per-task=16 \
    --quotatype=reserved \
    --mpi=pmi2 \
    sh WORKSPACE_DIR/iCT/scripts/imagenet64_sample.sh
```
In the example script, it loads weights from ```CKPT_DIR/ict_imagenet64_no32_qkv_4096/ema_0.99997_680000.pt```, samples 50,000 images and saves them to ```WORKSPACE_DIR/samples/ict_imagenet64_no32_qkv_4096``` for further evaluation.

### Inversion and Reconstruction (BCM only)
Inversion and reconstruction shares the same scripts as sampling. 
By setting ```--eval_mse=True``` in the sampling script, one can perform inversion and reconstruction for the images in ```--test_data_dir```.
The per pixel MSE will be calculated automatically at the end and both the original and reconstructed images will be saved.
Again for conciseness and clarity, we refer users to https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/BCM/scripts/image_sample.py#L172 to modify the code to enable one/multi-step inversion.

Note that the ImageNet validation set is not structured by categories as the training set, so we modify the ```load_data``` function in ```cm/image_datasets.py ``` to support loading both images and labels from the valiadtion set.
For convenience, the labels could be found in ```datasets/imagenet_val_label.txt``` and specified at https://github.com/Mosasaur5526/BCM-iCT-torch/blob/main/BCM/cm/image_datasets.py#L52; one may also load the image-label pairs in their customized ways by rewriting the loading function.
Please notice the labels are important as they will be sent into the model as conditions during inversion and reconstruction.



### Calculating Metrics
We follow the standard evaluation process in the [ADM repo](https://github.com/openai/guided-diffusion/tree/main/evaluations), as also adopted in the official CM repo.


### Visualizing Samples
We also provide a simple visualization script in ```scripts/visualize_image.py```.

## Citation
If you use this repository, including our code or the weights for BCM and our reproduced iCT, please cite the following work:
```
@article{li2024bidirectional,
  title={Bidirectional Consistency Models},
  author={Li, Liangchen and He, Jiajun},
  journal={arXiv preprint arXiv:2403.18035},
  year={2024}
}
```
