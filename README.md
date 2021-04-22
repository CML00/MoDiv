# Knowledge Distillation with Multi-Objective Divergence Learning

This code repository includes the source code for the Paper "Knowledge Distillation with Multi-Objective Divergence Learning":


## Datasets
- CIFAR10
- CIFAR100

## Networks
- Resnet-20
- Resnet-110

## Requirements and References
The code uses the following Python packages and they are required: ``tensorboardX, pytorch, click, numpy, torchvision, tqdm, scipy, Pillow``

The code is only tested in ``Python 3.7`` using ``Anaconda`` environment.

We adapt and use some code snippets from:
* [Knowledge-Distillation-Zoo](https://github.com/AberHu/Knowledge-Distillation-Zoo)
* [MultiObjectiveOptimization](https://github.com/intel-isl/MultiObjectiveOptimization)


## Usage
To train a model, use the command: 
```bash
python train_baseline.py
--data_name=cifar10/cifar100
--net_name=resnet20/resnet110
--num_class=10/100
```
```bash
python train_st.py
--t_model=/path/to/your/teacher_model 
--s_init=/path/to/your/student_initial_model 
--data_name=cifar10/cifar100  
--t_name=resnet20/resnet110 
--s_name=resnet20/resnet110 
--num_class=10/100
```


## Contact
For any question, you can contact mathshenli@gmail.com

## Citation
If you use this codebase or any part of it for a publication, please cite:
```
Knowledge Distillation with Multi-Objective Divergence Learning
Tian Ye, Chen Meiling, Shen Li, Jiang Bo, Li Zhifeng
IEEE Signal Processing Letters 
```
