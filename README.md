This is a Caffe implementation of Excitation Dropout described in

> [Andrea Zunino*,Sarah Adel Bargal*, Pietro Morerio, Jianming Zhang, Stan Sclaroff, Vittorio Murino. "Excitation Dropout: Encouraging Plasticity in Deep Neural Networks". International Journal of Computer Vision (IJCV), 2021.](https://link.springer.com/article/10.1007/s11263-020-01422-y)

__This software implementation is provided for academic research and non-commercial purposes only.  This implementation is provided without warranty.__

## Prerequisites
1. The same prerequisites as Caffe
2. Excitation Backprop framework implemented in Caffe
3. Anaconda (python packages)

## Quick Start
The provided code is used to train the CNN-2 architecture with Excitation Dropout applied to the fully-connected layer ip1. 

To run the training process, please follow the following steps:
1) Download and install the Excitation Backprop framework implemented in Caffe from: https://github.com/jimmie33/Caffe-ExcitationBP
2) Download your dataset of images (e.g. Caltech256)
3) Create the train/test splits: shuffled_train.txt and shuffled_test.txt. Each of these files are expected to have the absolute paths of the train/test images
4) Create a directory: ./snapshots
4) Run the following command in your terminal: python train_with_Excitation_dropout.py

## Reference
```
@article{zunino2018excitation,
  author={Zunino, Andrea and Adel Bargal, Sarah and Morerio, Pietro and Zhang, Jianming and Sclaroff, Stan and Murino, Vittorio},
  title={Excitation Dropout: Encouraging Plasticity in Deep Neural Networks},
  journal={International Journal of Computer Vision (IJCV)},
  year={2021}
}
```
