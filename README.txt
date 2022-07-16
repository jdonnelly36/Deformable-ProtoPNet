This code package implements the deformable prototypical part network (Deformable ProtoPNet) model
described in "Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes" 
by Jon Donnelly, Alina Jade Barnett, and Chaofan Chen, published in CVPR 2022 and accessible at:
https://openaccess.thecvf.com/content/CVPR2022/html/Donnelly_Deformable_ProtoPNet_An_Interpretable_Image_Classifier_Using_Deformable_Prototypes_CVPR_2022_paper.html

A trained Deformable ProtoPNet and the auxiliary files needed to perform local and global analysis
on it can be downloaded from https://duke.box.com/v/deformable-protopnet

A video summary of this paper is available at https://youtu.be/2cgidJJtGU8

This code integrates the publicly available code from (https://github.com/cfchen-duke/ProtoPNet) 
implementing "This Looks Like That: Deep Learning for Interpretable Image Recognition" and from 
(https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0) implementing
"Deformable ConvNets v2: More Deformable, Better Results" and "Deformable Convolutional Networks."

Prerequisites: Python version 3.8.5; PyTorch (version 1.8.1), TorchVision (version 0.9.1), NumPy (version 1.20.2), cv2 (version 4.5.1)
Recommended hardware: 2  Nvidia A100 SXM4 or 2 NVIDIA Tesla V-100 GPUs

Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack CUB_200_2011.tgz
3. Split the images into training and test sets, using train_test_split.txt (included in the dataset)
5. Put the training images in the directory "./datasets/CUB_200_2011/train/"
6. Put the test images in the directory "./datasets/CUB_200_2011/test/"

Instructions for building Deformable-Convolution-V2:
1. Navigate to the Deformable-Convolution-V2-PyTorch subdirectory
2. Run make.sh

Instructions for training the model:
1. Run main.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-m is the margin to use for subtractive margin cross entropy
-last_layer_fixed is a boolean indicating whether the last layer connections will be optimized during training
-subtractive_margin is a boolean indicating whether to use subtractive margin during training or not
-using_deform is a boolean indicating whether to use subtractive margin during training or not
-topk_k is an integer indicating the number 'k' of top activations to consider; k=1 was used in our experiments
-num_prototypes is an integer indicating the number of prototypes to use (must be a multiple of the number of classes in the dataset)
-incorrect_class_connection is the value incorrect class connections are initialized to
-deformable_conv_hidden_channels is the integer number of hidden channels to use on offset prediction branch
-rand_seed is an integer setting the random seed to use for this experiment

Recommended values for all arguments on CUB_200 can be found in run.sh

Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-imgdir is the directory containing the image you want to analyze
-img is the filename of the image you want to analyze
-imgclass is the (0-based) index of the correct class of the image

Instructions for finding the nearest patches to each prototype:
1. Run global_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze

Suggested citation: 
@inproceedings{donnelly2022deformable,
  title={{Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes}},
  author={Donnelly, Jon and Barnett, Alina Jade and Chen, Chaofan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={10265--10275},
  year={2022}
}
