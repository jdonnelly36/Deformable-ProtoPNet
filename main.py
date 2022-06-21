import os
import shutil

import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import argparse
import re

from helpers import makedir
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-m', nargs=1, type=float, default=None)
parser.add_argument('-last_layer_fixed', nargs=1, type=str, default=None)
parser.add_argument('-subtractive_margin', nargs=1, type=str, default=None)
parser.add_argument('-using_deform', nargs=1, type=str, default=None)
parser.add_argument('-topk_k', nargs=1, type=int, default=None)
parser.add_argument('-deformable_conv_hidden_channels', nargs=1, type=int, default=None)
parser.add_argument('-num_prototypes', nargs=1, type=int, default=None)
parser.add_argument('-dilation', nargs=1, type=float, default=2)
parser.add_argument('-incorrect_class_connection', nargs=1, type=float, default=0)
parser.add_argument('-rand_seed', nargs=1, type=int, default=None)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
m = args.m[0]
rand_seed = args.rand_seed[0]
last_layer_fixed = args.last_layer_fixed[0] == 'True'
subtractive_margin = args.subtractive_margin[0] == 'True'
using_deform = args.using_deform[0] == 'True'
topk_k = args.topk_k[0]
deformable_conv_hidden_channels = args.deformable_conv_hidden_channels[0]
num_prototypes = args.num_prototypes[0]

dilation = args.dilation
incorrect_class_connection = args.incorrect_class_connection[0]

print("---- USING DEFORMATION: ", using_deform)
print("Margin set to: ", m)
print("last_layer_fixed set to: {}".format(last_layer_fixed))
print("subtractive_margin set to: {}".format(subtractive_margin))
print("topk_k set to: {}".format(topk_k))
print("num_prototypes set to: {}".format(num_prototypes))
print("incorrect_class_connection: {}".format(incorrect_class_connection))
print("deformable_conv_hidden_channels: {}".format(deformable_conv_hidden_channels))

np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
print("Random seed: ", rand_seed)
    
print(os.environ['CUDA_VISIBLE_DEVICES'])

from settings import img_size, experiment_run, base_architecture

if num_prototypes is None:
    num_prototypes = 1200

if 'resnet34' in base_architecture:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet152' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'resnet50' in base_architecture:
    prototype_shape = (num_prototypes, 2048, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet121' in base_architecture:
    prototype_shape = (num_prototypes, 1024, 2, 2)
    add_on_layers_type = 'upsample'
elif 'densenet161' in base_architecture:
    prototype_shape = (num_prototypes, 2208, 2, 2)
    add_on_layers_type = 'upsample'
else:
    prototype_shape = (num_prototypes, 512, 2, 2)
    add_on_layers_type = 'upsample'
print("Add on layers type: ", add_on_layers_type)


base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

from settings import train_dir, test_dir, train_push_dir

model_dir = './saved_models/' + base_architecture + '/' + train_dir + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

from settings import train_batch_size, test_batch_size, train_push_batch_size

normalize = transforms.Normalize(mean=mean,
                                 std=std)

if 'stanford_dogs' in train_dir:
    num_classes = 120
else:
    num_classes = 200
log("{} classes".format(num_classes))

if 'augmented' not in train_dir:
    print("Using online augmentation")
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
else:
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)

log('training set size: {0}'.format(len(train_loader.dataset)))
log('push set size: {0}'.format(len(train_push_loader.dataset)))
log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                            pretrained=True, img_size=img_size,
                            prototype_shape=prototype_shape,
                            num_classes=num_classes, topk_k=topk_k, m=m,
                            add_on_layers_type=add_on_layers_type,
                            using_deform=using_deform,
                            incorrect_class_connection=incorrect_class_connection,
                            deformable_conv_hidden_channels=deformable_conv_hidden_channels,
                            prototype_dilation=2)
    
ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    joint_optimizer_lrs['features'] = 1e-5
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.conv_offset.parameters(), 'lr': joint_optimizer_lrs['conv_offset']},
 {'params': ppnet.last_layer.parameters(), 'lr': joint_optimizer_lrs['joint_last_layer_lr']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.2)
log("joint_optimizer_lrs: ")
log(str(joint_optimizer_lrs))

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
log("warm_optimizer_lrs: ")
log(str(warm_optimizer_lrs))

from settings import warm_pre_offset_optimizer_lrs
if 'resnet152' in base_architecture and 'stanford_dogs' in train_dir:
    warm_pre_offset_optimizer_lrs['features'] = 1e-5
warm_pre_offset_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_pre_offset_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_pre_offset_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.features.parameters(), 'lr': warm_pre_offset_optimizer_lrs['features'], 'weight_decay': 1e-3},
]
warm_pre_offset_optimizer = torch.optim.Adam(warm_pre_offset_optimizer_specs)

warm_lr_scheduler = None
if 'stanford_dogs' in train_dir:
    warm_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)
    log("warm_pre_offset_optimizer_lrs: ")
    log(str(warm_pre_offset_optimizer_lrs))

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs
# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_warm_epochs, num_train_epochs, push_epochs, \
                    num_secondary_warm_epochs, push_start

# train the model
log('start training')

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False)
    elif epoch >= num_warm_epochs and epoch - num_warm_epochs < num_secondary_warm_epochs:
        tnt.warm_pre_offset(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        if 'stanford_dogs' in train_dir:
            warm_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_pre_offset_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=False)
    else:
        if epoch == num_warm_epochs + num_secondary_warm_epochs:
            ppnet_multi.module.initialize_offset_weights()
        tnt.joint(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, subtractive_margin=subtractive_margin,
                    use_ortho_loss=True)

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log, subtractive_margin=subtractive_margin)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                target_accu=0.70, log=log)

    if (epoch == push_start and push_start < 20) or (epoch >= push_start and epoch in push_epochs):
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.70, log=log)

        if not last_layer_fixed:
            tnt.last_only(model=ppnet_multi, log=log, last_layer_fixed=last_layer_fixed)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, 
                            subtractive_margin=subtractive_margin)
                accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.70, log=log)
logclose()

