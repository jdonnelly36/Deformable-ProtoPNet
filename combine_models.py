import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import re

import os

import time
from preprocess import mean, std
from settings import joint_optimizer_lrs, joint_lr_step_size

class PPNet_ensemble(nn.Module):
    
    def __init__(self, ppnets):
        
        super(PPNet_ensemble, self).__init__()
        self.ppnets = ppnets # a list of ppnets
    
    def forward(self, x):
        logits, min_distances_0 = self.ppnets[0](x)
        min_distances = [min_distances_0]
        for i in range(1, len(self.ppnets)):
            logits_i, min_distances_i = self.ppnets[i](x)
            logits.add_(logits_i)
            min_distances.append(min_distances_i)
        return logits, min_distances


# only supports last layer adjustment
def _train_or_test_ppnet_ensemble(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                                  coefs=None, log=print):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    
    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, _ = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)
            l1 = torch.tensor(0.0).cuda()

            if class_specific:
                if use_l1_mask:
                    for ppnet in model.module.ppnets:
                        l1_mask = 1 - torch.t(ppnet.prototype_class_identity).cuda()
                        l1_ = (ppnet.last_layer.weight * l1_mask).norm(p=1)
                        l1.add_(l1_)
                else:
                    for ppnet in model.module.ppnets:
                        l1_ = ppnet.last_layer.weight.norm(p=1)
                        l1.add_(l1_)

            else:
                for ppnet in model.module.ppnets:
                    l1_ = ppnet.last_layer.weight.norm(p=1)
                    l1.add_(l1_)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            
        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input, target, output, predicted

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    last_layer_p1_norm = 0
    for ppnet in model.module.ppnets:
        last_layer_p1_norm += ppnet.last_layer.weight.norm(p=1).item()
    log('\tl1: \t\t{0}'.format(last_layer_p1_norm))

    return n_correct / n_examples

def train_ensemble(model, dataloader, optimizer, class_specific=True, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=optimizer,
                                         class_specific=class_specific, coefs=coefs, log=log)


def test_ensemble(model, dataloader, class_specific=True, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test_ppnet_ensemble(model=model, dataloader=dataloader, optimizer=None,
                                         class_specific=class_specific, log=log)


def ensemble_last_only(model, log=print):
    for ppnet in model.module.ppnets:
        for p in ppnet.features.parameters():
            p.requires_grad = False
        for p in ppnet.add_on_layers.parameters():
            p.requires_grad = False
        ppnet.prototype_vectors.requires_grad = False
        for p in ppnet.last_layer.parameters():
            p.requires_grad = True
    log('\tensemble last layer')            

##### MODEL AND DATA LOADING
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# load the models
# provide paths to saved models you want to combine:
# e.g. load_model_paths = ['./saved_models/densenet121/003/30_18push0.8043.pth',
#                          './saved_models/resnet34/002/30_19push0.7920.pth',
#                          './saved_models/vgg19/003/30_18push0.7822.pth']
load_model_paths = ['./saved_models/densenet121/003/30_18push0.7901.pth',
                    './saved_models/resnet50/002/30_19push0.8640.pth',
                    './saved_models/vgg19/003/30_18push0.7600.pth']
                  
load_model_paths = list(load_model_paths)
ppnets = []
epoch_number_strs = []
start_epoch_numbers = []

for load_model_path in load_model_paths:
    load_model_name = load_model_path.split('/')[-1]
    epoch_number_str = re.search(r'\d+', load_model_name).group(0)
    epoch_number_strs.append(epoch_number_str)
    
    start_epoch_number = int(epoch_number_str)
    start_epoch_numbers.append(start_epoch_number)

    print('load model from ' + load_model_path)
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnets.append(ppnet)

ppnet_ensemble = PPNet_ensemble(ppnets)
ppnet_ensemble = ppnet_ensemble.cuda()
ppnet_ensemble_multi = torch.nn.DataParallel(ppnet_ensemble)

img_size = ppnets[0].img_size

# load the (test) data
from settings import test_dir
test_batch_size = 100

normalize = transforms.Normalize(mean=mean,
                                std=std)
train_dir = './datasets/cub200_cropped/train_cropped/'
train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomAffine(degrees=(-25, 25), shear=15),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=80, shuffle=True,
    num_workers=4, pin_memory=False)

test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=True,
    num_workers=4, pin_memory=False)
print('test set size: {0}'.format(len(test_loader.dataset)))

for ppnet in ppnet_ensemble_multi.module.ppnets:
    print(ppnet)


class_specific = True

optimizer_specs = []
for ppnet in ppnet_ensemble_multi.module.ppnets:
    optimizer_specs = optimizer_specs + [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': 0},
    {'params': ppnet.conv_offset.parameters(), 'lr': joint_optimizer_lrs['conv_offset']},
    {'params': ppnet.last_layer.parameters(), 'lr': 1e-4}
    ]
optimizer = torch.optim.Adam(optimizer_specs)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=joint_lr_step_size, gamma=0.1)

#check test accuracy
accu = test_ensemble(model=ppnet_ensemble_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)
for i in range(0):
    print("Epoch {}".format(i))
    ensemble_last_only(model=ppnet_ensemble_multi, log=print)
    train_ensemble(model=ppnet_ensemble_multi, dataloader=train_loader, optimizer=optimizer,
                        class_specific=class_specific, log=print)
    #check test accuracy
    accu = test_ensemble(model=ppnet_ensemble_multi, dataloader=test_loader,
                        class_specific=class_specific, log=print)
    if (i + 1) % 10 == 0:
        lr_scheduler.step()



