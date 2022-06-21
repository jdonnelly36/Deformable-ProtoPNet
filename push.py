import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

from helpers import makedir, find_high_activation_crop

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes

    # saves the closest distance seen so far
    global_max_proto_act = np.full(n_prototypes, -np.inf)
    # saves the patch representation that gives the current smallest distance
    global_max_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    log('\tExecuting push ...')
    
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_max_proto_act,
                                   global_max_fmap_patches,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    n_prototypes = prototype_network_parallel.module.num_prototypes
    
    global_max_acts = np.array([-1] * n_prototypes)
    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        search_batch_input = search_batch_input.cuda()
        # this computation currently is not parallelized
        _, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch_input)
        proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())
        # get max activation for each proto
        max_acts = np.max(proto_act_, axis=(0, 2, 3))
        
        global_max_acts = np.maximum(global_max_acts, max_acts)
        del _, proto_act_torch, proto_act_, max_acts, search_batch_input
    
    prototype_update = np.reshape(global_max_fmap_patches,
                                  tuple(prototype_network_parallel.module.prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

    
    save_projected_prototype_images(prototype_network_parallel=prototype_network_parallel,
                                    dataloader=dataloader,
                                    dir_for_saving_prototypes=proto_epoch_dir,
                                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                                    image_indices=proto_bound_boxes,
                                    prototype_layer_stride=prototype_layer_stride)

# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_max_proto_act, # this will be updated
                               global_max_fmap_patches, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_act_torch

    if class_specific:
        # Index class_to_img_index dict with class number, return list of images
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # img_y is the image's integer label
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes

    for j in range(n_prototypes):
        class_index = j
        
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[class_index]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_act_j = proto_act_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            proto_act_j = proto_act_[:,j,:,:]
        batch_max_proto_act_j = np.amax(proto_act_j)
        
        if batch_max_proto_act_j > global_max_proto_act[j]:
            batch_argmax_proto_act_j = \
                list(np.unravel_index(np.argmax(proto_act_j, axis=None),
                                      proto_act_j.shape))
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmax_proto_act_j[0] = class_to_img_index_dict[target_class][batch_argmax_proto_act_j[0]]

            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmax_proto_act_j[0]
            fmap_height_start_index = batch_argmax_proto_act_j[1] * prototype_layer_stride
            fmap_width_start_index = batch_argmax_proto_act_j[2] * prototype_layer_stride


            with torch.no_grad():
                offsets, input_normalized = get_deformation_info(torch.Tensor(protoL_input_).cuda(), prototype_network_parallel)
                #offset_filters = prototype_network_parallel.module.conv_offset.weight
                padding_type = 'zero'
                dilation = prototype_network_parallel.module.prototype_dilation
                padding_size = prototype_network_parallel.module.prototype_padding

                
                batch_max_fmap_patch_j = get_deformed_patch(input_normalized[img_index_in_batch, :, :, :].cpu(), 
                                    offsets[img_index_in_batch, :, fmap_height_start_index, fmap_width_start_index],
                                    (fmap_height_start_index, fmap_width_start_index),
                                    prototype_shape, dilation, padding_type, padding_size)

            global_max_proto_act[j] = batch_max_proto_act_j
            global_max_fmap_patches[j] = batch_max_fmap_patch_j
            
            # get the whole image
            original_img_j = search_batch_input[batch_argmax_proto_act_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # find the highly activated region of the original image
            proto_act_img_j = proto_act_[img_index_in_batch, j, :, :]
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = batch_argmax_proto_act_j[0] + start_index_of_search_batch
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[batch_argmax_proto_act_j[0]].item()
            
    if class_specific:
        del class_to_img_index_dict

# update each prototype for current search batch
def save_projected_prototype_images(prototype_network_parallel,
                                    dataloader,
                                    dir_for_saving_prototypes,
                                    prototype_self_act_filename_prefix,
                                    prototype_img_filename_prefix,
                                    image_indices,
                                    prototype_layer_stride):
    n_prototypes = prototype_network_parallel.module.num_prototypes

    with torch.no_grad():
        global_max_acts = np.array([-1] * n_prototypes)
        for push_iter, (search_batch_input, search_y) in enumerate(dataloader):

            num_classes = prototype_network_parallel.module.num_classes
            prototype_shape = prototype_network_parallel.module.prototype_shape
            dilation = prototype_network_parallel.module.prototype_dilation

            # Index class_to_img_index dict with class number, return list of images
            class_to_img_index_dict = {key: [] for key in range(num_classes)}
            # img_y is the image's integer label
            for img_index, img_y in enumerate(search_y):
                img_label = img_y.item()
                class_to_img_index_dict[img_label].append(img_index)

            search_batch_input = search_batch_input.cuda()
            # this computation currently is not parallelized
            proto_in, proto_act_torch = prototype_network_parallel.module.push_forward(search_batch_input)
            proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())
            # get max activation for each proto
            max_acts = np.max(proto_act_, axis=(0, 2, 3))

            for j in range(n_prototypes):
                if max_acts[j] < global_max_acts[j]:
                    continue

                # target_class is the class of the class_specific prototype
                target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
                # if there is not images of the target_class from this batch
                # we go on to the next prototype
                if len(class_to_img_index_dict[target_class]) == 0:
                    continue
                proto_act_j = proto_act_[class_to_img_index_dict[target_class]][:,j,:,:]

                batch_argmax_proto_act_j = \
                    list(np.unravel_index(np.argmax(proto_act_j, axis=None),
                                                        proto_act_j.shape))
                fmap_height_start_index = batch_argmax_proto_act_j[1] * prototype_layer_stride
                fmap_width_start_index = batch_argmax_proto_act_j[2] * prototype_layer_stride
                
                if image_indices[j, 0] > dataloader.batch_size * (push_iter + 1) \
                    or image_indices[j, 0] < dataloader.batch_size * push_iter:
                    continue

                img_index_in_batch = image_indices[j, 0] % dataloader.batch_size

                original_img_j = search_batch_input[img_index_in_batch]
                original_img_j = original_img_j.cpu().numpy()
                original_img_j = np.transpose(original_img_j, (1, 2, 0))
                original_img_size = original_img_j.shape[0]

                original_img_j_with_boxes = 0.999*(original_img_j - np.min(original_img_j) + 1e-4)/np.max(original_img_j)
                original_img_j_with_boxes = original_img_j_with_boxes.astype(np.float32).copy()

                proto_act_img_j = proto_act_[img_index_in_batch, j, :, :]
                upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                                    interpolation=cv2.INTER_CUBIC)
                proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
                # crop out the image patch with high activation as prototype image
                proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                                proto_bound_j[2]:proto_bound_j[3], :]

                if dir_for_saving_prototypes is not None:
                    if prototype_self_act_filename_prefix is not None:
                        # save the numpy array of the prototype self activation
                        np.save(os.path.join(dir_for_saving_prototypes,
                                                prototype_self_act_filename_prefix + str(j) + '.npy'),
                                proto_act_img_j)
                    if prototype_img_filename_prefix is not None:
                        # Add a rectangle for each deformable part of the prototype
                        offsets, _ = get_deformation_info(proto_in, prototype_network_parallel)
                        
                        offsets_j = offsets[img_index_in_batch]

                        colors = [(230/255, 25/255, 75/255), (60/255, 180/255, 75/255), (255/255, 225/255, 25/255),
                                (0.01, 130/255, 200/255), (245/255, 130/255, 48/255), (70/255, 240/255, 240/255),
                                (240/255, 50/255, 230/255), (170/255, 110/255, 40/255), (0.01,0.01,0.01)]
                        
                        for i in range(prototype_shape[-2]):
                            for k in range(prototype_shape[-1]):
                                h_index = 2 * (k + prototype_shape[-2]*i)
                                w_index = h_index + 1
                                h_offset = offsets_j[h_index, fmap_height_start_index, fmap_width_start_index]
                                w_offset = offsets_j[w_index, fmap_height_start_index, fmap_width_start_index]

                                # Subtract prototype_shape // 2 because fmap start indices give the center location, and we start with top left
                                def_latent_space_row = fmap_height_start_index + h_offset + (i - prototype_shape[-2] // 2) * dilation[0] - (1 - prototype_shape[-2] % 2)
                                def_latent_space_col = fmap_width_start_index + w_offset + (k - prototype_shape[-1] // 2)* dilation[1] - (1 - prototype_shape[-1] % 2)

                                def_image_space_row_start = int(def_latent_space_row * original_img_size / proto_act_img_j.shape[-2])
                                def_image_space_row_end = int((1 + def_latent_space_row) * original_img_size / proto_act_img_j.shape[-2])
                                def_image_space_col_start = int(def_latent_space_col * original_img_size / proto_act_img_j.shape[-1])
                                def_image_space_col_end = int((1 + def_latent_space_col) * original_img_size / proto_act_img_j.shape[-1])

                                img_with_just_this_box = original_img_j.copy()
                                cv2.rectangle(img_with_just_this_box,(def_image_space_col_start, def_image_space_row_start),
                                                                        (def_image_space_col_end, def_image_space_row_end),
                                                                        colors[i*prototype_shape[-1] + k],
                                                                        1)
                                plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '_patch_' + str(i*prototype_shape[-1] + k) + '-with_box.png'),
                                    img_with_just_this_box,
                                    vmin=0.0,
                                    vmax=1.0)

                                cv2.rectangle(original_img_j_with_boxes,(def_image_space_col_start, def_image_space_row_start),
                                                                        (def_image_space_col_end, def_image_space_row_end),
                                                                        colors[i*prototype_shape[-1] + k],
                                                                        1)
                                if not (def_image_space_col_start < 0 
                                    or def_image_space_row_start < 0
                                    or def_image_space_col_end >= original_img_j.shape[0]
                                    or def_image_space_row_end >= original_img_j.shape[1]):
                                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                    prototype_img_filename_prefix + str(j) + '_patch_' + str(i*prototype_shape[-1] + k) + '.png'),
                                        original_img_j[def_image_space_col_start:def_image_space_col_end, def_image_space_row_start:def_image_space_row_end, :],
                                        vmin=0.0,
                                        vmax=1.0)

                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-with_box' + str(j) + '.png'),
                                    original_img_j_with_boxes,
                                    vmin=0.0,
                                    vmax=1.0)

                        # save the whole image containing the prototype as png
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                                    original_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
                        
                        # save the prototype image (highly activated region of the whole image)
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '.png'),
                                    proto_img_j,
                                    vmin=0.0,
                                    vmax=1.0)
            global_max_acts = np.maximum(global_max_acts, max_acts)

            del proto_act_torch

        np.save(dir_for_saving_prototypes + 'proto_max_activations.npy', global_max_acts)

'''
This function gets the offsets and normalized input for a given input'''
def get_deformation_info(protoL_input_, prototype_network_parallel):
    prototype_shape = prototype_network_parallel.module.prototype_shape
    epsilon_val = prototype_network_parallel.module.epsilon_val
    n_eps_channels = prototype_network_parallel.module.n_eps_channels

    x = protoL_input_
    epsilon_channel_x = torch.ones(x.shape[0], n_eps_channels, x.shape[2], x.shape[3]) * epsilon_val
    epsilon_channel_x = epsilon_channel_x.cuda()
    x = torch.cat((x, epsilon_channel_x), -3)
    input_vector_length = prototype_network_parallel.module.input_vector_length
    normalizing_factor = (prototype_shape[-2] * prototype_shape[-1])**0.5
    
    input_length = torch.sqrt(torch.sum(torch.square(x), dim=-3))
    input_length = input_length.view(input_length.size()[0], 1, input_length.size()[1], input_length.size()[2]) 
    input_normalized = input_vector_length * x / input_length
    input_normalized = input_normalized / normalizing_factor

    offsets = prototype_network_parallel.module.conv_offset(input_normalized).cpu()
    return offsets, input_normalized

'''
This function gets the deformed version of the input
at the given location using the given offsets.
'''
def get_deformed_patch(input, offsets, optimal_location, prototype_shape, 
                        dilation=(1,1), padding='nearest', padding_size=(1,1)):
    with torch.no_grad():
        result = np.zeros((prototype_shape[1], prototype_shape[2], prototype_shape[3]))
        n_eps_layers = int(input.shape[-3] - prototype_shape[1])
        for i in range(prototype_shape[-2]):
            for j in range(prototype_shape[-1]):
                '''
                - The offset is setup as batch * see_next * spatial * spatial
                    - see_next = 2 * in_h * in_w; it is ordered as 
                    - (h_00, w_00, h_01, w_01, …, h_33, w_33)
                '''
                h_index = 2 * (j + prototype_shape[-2]*i)
                w_index = h_index + 1
                h_offset = offsets[h_index]
                w_offset = offsets[w_index]

                input_row = optimal_location[0] + h_offset + i * dilation[0] - padding_size[0]
                input_col = optimal_location[1] + w_offset + j * dilation[1] - padding_size[1]

                interpolated_input = interpolate_from_location(input, (input_row, input_col), padding)
                if n_eps_layers > 0:
                    result[:, i, j] =  interpolated_input[:-n_eps_layers]
                else:
                    result[:, i, j] =  interpolated_input

        return result

# location is (row, col) tuple
def interpolate_from_location(input, location, padding='nearest'):
    assert padding=='zero' or padding=='nearest', 'Currently only 0-padding and nearest padding are implemented'

    h_low = int(np.floor(location[0]))
    h_high = int(np.ceil(location[0]))
    w_low = int(np.floor(location[1]))
    w_high = int(np.ceil(location[1]))
    h_offset = location[0] - h_low
    w_offset = location[1] - w_low
    
    # Shape of values is 4 x channels (one for each corner)
    # ordered as top left, top right, bot left, bot right
    values = np.empty((4, input.shape[-3]))
    values[:,:] = None

    # For now, just drop any offsetted bits that are out of bounds
    if padding == 'zero':
        if h_low < 0 or h_low > input.shape[-1] - 1:
            values[0, :] = 0
            values[1, :] = 0
        if h_high > input.shape[-1] - 1 or h_high < 0:
            values[2, :] = 0
            values[3, :] = 0
        if w_low < 0 or w_low > input.shape[-2] - 1 :
            values[0, :] = 0
            values[2, :] = 0
        if w_high > input.shape[-2] - 1 or w_high < 0:
            values[1, :] = 0
            values[3, :] = 0
    elif padding == 'nearest':
        if h_low < 0:
            h_low = 0
        if h_high > input.shape[-2] - 1:
            h_high = input.shape[-2] - 1
        if w_low < 0:
            w_low = 0
        if w_high > input.shape[-1] - 1:
            w_high = input.shape[-1] - 1

    if np.isnan(values[0, :]).any():
        values[0, :] = input[:, h_low, w_low]
    if np.isnan(values[1, :]).any():
        values[1, :] = input[:, h_low, w_high]
    if np.isnan(values[2, :]).any():
        values[2, :] = input[:, h_high, w_low]
    if np.isnan(values[3, 0]).any():
        values[3, :] = input[:, h_high, w_high]


    result = (1 - h_offset) * (1 - w_offset) * values[0, :] +\
            (1 - h_offset) * w_offset * values[1, :] +\
            h_offset * (1 - w_offset) * values[2, :] +\
            h_offset * w_offset * values[3, :]

    return result