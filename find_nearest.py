from push import get_deformation_info
import torch
import numpy as np

import heapq

import matplotlib.pyplot as plt
import os
import time

import cv2

from receptive_field import compute_rf_prototype
from helpers import makedir

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=1)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    
    plt.imsave(fname, img_rgb_float)

class DeformedProtoImage:
    def __init__(self, def_box_info, original_box_info, 
                 label, activation,
                 original_img=None, act_pattern=None,
                 offsets=None):
        self.def_box_info = def_box_info
        self.original_box_info = original_box_info
        self.offsets = offsets
        self.label = label
        self.activation = activation

        self.original_img = original_img
        self.act_pattern = act_pattern

    def __lt__(self, other):
        return self.activation < other.activation

    def __str__(self):
        return str(self.label) + str(self.activation)

class ImagePatch:

    def __init__(self, patch, label, activation,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.patch = patch
        self.label = label
        self.activation = activation

        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.activation < other.activation


class ImagePatchInfo:

    def __init__(self, label, activation):
        self.label = label
        self.activation = activation

    def __lt__(self, other):
        return self.activation < other.activation


# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                                         prototype_network_parallel, # pytorch network with prototype_vectors
                                         num_nearest_neighbors=5,
                                         preprocess_input_function=None, # normalize if needed
                                         full_save=False, # save all the images
                                         root_dir_for_saving_images='./nearest',
                                         log=print,
                                         deformable=True,
                                         prototype_layer_stride=1):
    assert not (full_save and deformable), "Error: setting deformable=True will override full_save=True. Set one to False for clarity."
    
    prototype_network_parallel.eval()
    '''
    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    '''
    log('find nearest patches')
    start = time.time()
    n_prototypes = prototype_network_parallel.module.num_prototypes
    
    prototype_shape = prototype_network_parallel.module.prototype_shape
    dilation = prototype_network_parallel.module.prototype_dilation

    protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info

    # allocate an array of n_prototypes number of heaps
    heaps = []
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    for index, (search_batch_input, search_y) in enumerate(dataloader):
        print('batch {}'.format(index))
        if preprocess_input_function is not None:
            search_batch = preprocess_input_function(search_batch_input)
        else:
            search_batch = search_batch_input

        with torch.no_grad():
            search_batch = search_batch.cuda()
            protoL_input_torch, proto_act_torch = \
                prototype_network_parallel.module.push_forward(search_batch)

        proto_act_ = np.copy(proto_act_torch.detach().cpu().numpy())
        protoL_input_torch = protoL_input_torch.detach()
        with torch.no_grad():
            if deformable:
                offsets, _ = get_deformation_info(protoL_input_torch, prototype_network_parallel)
                del _
        for img_idx, act_map in enumerate(proto_act_):
            for j in range(n_prototypes):
                # find the closest patches in this batch to prototype j

                highest_patch_activation_to_prototype_j = np.amax(act_map[j])
                if deformable:
                    highest_center_indices_in_activation_map_j = \
                        list(np.unravel_index(np.argmax(act_map[j],axis=None),
                                              act_map[j].shape))
                    highest_center_indices_in_activation_map_j = [0] + highest_center_indices_in_activation_map_j
                    fmap_height_start_index = highest_center_indices_in_activation_map_j[1] * prototype_layer_stride
                    fmap_width_start_index = highest_center_indices_in_activation_map_j[2] * prototype_layer_stride

                    original_img = search_batch_input[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))
                    original_img_size = search_batch_input[img_idx].shape[-1]

                    act_pattern = act_map[j]

                    # row_start, row_end, col_start, col_end
                    def_box_info = [[None] * 4] *(prototype_shape[-2] * prototype_shape[-1])
                    original_box_info = [[None] * 4] * (prototype_shape[-2] * prototype_shape[-1])

                    def_grp_index = 0
                    
                    for i in range(prototype_shape[-2]):
                        for k in range(prototype_shape[-1]):
                            # offsets go in order height offset, width offset
                            h_index = 2 * (k + prototype_shape[-2]*i)
                            w_index = h_index + 1

                            h_offset = offsets[img_idx, def_grp_index + h_index, fmap_height_start_index, fmap_width_start_index].item()
                            w_offset = offsets[img_idx, def_grp_index + w_index, fmap_height_start_index, fmap_width_start_index].item()

                            def_latent_space_row = fmap_height_start_index + h_offset + (i - prototype_shape[-2] // 2) * dilation[0] + (1 - prototype_shape[-2] % 2)
                            def_latent_space_col = fmap_width_start_index + w_offset + (k - prototype_shape[-1] // 2)* dilation[1] + (1 - prototype_shape[-1] % 2)

                            def_image_space_row_start = int(def_latent_space_row * original_img_size / act_map[j].shape[-2])
                            def_image_space_row_end = int((1 + def_latent_space_row) * original_img_size / act_map[j].shape[-2])
                            def_image_space_col_start = int(def_latent_space_col * original_img_size / act_map[j].shape[-1])
                            def_image_space_col_end = int((1 + def_latent_space_col) * original_img_size / act_map[j].shape[-1])

                            latent_space_row = fmap_height_start_index + (i - prototype_shape[-2] // 2) * dilation[0]
                            latent_space_col = fmap_width_start_index + (k - prototype_shape[-1] // 2)* dilation[1]

                            image_space_row_start = int(latent_space_row * original_img_size / act_map[j].shape[-2])
                            image_space_row_end = int((1 + latent_space_row) * original_img_size / act_map[j].shape[-2])
                            image_space_col_start = int(latent_space_col * original_img_size / act_map[j].shape[-1])
                            image_space_col_end = int((1 + latent_space_col) * original_img_size / act_map[j].shape[-1])

                            def_box_info[prototype_shape[-1] * i + k] = [def_image_space_row_start, def_image_space_row_end,
                                                                        def_image_space_col_start, def_image_space_col_end]
                            original_box_info[prototype_shape[-1] * i + k] = [image_space_row_start, image_space_row_end,
                                                                        image_space_col_start, image_space_col_end]

                    highest_patch = DeformedProtoImage(def_box_info=def_box_info, 
                                        original_box_info=original_box_info,
                                        label=search_y[img_idx],
                                        activation=highest_patch_activation_to_prototype_j,
                                        original_img=original_img,
                                        act_pattern=act_pattern,
                                        offsets=None)
                    del h_offset, w_offset, def_latent_space_row, def_latent_space_col
                    del fmap_height_start_index, fmap_width_start_index
                    del act_pattern
                elif full_save:
                    highest_patch_indices_in_activation_map_j = \
                        list(np.unravel_index(np.argmax(act_map[j],axis=None),
                                              act_map[j].shape))
                    highest_patch_indices_in_activation_map_j = [0] + highest_patch_indices_in_activation_map_j
                    highest_patch_indices_in_img = \
                        compute_rf_prototype(search_batch.size(2),
                                             highest_patch_indices_in_activation_map_j,
                                             protoL_rf_info)
                    highest_patch = \
                        search_batch_input[img_idx, :,
                                           highest_patch_indices_in_img[1]:highest_patch_indices_in_img[2],
                                           highest_patch_indices_in_img[3]:highest_patch_indices_in_img[4]]
                    highest_patch = highest_patch.numpy()
                    highest_patch = np.transpose(highest_patch, (1, 2, 0))

                    original_img = search_batch_input[img_idx].numpy()
                    original_img = np.transpose(original_img, (1, 2, 0))

                    act_pattern = act_map[j]

                    # 4 numbers: height_start, height_end, width_start, width_end
                    patch_indices = highest_patch_indices_in_img[1:5]

                    # construct the closest patch object
                    highest_patch = ImagePatch(patch=highest_patch,
                                               label=search_y[img_idx],
                                               activation=highest_patch_activation_to_prototype_j,
                                               original_img=original_img,
                                               act_pattern=act_pattern,
                                               patch_indices=patch_indices)
                else:
                    highest_patch = ImagePatchInfo(label=search_y[img_idx],
                                                   activation=highest_patch_activation_to_prototype_j)


                # add to the j-th heap
                if len(heaps[j]) < num_nearest_neighbors:
                    heapq.heappush(heaps[j], highest_patch)
                else:
                    # heappushpop runs more efficiently than heappush
                    # followed by heappop
                    heapq.heappushpop(heaps[j], highest_patch)
                    
                del highest_patch_activation_to_prototype_j
            del img_idx, act_map
        del search_batch_input, search_y, protoL_input_torch
        del search_batch, proto_act_torch, proto_act_
        if deformable:
            del offsets

    # after looping through the dataset every heap will
    # have the num_nearest_neighbors closest prototypes
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the num_nearest_neighbors closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if deformable:
            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
            makedir(dir_for_saving_images)

            labels = []

            for i, patch in enumerate(heaps[j]):
                # save the activation pattern of the original image where the patch comes from
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i+1) + '_act.npy'),
                        patch.act_pattern)
                
                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)

                img_with_boxes = patch.original_img.copy()
                colors = [(230/255, 25/255, 75/255), (60/255, 180/255, 75/255), (255/255, 225/255, 25/255),
                                (0.01, 130/255, 200/255), (245/255, 130/255, 48/255), (70/255, 240/255, 240/255),
                                (240/255, 50/255, 230/255), (170/255, 110/255, 40/255), (0.01,0.01,0.01)]
                for l in range(prototype_shape[-2]):
                    for k in range(prototype_shape[-1]):
                        def_box_info = patch.def_box_info[l*prototype_shape[-1]+k]
                        original_box_info = patch.original_box_info[l*prototype_shape[-1]+k]

                        cv2.rectangle(img_with_boxes,(def_box_info[2], def_box_info[0]),
                                                                (def_box_info[3], def_box_info[1]),
                                                                colors[l*prototype_shape[-1] + k],
                                                                1)
                        img_with_just_this_box = patch.original_img.copy()
                        cv2.rectangle(img_with_just_this_box,(def_box_info[2], def_box_info[0]),
                                                                (def_box_info[3], def_box_info[1]),
                                                                colors[l*prototype_shape[-1] + k],
                                                                1)
                        plt.imsave(os.path.join(dir_for_saving_images,
                                        'nearest-' + str(i) + '_patch_' + str(l*prototype_shape[-1] + k) + '-with_box.png'),
                            img_with_just_this_box,
                            vmin=0.0,
                            vmax=1.0)
                        
                plt.imsave(os.path.join(dir_for_saving_images,
                                'nearest-' + str(i) + '-with_box' + '.png'),
                            img_with_boxes,
                            vmin=0.0,
                            vmax=1.0)

                del original_box_info, def_box_info
            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)
        elif full_save:

            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
            makedir(dir_for_saving_images)

            labels = []

            for i, patch in enumerate(heaps[j]):
                # save the activation pattern of the original image where the patch comes from
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i+1) + '_act.npy'),
                        patch.act_pattern)
                
                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)
                
                # overlay (upsampled) activation on original image and save the result
                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                                   dsize=(img_size, img_size),
                                                   interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_original_with_heatmap.png'),
                           arr=overlayed_original_img,
                           vmin=0.0,
                           vmax=1.0)
                
                # if different from original image, save the patch (i.e. receptive field)
                if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                    np.save(os.path.join(dir_for_saving_images,
                                         'nearest-' + str(i+1) + '_receptive_field_indices.npy'),
                            patch.patch_indices)
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_receptive_field.png'),
                               arr=patch.patch,
                               vmin=0.0,
                               vmax=1.0)
                    # save the receptive field patch with heatmap
                    overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1],
                                                             patch.patch_indices[2]:patch.patch_indices[3], :]
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_receptive_field_with_heatmap.png'),
                               arr=overlayed_patch,
                               vmin=0.0,
                               vmax=1.0)
                    
                # save the highly activated patch    
                high_act_patch_indices = [patch.patch_indices[0],patch.patch_indices[1],
                                            patch.patch_indices[2],patch.patch_indices[3]]
                                                    #find_high_activation_crop(upsampled_act_pattern)
                high_act_patch = patch.original_img[patch.patch_indices[0]:patch.patch_indices[1],
                                                    patch.patch_indices[2]:patch.patch_indices[3], :]
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i+1) + '_high_act_patch_indices.npy'),
                        high_act_patch_indices)
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i+1) + '_high_act_patch.png'),
                           arr=high_act_patch,
                           vmin=0.0,
                           vmax=1.0)
                # save the original image with bounding box showing high activation patch
                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                       'nearest-' + str(i+1) + '_high_act_patch_in_original_img.png'),
                                 img_rgb=patch.original_img,
                                 bbox_height_start=high_act_patch_indices[0],
                                 bbox_height_end=high_act_patch_indices[1],
                                 bbox_width_start=high_act_patch_indices[2],
                                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
            
            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)


    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)

    end = time.time()
    log('\tfind nearest patches time: \t{0}'.format(end - start))

    return labels_all_prototype
