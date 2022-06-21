import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features

from receptive_field import compute_proto_layer_rf_info_v2

sys.path.insert(1, './Deformable-Convolution-V2-PyTorch')

from functions.norm_preserve_deform_conv_func import NormPreserveDeformConvFunction

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape, 
                 proto_layer_rf_info, num_classes, topk_k=1,
                 m=None, init_weights=True, add_on_layers_type='bottleneck', using_deform=True,
                 incorrect_class_connection=-1, deformable_conv_hidden_channels=0, prototype_dilation=2):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.m = m
        self.using_deform = using_deform
        self.relu_on_cos = True
        self.incorrect_class_connection = incorrect_class_connection
        self.input_vector_length = 64
        self.n_eps_channels = 2
        self.epsilon_val = 1e-4
        self.prototype_dilation = (prototype_dilation, prototype_dilation)
        self.prototype_padding = (1, 1)
        self.topk_k = topk_k

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)

        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        elif add_on_layers_type == 'identity':
            self.add_on_layers = nn.Sequential(nn.Identity())
        elif add_on_layers_type == 'upsample':
            self.add_on_layers = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.deformable_conv_out_channels = 2 * self.prototype_shape[-1] * self.prototype_shape[-2]
        self.deformable_conv_hidden_channels = deformable_conv_hidden_channels

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)


        # The convolution used to produce offsets for deformation
        # Add one to the number of channels to account for epsilon channel
        if not self.deformable_conv_hidden_channels:
            conv_offset_1 = nn.Conv2d(self.prototype_shape[-3] + self.n_eps_channels,
                                            self.deformable_conv_out_channels,
                                            kernel_size=(self.prototype_shape[-2]+1, self.prototype_shape[-1]+1),
                                            stride=(1, 1),
                                            padding=(1, 1),
                                            dilation=(1, 1),
                                            bias=False)
            self.conv_offset = nn.Sequential(conv_offset_1)
        else:
            conv_offset_1 = nn.Conv2d(self.prototype_shape[-3] + self.n_eps_channels,
                                            self.deformable_conv_hidden_channels,
                                            kernel_size=(self.prototype_shape[-2]+2, self.prototype_shape[-1]+2),
                                            stride=(1, 1),
                                            padding=(1, 1),
                                            dilation=(1, 1),
                                            bias=True)
            non_lin = nn.ReLU()
            conv_offset_2 = nn.Conv2d(self.deformable_conv_hidden_channels,
                                            self.deformable_conv_out_channels,
                                            kernel_size=(self.prototype_shape[-2], self.prototype_shape[-1]),
                                            stride=(1, 1),
                                            padding=(1, 1),
                                            dilation=(1, 1),
                                            bias=True)
                                            
            self.conv_offset = nn.Sequential(conv_offset_1, non_lin, conv_offset_2)

        for p in self.conv_offset.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.zeros_(p.weight)
                if p.bias is not None:
                    torch.nn.init.zeros_(p.bias)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes,
                                    bias=False) # do not use bias

        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x_feat = self.features(x)
        x = self.add_on_layers(x_feat)
        return x

    def cos_activation(self, x, is_train=True, 
                        prototypes_of_wrong_class=None):
        '''
        Takes convolutional features and gives arc distance as in
        https://arxiv.org/pdf/1801.07698.pdf
        '''
        input_vector_length = self.input_vector_length
        prototype_dilation = self.prototype_dilation
        '''
        This needs to be square root of window size, since conceptually using convolution
        for this is equivalent to stacking the prototype components and dotting that with
        the stacked x window. The length of such a stacked vector, if the components are normalized,
        is sqrt(n) where n is the number of vectors being stacked.
        '''
        normalizing_factor = (self.prototype_shape[-2] * self.prototype_shape[-1])**0.5

        # Append an additional channel of value epsilon to prevent 0 vector
        epsilon_channel_x = torch.ones(x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]) * self.epsilon_val
        epsilon_channel_x = epsilon_channel_x.cuda()
        epsilon_channel_x.requires_grad = False
        x = torch.cat((x, epsilon_channel_x), -3)
        # Normalize each 1 x 1 x latent piece to size s=64
        x_length = torch.sqrt(torch.sum(torch.square(x), dim=-3) + self.epsilon_val)
        x_length = x_length.view(x_length.size()[0], 1, x_length.size()[1], x_length.size()[2])
        x_normalized = input_vector_length * x / x_length 
        x_normalized = x_normalized / normalizing_factor

        # Similarly, append an additional channel of value epsilon to prototypes
        epsilon_channel_p = torch.ones(self.prototype_shape[0], self.n_eps_channels, self.prototype_shape[2], self.prototype_shape[3]) * self.epsilon_val
        epsilon_channel_p = epsilon_channel_p.cuda()
        epsilon_channel_p.requires_grad = False
        appended_protos = torch.cat((self.prototype_vectors, epsilon_channel_p), -3)

        # We normalize prototypes to unit length
        prototype_vector_length = torch.sqrt(torch.sum(torch.square(appended_protos), dim=-3) + self.epsilon_val)
        prototype_vector_length = prototype_vector_length.view(prototype_vector_length.size()[0], 
                                                                1,
                                                                prototype_vector_length.size()[1],
                                                                prototype_vector_length.size()[2])
        normalized_prototypes = appended_protos / (prototype_vector_length + self.epsilon_val)
        normalized_prototypes = normalized_prototypes / normalizing_factor

        # Compute offsets for this input
        offset = self.conv_offset(x_normalized)

        if self.using_deform:
            activations_dot = NormPreserveDeformConvFunction.apply(x_normalized, offset, 
                                                                normalized_prototypes, 
                                                                torch.zeros(self.prototype_shape[0]).cuda(), #bias
                                                                (1, 1), #stride
                                                                self.prototype_padding, #padding
                                                                prototype_dilation, #dilation
                                                                1, #groups
                                                                1, #deformable_groups
                                                                1, #im2col_step
                                                                True) #zero_padding
        else:
            activations_dot = F.conv2d(x_normalized, normalized_prototypes)

        marginless_activations = activations_dot / (input_vector_length * 1.01)
        if self.m == None or not is_train or prototypes_of_wrong_class == None:
            # If no margin is used
            activations = marginless_activations
        else:
            # This branch deals with subtractive margin
            wrong_class_margin = prototypes_of_wrong_class * self.m
            wrong_class_margin = wrong_class_margin.view(x.size()[0], self.prototype_vectors.size()[0], 1, 1)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, activations_dot.size()[-2], dim=-2)
            wrong_class_margin = torch.repeat_interleave(wrong_class_margin, activations_dot.size()[-1], dim=-1)
            penalized_angles = torch.arccos(activations_dot / (input_vector_length * 1.01)) - wrong_class_margin
            activations = torch.cos(torch.relu(penalized_angles))

        if self.relu_on_cos:
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)
            
        return activations, marginless_activations

    def prototype_activations(self, x, is_train=True, prototypes_of_wrong_class=None):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)
        activations, marginless_activations = self.cos_activation(conv_features, is_train=is_train, 
                                                        prototypes_of_wrong_class=prototypes_of_wrong_class)
        return activations, [marginless_activations, conv_features]

    def forward(self, x, is_train=True, prototypes_of_wrong_class=None):
        activations, additional_returns = self.prototype_activations(x, is_train=is_train, 
                                                prototypes_of_wrong_class=prototypes_of_wrong_class)
        marginless_activations = additional_returns[0]
        conv_features = additional_returns[1]
        if is_train:
            topk_k = self.topk_k
        else:
            topk_k = 1
        # global max pooling
        activations = activations.view(activations.shape[0], activations.shape[1], -1)

        # When k=1, this reduces to the maximum
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)
        mean_activations = torch.mean(topk_activations, dim=-1)

        marginless_max_activations = F.max_pool2d(marginless_activations,
                                      kernel_size=(marginless_activations.size()[2],
                                                   marginless_activations.size()[3]))
        marginless_max_activations = marginless_max_activations.view(-1, self.num_prototypes)

        logits = self.last_layer(mean_activations)
        marginless_logits = self.last_layer(marginless_max_activations)
        return logits, [mean_activations, marginless_logits, conv_features, marginless_max_activations]

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output = self.conv_features(x)
        _, marginless_activations = self.cos_activation(conv_output)
        return conv_output, marginless_activations

    '''
    Computes keypoint-wise orthogonality loss, ie encourage each piece
    of a prototype to be orthogonal to the others. Inspired by 
    https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf
    '''
    def get_prototype_orthogonalities(self):
        # We normalize prototypes to unit length
        prototype_vector_length = torch.sqrt(torch.sum(torch.square(self.prototype_vectors), dim=-3) + self.epsilon_val)
        prototype_vector_length = prototype_vector_length.view(prototype_vector_length.size()[0], 
                                                                1,
                                                                prototype_vector_length.size()[1],
                                                                prototype_vector_length.size()[2])
        normalized_prototypes = self.prototype_vectors / (prototype_vector_length + self.epsilon_val)

        # Reshape such that we have protos_per_class x total_parts_per_class x channel
        prototype_piece_matrices = normalized_prototypes.view(self.num_prototypes_per_class, 
                                                            self.num_prototypes // self.num_prototypes_per_class,
                                                            self.prototype_shape[-3],
                                                            self.prototype_shape[-2]*self.prototype_shape[-1])
        prototype_piece_matrices = prototype_piece_matrices.transpose(2,3).reshape(self.num_prototypes_per_class,
                                                                                    -1,
                                                                                    self.prototype_shape[-3])
        prototype_piece_matrices = prototype_piece_matrices.transpose(1,2)
        
        orthogonalities = torch.matmul(prototype_piece_matrices.transpose(-2,-1), prototype_piece_matrices)
        orthogonalities -= torch.eye((self.num_prototypes // self.num_prototypes_per_class) * self.prototype_shape[-2] * self.prototype_shape[-1]).cuda()
        # num_protos * 9 * 9
        return orthogonalities

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...],
                                 requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon_val)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def initialize_offset_weights(self):
        for m in self.conv_offset.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.normal_(m.weight, mean=0.0, std=0.00002)

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=self.incorrect_class_connection)



def construct_PPNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 512, 1, 1),
                    num_classes=200, topk_k=1, m=None,
                    add_on_layers_type='bottleneck', using_deform=True,
                    incorrect_class_connection=-1, deformable_conv_hidden_channels=128, prototype_dilation=2):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    print("prototype_shape is actually: ", prototype_shape)
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 topk_k=topk_k,
                 m=m,
                 init_weights=True,
                 add_on_layers_type=add_on_layers_type,
                 using_deform=using_deform,
                 incorrect_class_connection=incorrect_class_connection,
                 deformable_conv_hidden_channels=deformable_conv_hidden_channels,
                 prototype_dilation=prototype_dilation)

