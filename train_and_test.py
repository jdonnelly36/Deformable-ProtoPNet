import time
import torch

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, subtractive_margin=True, use_ortho_loss=False):
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
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    total_l2 = 0
    total_ortho_loss = 0
    max_offset = 0

    if use_l1_mask:
        l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
        l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
    else:
        l1 = model.module.last_layer.weight.norm(p=1) 

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            if subtractive_margin:
                output, additional_returns = model(input, is_train=is_train, 
                                                    prototypes_of_wrong_class=prototypes_of_wrong_class)
            else:
                output, additional_returns = model(input, is_train=is_train, prototypes_of_wrong_class=None)

            max_activations = additional_returns[0]
            marginless_logits = additional_returns[1]
            conv_features = additional_returns[2]
            
            with torch.no_grad():
                prototype_shape = model.module.prototype_shape
                epsilon_val = model.module.epsilon_val
                n_eps_channels = model.module.n_eps_channels

                x = conv_features
                epsilon_channel_x = torch.ones(x.shape[0], n_eps_channels, x.shape[2], x.shape[3]) * epsilon_val
                epsilon_channel_x = epsilon_channel_x.cuda()
                x = torch.cat((x, epsilon_channel_x), -3)
                input_vector_length = model.module.input_vector_length
                normalizing_factor = (prototype_shape[-2] * prototype_shape[-1])**0.5
                
                input_length = torch.sqrt(torch.sum(torch.square(x), dim=-3))
                input_length = input_length.view(input_length.size()[0], 1, input_length.size()[1], input_length.size()[2]) 
                input_normalized = input_vector_length * x / input_length
                input_normalized = input_normalized / normalizing_factor
                offsets = model.module.conv_offset(input_normalized)

            epsilon_val = model.module.epsilon_val
            n_eps_channels = model.module.n_eps_channels
            epsilon_channel_x = torch.ones(conv_features.shape[0], n_eps_channels, conv_features.shape[2], conv_features.shape[3]) * epsilon_val
            epsilon_channel_x = epsilon_channel_x.cuda()
            conv_features = torch.cat((conv_features, epsilon_channel_x), -3)
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                # calculate cluster cost
                correct_class_prototype_activations, _ = torch.max(max_activations * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(correct_class_prototype_activations)

                # calculate separation cost
                incorrect_class_prototype_activations, _ = \
                    torch.max(max_activations * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(incorrect_class_prototype_activations)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(max_activations * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                offset_l2 = offsets.norm()

            else:
                max_activations, _ = torch.max(max_activations, dim=1)
                cluster_cost = torch.mean(max_activations)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(marginless_logits.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_l2 += offset_l2
            total_avg_separation_cost += avg_separation_cost.item()
            batch_max = torch.max(torch.abs(offsets))
            max_offset = torch.max(torch.Tensor([batch_max, max_offset]))

            '''
            Compute keypoint-wise orthogonality loss, i.e. encourage each piece
            of a prototype to be orthogonal to the others.
            '''
            orthogonalities = model.module.get_prototype_orthogonalities()
            orthogonality_loss = torch.norm(orthogonalities)
            total_ortho_loss += orthogonality_loss.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    total_ortho_loss += orthogonality_loss.item()
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          + coefs['offset_bias_l2'] * offset_l2)
                    if use_ortho_loss:
                        loss += coefs['orthogonality_loss'] * orthogonality_loss
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            
            loss.backward(retain_graph=True)
            optimizer.step()

        del input, batch_max, target, output, predicted, max_activations
        del offsets, conv_features, prototypes_of_correct_class, prototypes_of_wrong_class
        del epsilon_channel_x, input_normalized, input_vector_length, x, additional_returns
        del marginless_logits, offset_l2, cross_entropy, cluster_cost, separation_cost
        del orthogonalities, orthogonality_loss, correct_class_prototype_activations
        del incorrect_class_prototype_activations, avg_separation_cost, input_length
        if is_train:
            del loss

    end = time.time()
    log('\ttime: \t{0}'.format(end -  start))
    if use_ortho_loss:
        log('\tUsing ortho loss')
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\torthogonality loss:\t{0}'.format(total_ortho_loss / n_batches))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    log('\tavg l2: \t\t{0}'.format(total_l2 / n_batches))
    if coefs is not None:
        log('\tavg l2 with weight: \t\t{0}'.format(coefs['offset_bias_l2'] * total_l2 / n_batches))
        log('\torthogonality loss with weight:\t{0}'.format(coefs['orthogonality_loss'] * total_ortho_loss / n_batches))
    log('\tmax offset: \t{0}'.format(max_offset))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=False, coefs=None, 
            log=print, subtractive_margin=True, use_ortho_loss=False):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, 
                          subtractive_margin=subtractive_margin, use_ortho_loss=use_ortho_loss)


def test(model, dataloader, class_specific=False, log=print, subtractive_margin=True):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, subtractive_margin=subtractive_margin)


def last_only(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.conv_offset.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\tlast layer')


def warm_only(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\twarm')

def warm_pre_offset(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\twarm pre offset')

def joint(model, log=print, last_layer_fixed=True):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.conv_offset.parameters():
        p.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = not last_layer_fixed
    
    log('\tjoint')
