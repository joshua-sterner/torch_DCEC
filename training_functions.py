import utils
import time
import torch
import numpy as np
import copy
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import math


# Training function (from my torch_DCEC implementation, kept for completeness)
def train_model(model, dataloader, criteria, optimizers, schedulers, num_epochs, params):

    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    pretrain = params['pretrain']
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']
    pretrain_epochs = params['pretrain_epochs']
    gamma = params['gamma']
    update_interval = params['update_interval']
    tol = params['tol']

    dl = dataloader

    # Pretrain or load weights
    if pretrain:
        while True:
            pretrained_model = pretraining(model, copy.deepcopy(dl), criteria[0], optimizers[1], schedulers[1], pretrain_epochs, params)
            if pretrained_model:
                break
            else:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()
        model = pretrained_model
    else:
        try:
            utils.load_pretrained_net(model, pretrained)
            utils.print_both(txt_file, 'Pretrained weights loaded from file: ' + str(pretrained))
        except:
            print("Couldn't load pretrained weights")

    # Initialise clusters
    if params['train_init_clusters'] or pretrain:
        init_clusters(txt_file, model, dl, params)
    utils.print_both(txt_file, '\nBegin clusters training')

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Initial target distribution
    utils.print_both(txt_file, '\nUpdating target distribution')
    output_distribution, labels, preds_prev, embedding, label_img = calculate_predictions(model, copy.deepcopy(dl), params)
    if board:
        writer.add_embedding(embedding, metadata=labels, global_step=0, label_img=label_img, tag='embedding_layer')
        writer.add_embedding(output_distribution, metadata=labels, global_step=0, label_img=label_img, tag='clustering_output')
    target_distribution = target(output_distribution)
    if params['class_dependent_metrics']:
        nmi = utils.metrics.nmi(labels, preds_prev)
        ari = utils.metrics.ari(labels, preds_prev)
        acc = utils.metrics.acc(labels, preds_prev)
        utils.print_both(txt_file,
                         'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\n'.format(nmi, ari, acc))

        if board:
            niter = 0
            writer.add_scalar('/NMI', nmi, niter)
            writer.add_scalar('/ARI', ari, niter)
            writer.add_scalar('/Acc', acc, niter)

    update_iter = 1
    finished = False

    # Go through all epochs
    for epoch in range(num_epochs):
        if epoch < params['zero_gamma_epochs']:
            gamma = 0
        else:
            gamma = params['gamma']

        utils.print_both(txt_file, 'Epoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file,  '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0
        running_loss_rec = 0.0
        running_loss_clust = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, provided_labels = data

            inputs = inputs.to(device)

            # Uptade target distribution, chack and print performance
            if (batch_num - 1) % update_interval == 0 and not (batch_num == 1 and epoch == 0):
                utils.print_both(txt_file, '\nUpdating target distribution:')
                output_distribution, labels, preds, embedding, label_img = calculate_predictions(model, dataloader, params)
                if (board and batch_num == 1 and epoch % params['embedding_interval'] == 0):
                    writer.add_embedding(embedding, metadata=labels, global_step=epoch, label_img=label_img, tag='embedding_layer')
                    writer.add_embedding(output_distribution, metadata=labels, global_step=epoch, label_img=label_img, tag='clustering_output')

                target_distribution = target(output_distribution)
                if params['class_dependent_metrics']:
                    nmi = utils.metrics.nmi(labels, preds)
                    ari = utils.metrics.ari(labels, preds)
                    acc = utils.metrics.acc(labels, preds)
                    utils.print_both(txt_file,
                                     'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
                    if board:
                        niter = update_iter
                        writer.add_scalar('/NMI', nmi, niter)
                        writer.add_scalar('/ARI', ari, niter)
                        writer.add_scalar('/Acc', acc, niter)
                update_iter += 1

                # check stop criterion
                delta_label = np.sum(preds != preds_prev).astype(np.float32) / preds.shape[0]
                preds_prev = np.copy(preds)
                if delta_label < tol:
                    utils.print_both(txt_file, 'Label divergence ' + str(delta_label) + '< tol ' + str(tol))
                    utils.print_both(txt_file, 'Reached tolerance threshold. Stopping training.')
                    finished = True
                    break

            tar_dist = target_distribution[((batch_num - 1) * batch):(batch_num*batch), :]
            tar_dist = torch.from_numpy(tar_dist).to(device)
            # print(tar_dist)

            # zero the parameter gradients
            optimizers[0].zero_grad()

            # Calculate losses and backpropagate
            with torch.set_grad_enabled(True):
                outputs, clusters, _ = model(inputs)
                loss_rec = criteria[0](outputs, inputs)
                loss_clust = criteria[1](torch.log(clusters), tar_dist) / batch
                if (params['DEC']):
                    loss = gamma*loss_clust
                else:
                    loss = loss_rec + gamma*loss_clust
                loss.backward()
                optimizers[0].step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)
            running_loss_rec += loss_rec.item() * inputs.size(0)
            running_loss_clust += loss_clust.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_batch_rec = loss_rec.item()
            loss_batch_clust = loss_clust.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_rec = running_loss_rec / ((batch_num - 1) * batch + inputs.size(0))
            loss_accum_clust = running_loss_clust / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Epoch: [{0}][{1}/{2}]\t'
                                           'Loss {3:.4f} ({4:.4f})\t'
                                           'Loss_recovery {5:.4f} ({6:.4f})\t'
                                           'Loss clustering {7:.4f} ({8:.4f})\t'.format(epoch + 1, batch_num,
                                                                                        len(dataloader),
                                                                                        loss_batch,
                                                                                        loss_accum, loss_batch_rec,
                                                                                        loss_accum_rec,
                                                                                        loss_batch_clust,
                                                                                        loss_accum_clust))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('/Loss', loss_accum, niter)
                    writer.add_scalar('/Loss_recovery', loss_accum_rec, niter)
                    writer.add_scalar('/Loss_clustering', loss_accum_clust, niter)
            batch_num = batch_num + 1

            # Print image to tensorboard
            if batch_num == len(dataloader) and (epoch+1) % 5:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Clustering/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        schedulers[0].step()
        if finished: break

        epoch_loss = running_loss / dataset_size
        epoch_loss_rec = running_loss_rec / dataset_size
        epoch_loss_clust = running_loss_clust / dataset_size

        if board:
            writer.add_scalar('/Loss' + '/Epoch', epoch_loss, epoch + 1)
            writer.add_scalar('/Loss_rec' + '/Epoch', epoch_loss_rec, epoch + 1)
            writer.add_scalar('/Loss_clust' + '/Epoch', epoch_loss_clust, epoch + 1)

        utils.print_both(txt_file, 'Loss: {0:.4f}\tLoss_recovery: {1:.4f}\tLoss_clustering: {2:.4f}'.format(epoch_loss,
                                                                                                            epoch_loss_rec,
                                                                                                            epoch_loss_clust))

        # If wanted to do some criterium in the future (for now useless)
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    model.eval()
    output_distribution, labels, preds, embedding, label_img = calculate_predictions(model, dataloader, params)
    if params['class_dependent_metrics']:
        nmi = utils.metrics.nmi(labels, preds)
        ari = utils.metrics.ari(labels, preds)
        acc = utils.metrics.acc(labels, preds)
        utils.print_both(txt_file,
                         'NMI: {0:.5f}\tARI: {1:.5f}\tAcc {2:.5f}\t'.format(nmi, ari, acc))
        niter = update_iter
        writer.add_scalar('/NMI', nmi, niter)
        writer.add_scalar('/ARI', ari, niter)
        writer.add_scalar('/Acc', acc, niter)
    if board:
        writer.add_embedding(embedding, metadata=labels, global_step=num_epochs, label_img=label_img, tag='embedding_layer')
        writer.add_embedding(output_distribution, metadata=labels, global_step=num_epochs, label_img=label_img, tag='clustering_output')

    log_func = lambda x: utils.print_both(txt_file, x)

    if params['use_ssim']:
        ssim_metrics = utils.matrix_metrics(params['ssim_matrix'], params['num_clusters'], preds, labels)
        utils.log_matrix_metrics(log_func, ssim_metrics, 'SSIM')

    if params['use_mse']:
        mse_metrics = utils.matrix_metrics(params['mse_matrix'], params['num_clusters'], preds, labels)
        utils.log_matrix_metrics(log_func, mse_metrics, 'MSE')

    #if params['use_ssim']:
    #    # masks out self-pairs -- including these would increase the
    #    # average ssim for in-cluster pairs.
    #    self_pair_mask = 1 - np.identity(len(preds))
    #    total_sum_ssim_in = 0
    #    total_num_pairs_in = 0
    #    total_sum_ssim_out = 0
    #    total_num_pairs_out = 0
    #    
    #    # x and y are used to select pairs from the matrix. Each element of
    #    # x and y is a mask representing the images in (in the case of x) or out
    #    # (in the case of y) of a cluster. These masks are 2-dimensional so that
    #    # the transpose operation can be used.
    #    #
    #    # this will be 1 for each element predicted in class, 0 otherwise
    #    x = np.zeros((params['num_clusters'], 1, len(preds)))
    #    # this will be 0 for each element predicted in class, 1 otherwise
    #    y = np.ones((params['num_clusters'], 1, len(preds)))

    #    encountered_predictions = np.zeros(params['num_clusters'])
    #    for i in range(len(preds)):
    #        encountered_predictions[preds[i]] += 1
    #        predicted_class_index = preds[i]
    #        x[predicted_class_index][0][labels[i]] = 1
    #        y[predicted_class_index][0][labels[i]] = 0
    #    utils.print_both(txt_file, f'Predictions per cluster: {encountered_predictions}')
    #    for i in range(params['num_clusters']):
    #        if encountered_predictions[i] == 0:
    #            utils.print_both(txt_file, f'WARNING: No inputs predicted to exist withing cluster {i}.')
    #        # select in-cluster pairs
    #        pairs_in_mask = x[i] * x[i].transpose() * self_pair_mask
    #        pairs_in = params['ssim_matrix'] * pairs_in_mask
    #        num_pairs_in = sum(sum(pairs_in_mask > 0))
    #        # select pairs with one image in the cluster and one image not in the cluster
    #        pairs_out_mask = x[i] * y[i].transpose() + x[i].transpose() * y[i]
    #        pairs_out = params['ssim_matrix'] * pairs_out_mask
    #        num_pairs_out = sum(sum(pairs_out_mask > 0))

    #        sum_ssim_in = sum(sum(pairs_in))
    #        sum_ssim_out = sum(sum(pairs_out))
    #        avg_ssim_in = sum_ssim_in/num_pairs_in
    #        avg_ssim_out = sum_ssim_out/num_pairs_out

    #        total_sum_ssim_in += sum_ssim_in
    #        total_sum_ssim_out += sum_ssim_out
    #        total_num_pairs_in += num_pairs_in
    #        total_num_pairs_out += num_pairs_out

    #        utils.print_both(txt_file, f'Cluster {i}: Average SSIM (in cluster): {avg_ssim_in}')
    #        utils.print_both(txt_file, f'Cluster {i}: Average SSIM (out cluster): {avg_ssim_out}')
    #    total_avg_ssim_in = total_sum_ssim_in / total_num_pairs_in
    #    total_avg_ssim_out = total_sum_ssim_out / total_num_pairs_out
    #    utils.print_both(txt_file, f'SSIM (in cluster): {total_avg_ssim_in}')
    #    utils.print_both(txt_file, f'SSIM (out cluster): {total_avg_ssim_out}')


    update_iter += 1


    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# Pretraining function for recovery loss only
def pretraining(model, dataloader, criterion, optimizer, scheduler, num_epochs, params):
    # Note the time
    since = time.time()

    # Unpack parameters
    writer = params['writer']
    if writer is not None: board = True
    txt_file = params['txt_file']
    pretrained = params['model_files'][1]
    print_freq = params['print_freq']
    dataset_size = params['dataset_size']
    device = params['device']
    batch = params['batch']

    # Prep variables for weights and accuracy of the best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    # Go through all epochs
    for epoch in range(num_epochs):
        utils.print_both(txt_file, 'Pretraining:\tEpoch {}/{}'.format(epoch + 1, num_epochs))
        utils.print_both(txt_file, '-' * 10)

        model.train(True)  # Set model to training mode

        running_loss = 0.0

        # Keep the batch number for inter-phase statistics
        batch_num = 1
        # Images to show
        img_counter = 0

        # Iterate over data.
        for data in dataloader:
            # Get the inputs and labels
            inputs, _ = data
            inputs = inputs.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

            # For keeping statistics
            running_loss += loss.item() * inputs.size(0)

            # Some current stats
            loss_batch = loss.item()
            loss_accum = running_loss / ((batch_num - 1) * batch + inputs.size(0))

            if batch_num % print_freq == 0:
                utils.print_both(txt_file, 'Pretraining:\tEpoch: [{0}][{1}/{2}]\t'
                           'Loss {3:.4f} ({4:.4f})\t'.format(epoch + 1, batch_num, len(dataloader),
                                                             loss_batch,
                                                             loss_accum))
                if board:
                    niter = epoch * len(dataloader) + batch_num
                    writer.add_scalar('Pretraining/Loss', loss_accum, niter)
            batch_num = batch_num + 1

            if batch_num in [len(dataloader), len(dataloader)//2, len(dataloader)//4, 3*len(dataloader)//4]:
                inp = utils.tensor2img(inputs)
                out = utils.tensor2img(outputs)
                if board:
                    img = np.concatenate((inp, out), axis=1)
                    writer.add_image('Pretraining/Epoch_' + str(epoch + 1).zfill(3) + '/Sample_' + str(img_counter).zfill(2), img)
                    img_counter += 1

        scheduler.step()

        epoch_loss = running_loss / dataset_size
        if epoch == 0: first_loss = epoch_loss
        if epoch == 4 and epoch_loss / first_loss > 1:
            utils.print_both(txt_file, "\nLoss not converging, starting pretraining again\n")
            return False

        if board:
            writer.add_scalar('Pretraining/Loss' + '/Epoch', epoch_loss, epoch + 1)

        utils.print_both(txt_file, 'Pretraining:\t Loss: {:.4f}'.format(epoch_loss))

        # If wanted to add some criterium in the future
        if epoch_loss < best_loss or epoch_loss >= best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        utils.print_both(txt_file, '')

    time_elapsed = time.time() - since
    utils.print_both(txt_file, 'Pretraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    model.pretrained = True
    torch.save(model.state_dict(), pretrained)

    return model


def embedded_outputs(model, dataloader, params, threshold=50000):
    output_array = None
    model.eval()
    # Itarate throught the data and concatenate the latent space representations of images
    for data in dataloader:
        inputs, _ = data
        inputs = inputs.to(params['device'])
        _, _, outputs = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
        else:
            output_array = outputs.cpu().detach().numpy()
        # print(output_array.shape)
        if threshold is not None and output_array.shape[0] > threshold: break
    return output_array

# K-means clusters initialisation
def kmeans(model, dataloader, params):
    km = KMeans(n_clusters=model.num_clusters, n_init=20)
    # Perform K-means
    km.fit_predict(embedded_outputs(model, dataloader, params))
    # Update clustering layer weights
    weights = torch.from_numpy(km.cluster_centers_)
    model.clustering.set_weight(weights.to(params['device']))
    # torch.cuda.empty_cache()

def gmm(model, dataloader, params):
    gm = GaussianMixture(n_components=model.num_clusters, covariance_type=params['gmm_covariance_type'],
                         tol=params['gmm_tol'], max_iter=params['gmm_max_iter'])
    gm.fit_predict(embedded_outputs(model, dataloader, params))
    weights = torch.from_numpy(gm.means_)
    model.clustering.set_weight(weights.to(params['device']))

def can_use_label_img(inputs, data_loader):
    #if (inputs.size()[2] != inputs.size()[3]):
    #    return False # label images must be square
    # sqrt(num_images)*width must be <= 8192 according to tensorboardX documentation
    return math.sqrt(len(data_loader)*inputs.size()[2]*inputs.size()[2]) <= 8192

# Function forwarding data through network, collecting clustering weight output and returning prediciotns and labels
def calculate_predictions(model, dataloader, params):
    output_array = None
    label_array = None
    embedding_array = None
    label_img = None
    use_label_img = params['save_embedding_inputs']
    model.eval()
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(params['device'])
        labels = labels.to(params['device'])
        _, outputs, embedding = model(inputs)
        if output_array is not None:
            output_array = np.concatenate((output_array, outputs.cpu().detach().numpy()), 0)
            label_array = np.concatenate((label_array, labels.cpu().detach().numpy()), 0)
            embedding_array = np.concatenate((embedding_array, embedding.cpu().detach().numpy()), 0)
            if use_label_img and can_use_label_img(inputs, dataloader):
                label_img = np.concatenate((label_img, inputs.cpu().detach().numpy()), 0)
                print(f'label_img.shape={label_img.shape}')
        else:
            output_array = outputs.cpu().detach().numpy()
            label_array = labels.cpu().detach().numpy()
            embedding_array = embedding.cpu().detach().numpy()
            if use_label_img and can_use_label_img(inputs, dataloader):
                label_img = inputs.cpu().detach().numpy()
                print(f'label_img.shape={label_img.shape}')
    preds = np.argmax(output_array.data, axis=1)
    if (label_img is not None and label_img.shape[2] != label_img.shape[3]):
        # pad with zeros so label_img is square (align to corner)
        label_img_size = max(label_img.shape[2], label_img.shape[3])
        label_img_square = np.zeros((label_img.shape[0], label_img.shape[1], label_img_size, label_img_size))
        label_img_square[:, :, :label_img.shape[2], :label_img.shape[3]] = label_img
        label_img = label_img_square
    return output_array, label_array, preds, embedding_array, label_img

# Calculate target distribution
def target(out_distr):
    tar_dist = out_distr ** 2 / np.sum(out_distr, axis=0)
    tar_dist = np.transpose(np.transpose(tar_dist) / np.sum(tar_dist, axis=1))
    return tar_dist

def init_clusters(txt_file, model, dl, params):
    if params['cluster_init_method'] == 'gmm':
        utils.print_both(txt_file, '\nInitializing cluster centers based on GMM')
        gmm(model, copy.deepcopy(dl), params)
    elif params['cluster_init_method'] == 'kmeans':
        utils.print_both(txt_file, '\nInitializing cluster centers based on K-means')
        kmeans(model, copy.deepcopy(dl), params)
    else:
        raise Exception('Unrecognized cluster_init_method: {}'.format(params['cluster_init_method']))
