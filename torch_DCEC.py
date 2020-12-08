from __future__ import print_function, division
import numpy as np

class ToRGB:
    def __call__(self, tensor):
        if tensor.shape[0] != 1:
            raise Exception('ToRGB expects a single-channel input.')
        return torch.cat((tensor, tensor, tensor))

def load_ssim_matrix(filename):
    matrix_fd = open(filename, 'r')
    ssim_matrix_lines = matrix_fd.read().strip().split('\n')
    file_count = len(ssim_matrix_lines) + 1
    ssim_matrix = np.zeros((file_count, file_count))
    for i in range(file_count - 1):
        entries = ssim_matrix_lines[i].rstrip(',').split(',')
        for j in range(len(entries)):
            ssim_matrix[i][i+j+1] = float(entries[j])
            ssim_matrix[i+j+1][i] = float(entries[j])
        ssim_matrix[i][i] = 1
    matrix_fd.close()
    return ssim_matrix


if __name__ == "__main__":

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from torchvision import datasets, models, transforms
    import os
    import math
    import fnmatch
    import nets
    import utils
    import training_functions
    import PIL
    from tensorboardX import SummaryWriter

    # Translate string entries to bool for parser
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument('--mode', default='train_full', choices=['train_full', 'pretrain', 'init_clusters'], help='mode')
    parser.add_argument('--tensorboard', default=True, type=bool, help='export training stats to tensorboard')
    parser.add_argument('--pretrain', default=True, type=str2bool, help='perform autoencoder pretraining')
    parser.add_argument('--pretrained_net', default=1, help='index or path of pretrained net')
    parser.add_argument('--net_architecture', default='CAE_3', choices=['CAE_3', 'CAE_bn3', 'CAE_4', 'CAE_bn4', 'CAE_5', 'CAE_bn5'], help='network architecture used')
    parser.add_argument('--dataset', default='MNIST-train',
                        choices=['MNIST-train', 'custom', 'MNIST-test', 'MNIST-full', 'USPS-train', 'USPS-test'],
                        help='custom or prepared dataset')
    parser.add_argument('--dataset_path', default='data', help='path to dataset')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate for clustering')
    parser.add_argument('--rate_pretrain', default=0.001, type=float, help='learning rate for pretraining')
    parser.add_argument('--weight', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--weight_pretrain', default=0.0, type=float, help='weight decay for clustering')
    parser.add_argument('--sched_step', default=200, type=int, help='scheduler steps for rate update')
    parser.add_argument('--sched_step_pretrain', default=200, type=int,
                        help='scheduler steps for rate update - pretrain')
    parser.add_argument('--sched_gamma', default=0.1, type=float, help='scheduler gamma for rate update')
    parser.add_argument('--sched_gamma_pretrain', default=0.1, type=float,
                        help='scheduler gamma for rate update - pretrain')
    parser.add_argument('--epochs', default=1000, type=int, help='clustering epochs')
    parser.add_argument('--epochs_pretrain', default=300, type=int, help='pretraining epochs')
    parser.add_argument('--printing_frequency', default=10, type=int, help='training stats printing frequency')
    parser.add_argument('--gamma', default=0.1, type=float, help='clustering loss weight')
    parser.add_argument('--update_interval', default=80, type=int, help='update interval for target distribution')
    parser.add_argument('--tol', default=1e-2, type=float, help='stop criterium tolerance')
    parser.add_argument('--num_clusters', default=10, type=int, help='number of clusters')
    parser.add_argument('--custom_img_size', nargs=3, type=int, help='size of custom images')
    parser.add_argument('--leaky', default=True, type=str2bool)
    parser.add_argument('--neg_slope', default=0.01, type=float)
    parser.add_argument('--activations', default=False, type=str2bool)
    parser.add_argument('--bias', default=True, type=str2bool)
    parser.add_argument('--device', default='', type=str, help='Device to perform computations on. Expects "cpu" or "cuda:n" where n is a zero-indexed gpu identifier.')
    parser.add_argument('--output_name', default='', type=str, help='The name to use for the output files.')
    parser.add_argument('--output_dir', default='', type=str, help='The directory in which to save the output. The directories runs, reports, and nets will be created here if they do not already exist.')
    parser.add_argument('--save_embedding_inputs', default=True, type=str2bool, help='Save input images in tensorboard embeddings.')
    parser.add_argument('--save_embedding_interval', default=1, type=int, help='How frequently to save the tensorboard embeddings. Embeddings will be saved every SAVE_EMBEDDING_INTERVAL epochs, before the first epoch, and after the last epoch.')
    parser.add_argument('--rgb_mnist', default=False, type=str2bool, help='Use three channels (RGB) for MNIST dataset instead of one.')
    parser.add_argument('--cluster_init_method', default='kmeans', choices=['kmeans', 'gmm'], help='Which clustering method to use to initialize the cluster centers at the beginning of the full training step.')
    parser.add_argument('--gmm_covariance_type', default='full', choices=['full', 'tied', 'diag', 'spherical'])
    parser.add_argument('--gmm_tol', default=1e-3, type=float)
    parser.add_argument('--gmm_max_iter', default=100, type=int)
    parser.add_argument('--train_init_clusters', default=False, type=str2bool, help='Initialize cluster centers at beginning of full training stage.')
    parser.add_argument('--usps_location', default='', type=str, help='The location in which usps_train.jf and usps_test.jf can be found.')
    parser.add_argument('--zero_gamma_epochs', default=0, type=int, help='Sets gamma to zero for the first N epochs during the training stage.')
    parser.add_argument('--l2_norm', default=True, type=str2bool, help='Enables l2-normalization before the embedding layer in the model.')
    parser.add_argument('--ssim_matrix', default='', type=str, help='SSIM Matrix file -- skip class-dependent metrics & use ssim instead')
    args = parser.parse_args()
    print(args)

    if args.mode == 'pretrain' and not args.pretrain:
        print("Nothing to do :(")
        exit()


    board = args.tensorboard

    # Deal with pretraining option and way of showing network path
    pretrain = args.pretrain
    net_is_path = True
    if not pretrain:
        try:
            int(args.pretrained_net)
            idx = args.pretrained_net
            net_is_path = False
        except:
            pass
    params = {'pretrain': pretrain}
    params['save_embedding_inputs'] = args.save_embedding_inputs
    params['embedding_interval'] = args.save_embedding_interval
    params['gmm_covariance_type'] = args.gmm_covariance_type
    params['gmm_tol'] = args.gmm_tol
    params['gmm_max_iter'] = args.gmm_max_iter
    params['cluster_init_method'] = args.cluster_init_method
    params['train_init_clusters'] = args.train_init_clusters
    params['zero_gamma_epochs'] = args.zero_gamma_epochs

    # Directories
    # Create directories structure
    dirs = ['runs', 'reports', 'nets']
    if (args.output_dir != ''):
        dirs = [os.path.join(args.output_dir, x) for x in dirs]
    list(map(lambda x: os.makedirs(x, exist_ok=True), dirs))

    # Net architecture
    model_name = args.net_architecture
    # Indexing (for automated reports saving) - allows to run many trainings and get all the reports collected
    if pretrain or (not pretrain and net_is_path):
        reports_list = sorted(os.listdir(os.path.join(args.output_dir, 'reports')), reverse=True)
        if reports_list:
            for file in reports_list:
                # print(file)
                try:
                    if fnmatch.fnmatch(file, model_name + '*'):
                        idx = int(str(file)[-7:-4]) + 1
                        break
                except:
                    pass
        try:
            idx
        except NameError:
            idx = 1

    # Base filename
    name = model_name + '_' + str(idx).zfill(3)
    if (args.output_name != ''):
        name = args.output_name

    # Filenames for report and weights
    name_txt = name + '.txt'
    name_net = name + '.pt'
    pretrained = name + '_pretrained.pt'

    # Arrange filenames for report, network weights, pretrained network weights
    name_txt = os.path.join('reports', name_txt)
    name_net = os.path.join('nets', name_net)
    if (args.output_dir != ''):
        name_txt = os.path.join(args.output_dir, name_txt)
        name_net = os.path.join(args.output_dir, name_net)
    if (os.path.exists(name_txt)):
        raise Exception('Output file {} already exists.'.format(name_txt))
    if (os.path.exists(name_net)):
        raise Exception('Output file {} already exists.'.format(name_net))
    if net_is_path and not pretrain:
        pretrained = args.pretrained_net
    else:
        pretrained = os.path.join('nets', pretrained)
        if (args.output_dir != ''):
            pretrained = os.path.join(args.output_dir, pretrained)
        if (os.path.exists(pretrained)):
            raise Exception('Output file {} already exists.'.format(pretrained))
    if not pretrain and not os.path.isfile(pretrained):
        print("No pretrained weights, try again choosing pretrained network or create new with pretrain=True")

    model_files = [name_net, pretrained]
    params['model_files'] = model_files

    # Open file
    if pretrain:
        f = open(name_txt, 'w')
    else:
        f = open(name_txt, 'a')
    params['txt_file'] = f

    tensorboard_output = os.path.join('runs', name)
    if (args.output_dir != ''):
        tensorboard_output = os.path.join(args.output_dir, 'runs', name)
    if os.path.exists(tensorboard_output):
        raise Exception('Output file {} already exists.'.format(tensorboard_output))
    # Initialize tensorboard writer
    if board:
        writer = SummaryWriter(tensorboard_output)
        params['writer'] = writer
    else:
        params['writer'] = None

    # Hyperparameters

    # Used dataset
    dataset = args.dataset

    # Batch size
    batch = args.batch_size
    params['batch'] = batch
    # Number of workers (typically 4*num_of_GPUs)
    workers = 4
    # Learning rate
    rate = args.rate
    rate_pretrain = args.rate_pretrain
    # Adam params
    # Weight decay
    weight = args.weight
    weight_pretrain = args.weight_pretrain
    # Scheduler steps for rate update
    sched_step = args.sched_step
    sched_step_pretrain = args.sched_step_pretrain
    # Scheduler gamma - multiplier for learning rate
    sched_gamma = args.sched_gamma
    sched_gamma_pretrain = args.sched_gamma_pretrain

    # Number of epochs
    epochs = args.epochs
    pretrain_epochs = args.epochs_pretrain
    params['pretrain_epochs'] = pretrain_epochs

    # Printing frequency
    print_freq = args.printing_frequency
    params['print_freq'] = print_freq

    # Clustering loss weight:
    gamma = args.gamma
    params['gamma'] = gamma

    # Update interval for target distribution:
    update_interval = args.update_interval
    params['update_interval'] = update_interval

    # Tolerance for label changes:
    tol = args.tol
    params['tol'] = tol

    # Number of clusters
    num_clusters = args.num_clusters

    # Report for settings
    tmp = "Training the '" + model_name + "' architecture"
    utils.print_both(f, tmp)
    tmp = "\n" + "The following parameters are used:"
    utils.print_both(f, tmp)
    tmp = "Batch size:\t" + str(batch)
    utils.print_both(f, tmp)
    tmp = "Number of workers:\t" + str(workers)
    utils.print_both(f, tmp)
    tmp = "Learning rate:\t" + str(rate)
    utils.print_both(f, tmp)
    tmp = "Pretraining learning rate:\t" + str(rate_pretrain)
    utils.print_both(f, tmp)
    tmp = "Weight decay:\t" + str(weight)
    utils.print_both(f, tmp)
    tmp = "Pretraining weight decay:\t" + str(weight_pretrain)
    utils.print_both(f, tmp)
    tmp = "Scheduler steps:\t" + str(sched_step)
    utils.print_both(f, tmp)
    tmp = "Scheduler gamma:\t" + str(sched_gamma)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler steps:\t" + str(sched_step_pretrain)
    utils.print_both(f, tmp)
    tmp = "Pretraining scheduler gamma:\t" + str(sched_gamma_pretrain)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of training:\t" + str(epochs)
    utils.print_both(f, tmp)
    tmp = "Number of epochs of pretraining:\t" + str(pretrain_epochs)
    utils.print_both(f, tmp)
    tmp = "Clustering loss weight:\t" + str(gamma)
    utils.print_both(f, tmp)
    tmp = "Update interval for target distribution:\t" + str(update_interval)
    utils.print_both(f, tmp)
    tmp = "Stop criterium tolerance:\t" + str(tol)
    utils.print_both(f, tmp)
    tmp = "Number of clusters:\t" + str(num_clusters)
    utils.print_both(f, tmp)
    tmp = "Leaky relu:\t" + str(args.leaky)
    utils.print_both(f, tmp)
    tmp = "Leaky slope:\t" + str(args.neg_slope)
    utils.print_both(f, tmp)
    tmp = "Activations:\t" + str(args.activations)
    utils.print_both(f, tmp)
    tmp = "Bias:\t" + str(args.bias)
    utils.print_both(f, tmp)

    utils.print_both(f, '\nargs: ' + str(args)+'\n')


    # MNIST-train, MNIST-test, MNIST-full use slightly modified torchvision MNIST class
    import mnist
    import usps

    dataset_loaders = {}

    mnist_transforms = [transforms.ToTensor()]
    if args.rgb_mnist:
        mnist_transforms.append(ToRGB())
    mnist_transforms = transforms.Compose(mnist_transforms)

    dataset_loaders['MNIST-train'] = lambda: mnist.MNIST('../data', train=True, download=True,
                                                         transform=mnist_transforms)

    dataset_loaders['MNIST-test'] = lambda: mnist.MNIST('../data', train=False, download=True,
                                                        transform=mnist_transforms)

    dataset_loaders['MNIST-full'] = lambda: mnist.MNIST('../data', full=True, download=True,
                                                        transform=mnist_transforms)
    dataset_loaders['USPS-train'] = lambda: usps.USPS(args.usps_location, 'train',
                                                        transform=mnist_transforms)
    dataset_loaders['USPS-test'] = lambda: usps.USPS(args.usps_location, 'test',
                                                        transform=mnist_transforms)

    # Data preparation
    if dataset in dataset_loaders:
        tmp = "\nData preparation\nReading data from: {} dataset".format(dataset)
        utils.print_both(f, tmp)
        dataset = dataset_loaders[dataset]()
        img_size = dataset.image_size
        if (args.rgb_mnist):
            img_size[2] = 3
        tmp = "Image size used:\t{0}x{1}x{2}".format(img_size[0], img_size[1], img_size[2])
        utils.print_both(f, tmp)


        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch, shuffle=False, num_workers=workers)

        dataset_size = len(dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)
    else:
        # Data folder
        data_dir = args.dataset_path
        tmp = "\nData preparation\nReading data from:\t./" + data_dir
        utils.print_both(f, tmp)

        # Image size
        if args.custom_img_size is None or len(args.custom_img_size) != 3:
            raise Exception('--custom_img_size is required when --dataset custom or --verification_dataset custom is specified.')

        img_size = args.custom_img_size

        tmp = "Image size used:\t{0}x{1}".format(img_size[0], img_size[1])
        utils.print_both(f, tmp)

        # Transformations
        data_transforms = transforms.Compose([
                transforms.Resize(img_size[0:2]),
                transforms.ToTensor(),
            ])

        # ImageFolder's default loader converts the input to RGB for some reason
        def img_loader(path):
            with open(path, 'rb') as f:
                img = PIL.Image.open(f)
                # copy is required because PIL.Image.open uses lazy loading
                return img.copy()

        # Read data from selected folder and apply transformations
        image_dataset = datasets.ImageFolder(data_dir, data_transforms, loader=img_loader)
        # Prepare data for network: schuffle and arrange batches
        dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch,
                                                      shuffle=False, num_workers=workers)

        # Size of data sets
        dataset_size = len(image_dataset)
        tmp = "Training set size:\t" + str(dataset_size)
        utils.print_both(f, tmp)

    params['dataset_size'] = dataset_size

    # GPU check
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if (args.device != ''):
        if (args.device.startswith('cuda:')):
            if (int(args.device[5:]) >= torch.cuda.device_count()):
                raise Exception('Unable to use device: {}'.format(args.device))
        elif (args.device != 'cpu'):
            raise Exception('Unrecognized device: {}'.format(args.device))
        device = torch.device(args.device)

    tmp = "\nPerforming calculations on:\t" + str(device)
    utils.print_both(f, tmp + '\n')
    params['device'] = device

    params['num_clusters'] = num_clusters

    params['class_dependent_metrics'] = True
    params['use_ssim'] = False
    
    if args.ssim_matrix != '':
        params['use_ssim'] = True
        params['class_dependent_metrics'] = False
        params['ssim_matrix'] = load_ssim_matrix(args.ssim_matrix)

    # Evaluate the proper model
    to_eval = "nets." + model_name + "(img_size, num_clusters=num_clusters, leaky = args.leaky, neg_slope = args.neg_slope, l2_norm = args.l2_norm)"
    model = eval(to_eval)

    # Tensorboard model representation
    # if board:
    #     writer.add_graph(model, torch.autograd.Variable(torch.Tensor(batch, img_size[2], img_size[0], img_size[1])))

    model = model.to(device)
    # Reconstruction loss
    criterion_1 = nn.MSELoss(size_average=True)
    # Clustering loss
    criterion_2 = nn.KLDivLoss(size_average=False)

    criteria = [criterion_1, criterion_2]

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate, weight_decay=weight)

    optimizer_pretrain = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=rate_pretrain, weight_decay=weight_pretrain)

    optimizers = [optimizer, optimizer_pretrain]

    scheduler = lr_scheduler.StepLR(optimizer, step_size=sched_step, gamma=sched_gamma)
    scheduler_pretrain = lr_scheduler.StepLR(optimizer_pretrain, step_size=sched_step_pretrain, gamma=sched_gamma_pretrain)

    schedulers = [scheduler, scheduler_pretrain]

    utils.print_both(f, 'Mode: {}\n'.format(args.mode))
    
    if args.mode == 'train_full':
        model = training_functions.train_model(model, dataloader, criteria, optimizers, schedulers, epochs, params)
    elif args.mode == 'pretrain':
        model = training_functions.pretraining(model, dataloader, criteria[0], optimizers[1], schedulers[1], epochs, params)
        training_functions.init_clusters(f, model, dataloader, params)
    elif args.mode == 'init_clusters':
        utils.load_pretrained_net(model, args.pretrained_net)
        training_functions.init_clusters(f, model, dataloader, params)

    # Save final model
    torch.save(model.state_dict(), name_net)
    print('Saved model to {}'.format(name_net))

    # Close files
    f.close()
    if board:
        writer.close()
