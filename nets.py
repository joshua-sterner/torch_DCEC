import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# Clustering layer definition (see DCEC article for equations)
class ClusterlingLayer(nn.Module):
    def __init__(self, in_features=10, out_features=10, alpha=1.0):
        super(ClusterlingLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.weight = nn.Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight = nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weight
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

    def extra_repr(self):
        return 'in_features={}, out_features={}, alpha={}'.format(
            self.in_features, self.out_features, self.alpha
        )

    def set_weight(self, tensor):
        self.weight = nn.Parameter(tensor)

def to_vec2(n):
    if type(n) == int or type(n) == float:
        return (n, n)
    return n

def conv2d_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Returns a tuple (height, width) containing the width and height dimensions
       of the tensor resulting from a conv2d operation."""
    kernel_size = to_vec2(kernel_size)
    stride = to_vec2(stride)
    padding = to_vec2(padding)
    dilation = to_vec2(dilation)
    return tuple(math.floor((input_shape[i] + 2*padding[i] - dilation[i]*(kernel_size[i] - 1) - 1)/stride[i] + 1) for i in (0, 1))

def convTranspose2d_output_shape(input_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Returns a tuple (height, width) containing the width and height dimensions
       of the tensor resulting from a convTranspose2d operation, not including any
       output padding."""
    kernel_size = to_vec2(kernel_size)
    stride = to_vec2(stride)
    padding = to_vec2(padding)
    dilation = to_vec2(dilation)
    return tuple((input_shape[i] - 1)*stride[i] - 2*padding[i] + dilation[i]*(kernel_size[i] - 1) + 1 for  i in (0,1))

def deconv_output_padding(input_shape, output_shape, kernel_size, stride=1, padding=0, dilation=1):
    """Returns the output padding needed for a convTranspose2d operation
       to produce an output with the specified height and width."""
    output_shape_no_padding = convTranspose2d_output_shape(input_shape, kernel_size, stride, padding, dilation)
    return tuple(output_shape[i] - output_shape_no_padding[i] for i in (0,1))

# Convolutional autoencoder directly from DCEC article
class CAE_3(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_3, self).__init__()
        self.l2_norm = l2_norm
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 3, stride=2, padding=0)
        self.conv3_shape = conv3_shape
        lin_features_len = conv3_shape[0] * conv3_shape[1] * filters[2]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 3, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.shape[0], self.filters[2], self.conv3_shape[0], self.conv3_shape[1])
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out


# Convolutional autoencoder from DCEC article with Batch Norms and Leaky ReLUs
class CAE_bn3(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_bn3, self).__init__()
        self.l2_norm = l2_norm
        self.activations=activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride=2, padding=0, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 3, stride=2, padding=0)
        self.conv3_shape = conv3_shape
        lin_features_len = conv3_shape[0] * conv3_shape[1] * filters[2]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 3, stride=2, padding=0)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu3_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu1_2(x)
        x = x.view(x.size(0), self.filters[2], self.conv3_shape[0], self.conv3_shape[1])
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu3_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out


# Convolutional autoencoder with 4 convolutional blocks
class CAE_4(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128, 256], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_4, self).__init__()
        self.l2_norm = l2_norm
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=0, bias=bias)
        conv4_shape = conv2d_output_shape(conv3_shape, 3, stride=2, padding=0)
        self.conv4_shape = conv4_shape

        lin_features_len = conv4_shape[0] * conv4_shape[1] * filters[3]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv4_shape, conv3_shape, 3, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], self.conv3_shape[0], self.conv3_shape[1])
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out

# Convolutional autoencoder with 4 convolutional blocks (BN version)
class CAE_bn4(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128, 256], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_bn4, self).__init__()
        self.l2_norm = l2_norm
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 5, stride=2, padding=2)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 3, stride=2, padding=0, bias=bias)
        conv4_shape = conv2d_output_shape(conv3_shape, 3, stride=2, padding=0)
        self.conv4_shape = conv4_shape

        lin_features_len = conv4_shape[0] * conv4_shape[1] * filters[3]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv4_shape, conv3_shape, 3, stride=2, padding=0)
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu4_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[3], self.conv4_shape[0], self.conv4_shape[1])
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out


# Convolutional autoencoder with 5 convolutional blocks
class CAE_5(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_5, self).__init__()
        self.l2_norm=True
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        conv4_shape = conv2d_output_shape(conv3_shape, 5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(filters[3], filters[4], 3, stride=2, padding=0, bias=bias)
        conv5_shape = conv2d_output_shape(conv4_shape, 3, stride=2, padding=0)
        self.conv5_shape = conv5_shape

        lin_features_len = conv5_shape[0] * conv5_shape[1] * filters[4]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv5_shape, conv4_shape, 3, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv4_shape, conv3_shape, 5, stride=2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu4_2(x)
        x = x.view(x.size(0), self.filters[4], self.conv5_shape[0], self.conv5_shape[1]) 
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out


# Convolutional autoencoder with 5 convolutional blocks (BN version)
class CAE_bn5(nn.Module):
    def __init__(self, input_shape=[128,128,3], num_clusters=10, filters=[32, 64, 128, 256, 512], leaky=True, neg_slope=0.01, activations=False, bias=True, l2_norm=True):
        super(CAE_bn5, self).__init__()
        self.l2_norm = l2_norm
        self.activations = activations
        self.pretrained = False
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.relu = nn.ReLU(inplace=False)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)

        self.conv1 = nn.Conv2d(input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        conv1_shape = conv2d_output_shape(input_shape, 5, stride=2, padding=2)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        conv2_shape = conv2d_output_shape(conv1_shape, 5, stride=2, padding=2)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.conv3 = nn.Conv2d(filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        conv3_shape = conv2d_output_shape(conv2_shape, 5, stride=2, padding=2)
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.conv4 = nn.Conv2d(filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        conv4_shape = conv2d_output_shape(conv3_shape, 5, stride=2, padding=2)
        self.bn4_1 = nn.BatchNorm2d(filters[3])
        self.conv5 = nn.Conv2d(filters[3], filters[4], 3, stride=2, padding=0, bias=bias)
        conv5_shape = conv2d_output_shape(conv4_shape, 3, stride=2, padding=0)
        self.conv5_shape = conv5_shape

        lin_features_len = conv5_shape[0] * conv5_shape[1] * filters[4]
        self.embedding = nn.Linear(lin_features_len, num_clusters, bias=bias)
        self.deembedding = nn.Linear(num_clusters, lin_features_len, bias=bias)
        out_pad = deconv_output_padding(conv5_shape, conv4_shape, 3, stride=2, padding=0)
        self.deconv5 = nn.ConvTranspose2d(filters[4], filters[3], 3, stride=2, padding=0, output_padding=out_pad,
                                          bias=bias)
        self.bn5_2 = nn.BatchNorm2d(filters[3])
        out_pad = deconv_output_padding(conv4_shape, conv3_shape, 5, stride=2, padding=2)
        self.deconv4 = nn.ConvTranspose2d(filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn4_2 = nn.BatchNorm2d(filters[2])
        out_pad = deconv_output_padding(conv3_shape, conv2_shape, 5, stride=2, padding=2)
        self.deconv3 = nn.ConvTranspose2d(filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn3_2 = nn.BatchNorm2d(filters[1])
        out_pad = deconv_output_padding(conv2_shape, conv1_shape, 5, stride=2, padding=2)
        self.deconv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.bn2_2 = nn.BatchNorm2d(filters[0])
        out_pad = deconv_output_padding(conv1_shape, input_shape, 5, stride=2, padding=2)
        self.deconv1 = nn.ConvTranspose2d(filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad,
                                          bias=bias)
        self.clustering = ClusterlingLayer(num_clusters, num_clusters)
        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        x = x.view(x.shape[0], -1)
        if self.l2_norm:
            x = F.normalize(x, p=2, dim=1)
        x = self.embedding(x)
        extra_out = x
        clustering_out = self.clustering(x)
        x = self.deembedding(x)
        x = self.relu5_2(x)
        x = x.view(x.size(0), self.filters[4], self.conv5_shape[0], self.conv5_shape[1])
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        return x, clustering_out, extra_out
