import numpy as np
import sklearn.metrics
import torch
from torchvision import transforms

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
inv_normalize = transforms.Normalize(
    mean=[-0.485 / .229, -0.456 / 0.224, -0.406 / 0.255],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
)


# Simple tensor to image translation
def tensor2img(tensor):
    img = tensor.cpu().data[0]
    img = torch.clamp(img, 0, 1)
    return img


# Define printing to console and file
def print_both(f, text):
    print(text)
    f.write(text + '\n')


# Metrics class was copied from DCEC article authors repository (link in README)
class metrics:
    nmi = sklearn.metrics.normalized_mutual_info_score
    ari = sklearn.metrics.adjusted_rand_score

    @staticmethod
    def acc(labels_true, labels_pred):
        labels_true = labels_true.astype(np.int64)
        assert labels_pred.size == labels_true.size
        D = max(labels_pred.max(), labels_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(labels_pred.size):
            w[labels_pred[i], labels_true[i]] += 1
        from scipy.optimize import linear_sum_assignment as linear_assignment
        row_ind, col_ind = linear_assignment(w.max() - w)
        return sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))]) * 1.0 / labels_pred.size

def load_pretrained_net(model, pretrained_net_path):
    try:
        model.load_state_dict(torch.load(pretrained_net_path))
    except:
        print('Unable to load {} into model.')
        print(('This can be caused by a mismatch in the expected input' +
              ' shape between models. Note that in this case the' + 
              ' expected input shape is {}.').format(model.input_shape))
        raise

def matrix_metrics(matrix, num_clusters, preds, labels, worker=None, num_images=None):
    """Compute pairwise metrics for pairs of images in each cluster, and pairs
    containing one image in and one image out of a cluster. (e.g. SSIM, MSE)

    Note that self-pairs are not included."""
    total_sum_in = 0
    total_num_pairs_in = 0
    total_sum_out = 0
    total_num_pairs_out = 0
    
    result = {'all_clusters': {}, 'cluster': []}
    # x and y are used to select pairs from the matrix. Each element of
    # x and y is a mask representing the images in (in the case of x) or out
    # (in the case of y) of a cluster. These masks are 2-dimensional so that
    # the transpose operation can be used.
    #
    if num_images is None:
        num_images = len(preds)

    # masks out self-pairs -- including these would increase the
    # average ssim for in-cluster pairs.
    self_pair_mask = torch.tensor(1 - np.identity(num_images))

    # this will be 1 for each element predicted in class, 0 otherwise
    x = np.zeros((num_clusters, 1, num_images))
    # this will be 0 for each element predicted in class, 1 otherwise
    y = np.ones((num_clusters, 1, num_images))

    x = torch.tensor(x)
    y = torch.tensor(y)
    matrix = torch.tensor(matrix)

    if worker is not None:
        self_pair_mask = self_pair_mask.send(worker)
        x = x.send(worker)
        y = y.send(worker)
        matrix = matrix.send(worker)

    for i in range(len(preds)):
        predicted_class_index = preds[i]
        x[predicted_class_index][0][labels[i]] = 1
        y[predicted_class_index][0][labels[i]] = 0
    for i in range(num_clusters):
        # select in-cluster pairs
        pairs_in_mask = x[i] * x[i].transpose(0, 1) * self_pair_mask
        pairs_in = matrix * pairs_in_mask
        num_pairs_in = sum(sum(pairs_in_mask > 0))
        # select pairs with one image in the cluster and one image not in the cluster
        pairs_out_mask = x[i] * y[i].transpose(0, 1) + x[i].transpose(0, 1) * y[i]
        pairs_out = matrix * pairs_out_mask
        num_pairs_out = sum(sum(pairs_out_mask > 0))

        sum_in = sum(sum(pairs_in))
        sum_out = sum(sum(pairs_out))
        avg_in = sum_in/num_pairs_in
        avg_out = sum_out/num_pairs_out

        total_sum_in += sum_in
        total_sum_out += sum_out
        total_num_pairs_in += num_pairs_in
        total_num_pairs_out += num_pairs_out
        result['cluster'].append({'average_in_cluster': avg_in, 'average_out_cluster': avg_out,
            'sum_in': sum_in, 'sum_out': sum_out, 'num_pairs_in': num_pairs_in, 'num_pairs_out': num_pairs_out})
    if worker is not None:
        total_sum_in = total_sum_in.get()
        total_sum_out = total_sum_out.get()
        total_num_pairs_in = total_num_pairs_in.get()
        total_num_pairs_out = total_num_pairs_out.get()

    total_avg_in = total_sum_in / total_num_pairs_in
    total_avg_out = total_sum_out / total_num_pairs_out
    result['all_clusters'] = {'average_in_cluster': float(total_avg_in), 'average_out_cluster': float(total_avg_out),
            'total_sum_in': total_sum_in, 'total_sum_out': total_sum_out,
            'total_num_pairs_in': total_num_pairs_in, 'total_num_pairs_out': total_num_pairs_out}
    return result

def log_matrix_metrics(log_func, metrics, metric_name):
    for i in range(len(metrics['cluster'])):
        avg_in = float(metrics['cluster'][i]['average_in_cluster'])
        avg_out = float(metrics['cluster'][i]['average_out_cluster'])
        log_func(f'Cluster {i}: Average {metric_name} (in cluster): {avg_in}')
        log_func(f'Cluster {i}: Average {metric_name} (out cluster): {avg_out}')
    total_avg_in = metrics['all_clusters']['average_in_cluster']
    total_avg_out = metrics['all_clusters']['average_out_cluster']
    log_func(f'{metric_name} (in cluster): {total_avg_in}')
    log_func(f'{metric_name} (out cluster): {total_avg_out}')

def load_ssim_matrix(filename):
    return load_diag_matrix(filename, 1)

def load_mse_matrix(filename):
    return load_diag_matrix(filename, 0)

def load_diag_matrix(filename, identity_value):
    matrix_fd = open(filename, 'r')
    matrix_lines = matrix_fd.read().strip().split('\n')
    file_count = len(matrix_lines) + 1
    matrix = np.zeros((file_count, file_count))
    for i in range(file_count - 1):
        entries = matrix_lines[i].rstrip(',').split(',')
        for j in range(len(entries)):
            matrix[i][i+j+1] = float(entries[j])
            matrix[i+j+1][i] = float(entries[j])
        matrix[i][i] = identity_value
    matrix_fd.close()
    return matrix
