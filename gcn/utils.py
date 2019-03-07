import numpy as np
import pickle as pkl
import networkx as nx
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

'''
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
'''

def load_data(dataset_str, device):
    """
    Loads input data from gcn/data directory
    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.
    All objects above must be saved using python pickle module.
    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    num_classes = y.shape[1]

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.from_numpy(features.todense()).to(device)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = torch.from_numpy(adj.todense()).to_sparse().to(device)

    labels = torch.tensor(np.vstack((ally, ty)).argmax(axis=1)).to(device)
    labels[test_idx_reorder] = labels[test_idx_range]

    idx_test = torch.from_numpy(test_idx_range).to(device)
    idx_train = torch.arange(len(y)).to(device)
    idx_val = torch.arange(len(y), len(y)+500).to(device)

    return features, adj, labels, num_classes, idx_train, idx_val, idx_test
'''

def to_sparse_tensor(x):
    x = x.tocoo()
    i = torch.tensor(np.vstack((x.row, x.col)), dtype=torch.long)
    v = torch.tensor(x.data, dtype=torch.float)
    return torch.sparse_coo_tensor(i, v, torch.Size(x.shape))

def read_file(dataset, name):
    filename = 'data/ind.' + dataset + '.{}'.format(name)
    if name == 'test.index':
        return np.loadtxt(filename, dtype=np.long)
    else:
        with open(filename, 'rb') as f:
            return pkl.load(f, encoding='latin1')

def normalize(x):
    return scipy.sparse.diags(np.array(x.sum(1)).flatten() ** -1).dot(x)

def load_data(dataset, device):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph', 'test.index']
    x, y, tx, ty, allx, ally, graph, test_index = [read_file(dataset, name) for name in names]
    num_classes = y.shape[1]
    train_index = torch.arange(y.shape[0]).to(device)
    dev_index = torch.arange(y.shape[0], y.shape[0] + 500).to(device)
    test_index_sorted = torch.tensor(np.sort(test_index)).to(device)
    test_index = torch.tensor(test_index).to(device)

    x = torch.tensor(normalize(scipy.sparse.vstack([allx, tx])).todense()).to(device)
    y = torch.tensor(np.vstack([ally, ty]).argmax(axis=1)).to(device)

    x[test_index] = x[test_index_sorted]
    y[test_index] = y[test_index_sorted]

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    a = to_sparse_tensor(normalize(adj + scipy.sparse.eye(adj.shape[0]))).to(device)

    return x, a, y, num_classes, train_index, dev_index, test_index
'''
