import networkx as nx
import numpy as np
import scipy
import pickle
import json
from torch.utils.data import DataLoader, Dataset
import dgl
import torch

def read_adjlist(fp):
    in_file = open(fp, 'r')
    adjlist = [line.strip() for line in in_file]
    in_file.close()
    return adjlist

def read_pickle(fp):
    in_file = open(fp, 'rb')
    idx = pickle.load(in_file)
    in_file.close()
    return idx

def read_json(fp):
    in_file = open(fp, 'r')
    idx = json.load(in_file)
    in_file.close()
    return idx

def fold_train_test_idx(pos_folds, neg_folds, nFold, foldID):
    train_pos_idx = []
    train_neg_idx = []
    test_fold_idx = []
    for fold in range(nFold):
        if fold == foldID:
            continue
        train_pos_idx.append(pos_folds['fold_' + str(fold)])
        train_neg_idx.append(neg_folds['fold_' + str(fold)])
    train_pos_idx = np.concatenate(train_pos_idx, axis=1)
    train_neg_idx = np.concatenate(train_neg_idx, axis=1)
    train_fold_idx = np.concatenate([train_pos_idx, train_neg_idx], axis=1)

    test_fold_idx.append(pos_folds['fold_' + str(foldID)])
    test_fold_idx.append(neg_folds['fold_' + str(foldID)])
    test_fold_idx = np.concatenate(test_fold_idx, axis=1)
    return train_fold_idx.T, test_fold_idx.T

def load_data(args, fold, rp, neg_times=1, train_test='train'):
    # Drug related
    if train_test == 'train':
        prefix0 = args.data_dir + '/processed/repeat{}/0/{}/fold_{}/'.format(rp, train_test, fold)
    else:
        prefix0 = args.data_dir + '/processed/repeat{}/0/{}/neg_times_{}/fold_{}/'.format(rp, train_test, neg_times, fold)

    adjlist00 = read_pickle(prefix0 + '/0-0.adjlist.pkl')
    adjlist01 = read_pickle(prefix0 + '/0-1-0.adjlist.pkl')
    adjlist02 = read_pickle(prefix0 + '/0-2-0.adjlist.pkl')
    adjlist03 = read_pickle(prefix0 + '/0-3-0.adjlist.pkl')
    adjlist04 = read_pickle(prefix0 + '/0-1-1-0.adjlist.pkl')
    # adjlist05 = read_pickle(prefix0 + '/0-1-0-1-0.adjlist.pkl')
    # adjlist06 = read_pickle(prefix0 + '/0-2-0-2-0.adjlist.pkl')
    # adjlist07 = read_pickle(prefix0 + '/0-3-0-3-0.adjlist.pkl')
    # adjlist08 = read_pickle(prefix0 + '/0-1-2-1-0.adjlist.pkl')
    # adjlist09 = read_pickle(prefix0 + '/0-2-1-2-0.adjlist.pkl')

    idx00 = read_pickle(prefix0 + '/0-0.idx.pkl')
    idx01 = read_pickle(prefix0 + '/0-1-0.idx.pkl')
    idx02 = read_pickle(prefix0 + '/0-2-0.idx.pkl')
    idx03 = read_pickle(prefix0 + '/0-3-0.idx.pkl')
    idx04 = read_pickle(prefix0 + '/0-1-1-0.idx.pkl')
    # idx05 = read_pickle(prefix0 + '/0-1-0-1-0.idx.pkl')
    # idx06 = read_pickle(prefix0 + '/0-2-0-2-0.idx.pkl')
    # idx07 = read_pickle(prefix0 + '/0-3-0-3-0.idx.pkl')
    # idx08 = read_pickle(prefix0 + '/0-1-2-1-0.idx.pkl')
    # idx09 = read_pickle(prefix0 + '/0-2-1-2-0.idx.pkl')

    # Protein related
    if train_test == 'train':
        prefix1 = args.data_dir + '/processed/repeat{}/1/{}/fold_{}/'.format(rp, train_test, fold)
    else:
        prefix1 = args.data_dir + '/processed/repeat{}/1/{}/neg_times_{}/fold_{}/'.format(rp, train_test, neg_times, fold)
    adjlist10 = read_pickle(prefix1 + '/1-1.adjlist.pkl')
    adjlist11 = read_pickle(prefix1 + '/1-0-1.adjlist.pkl')
    adjlist12 = read_pickle(prefix1 + '/1-2-1.adjlist.pkl')
    adjlist13 = read_pickle(prefix1 + '/1-0-0-1.adjlist.pkl')
    # adjlist14 = read_pickle(prefix1 + '/1-0-1-0-1.adjlist.pkl')
    # adjlist15 = read_pickle(prefix1 + '/1-0-2-0-1.adjlist.pkl')
    # adjlist16 = read_pickle(prefix1 + '/1-2-0-2-1.adjlist.pkl')
    # adjlist17 = read_pickle(prefix1 + '/1-2-1-2-1.adjlist.pkl')

    idx10 = read_pickle(prefix1 + '/1-1.idx.pkl')
    idx11 = read_pickle(prefix1 + '/1-0-1.idx.pkl')
    idx12 = read_pickle(prefix1 + '/1-2-1.idx.pkl')
    idx13 = read_pickle(prefix1 + '/1-0-0-1.idx.pkl')
    # idx14 = read_pickle(prefix1 + '/1-0-1-0-1.idx.pkl')
    # idx15 = read_pickle(prefix1 + '/1-0-2-0-1.idx.pkl')
    # idx16 = read_pickle(prefix1 + '/1-2-0-2-1.idx.pkl')
    idx17 = read_pickle(prefix1 + '/1-2-1-2-1.idx.pkl')

    return [[adjlist00, adjlist01, adjlist02, adjlist03, adjlist04],
            [adjlist10, adjlist11, adjlist12, adjlist13]], \
           [[idx00, idx01, idx02, idx03, idx04],
            [idx10, idx11, idx12, idx13]]
    # return [[adjlist00, adjlist01, adjlist02, adjlist03, adjlist04, adjlist05, adjlist06, adjlist07, adjlist08, adjlist09],
    #         [adjlist10, adjlist11, adjlist12, adjlist13, adjlist14, adjlist15, adjlist16, adjlist17]],\
    #        [[idx00, idx01, idx02, idx03, idx04, idx05, idx06, idx07, idx08, idx09],
    #         [idx10, idx11, idx12, idx13, idx14, idx15, idx16, idx17]]



class mydataset(Dataset):
    def __init__(self, drug_protein_idx, y_true):
        self.drug_protein_idx = drug_protein_idx
        self.Y = y_true

    def __len__(self):
        return len(self.drug_protein_idx)

    def __getitem__(self, index):
        d_p_idx = self.drug_protein_idx[index].tolist()
        y = self.Y[index]

        return d_p_idx, y

class collate_fc(object):
    def __init__(self, adjlists, edge_metapath_indices_list, num_samples, offset, device):
        self.adjlists = adjlists
        self.edge_metapath_indices_list = edge_metapath_indices_list
        self.num_samples = num_samples
        self.offset = offset
        self.device = device

    def collate_func(self, batch_list):
        y_true = [y for _, y in batch_list]
        batch_list = [idx for idx, _ in batch_list]

        g_lists = [[], []]
        result_indices_lists = [[], []]
        idx_batch_mapped_lists = [[], []]
        for mode, (adjlists, edge_metapath_indices_list) in enumerate(zip(self.adjlists, self.edge_metapath_indices_list)):
            for adjlist, indices in zip(adjlists, edge_metapath_indices_list):
                edges, result_indices, num_nodes, mapping = parse_adjlist([adjlist[row[mode]] for row in batch_list],
                                                                          [indices[row[mode]] for row in batch_list],
                                                                          self.num_samples, offset=self.offset, mode=mode)

                g = dgl.DGLGraph()
                g.add_nodes(num_nodes)
                if len(edges) > 0:
                    sorted_index = sorted(range(len(edges)), key=lambda i: edges[i])
                    g.add_edges(*list(zip(*[(edges[i][1], edges[i][0]) for i in sorted_index])))
                    result_indices = torch.LongTensor(result_indices[sorted_index]).to(self.device)
                else:
                    result_indices = torch.LongTensor(result_indices).to(self.device)
                g_lists[mode].append(g)
                result_indices_lists[mode].append(result_indices)
                idx_batch_mapped_lists[mode].append(np.array([mapping[row[mode]] for row in batch_list]))

        return g_lists, result_indices_lists, idx_batch_mapped_lists, y_true, batch_list

def parse_adjlist(adjlist, edge_metapath_indices, samples=None, offset=None, mode=None):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                neighbors = row_parsed[1:]
                result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(np.random.choice(len(row_parsed) - 1, samples, replace=False, p=p))
                neighbors = [row_parsed[i + 1] for i in sampled_idx]
                result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {map_from: map_to for map_to, map_from in enumerate(sorted(nodes))}
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping

def get_features(args, type_mask):
    features_list = []
    in_dims = []
    if args.feats_type == 0:
        for i in range(args.num_ntype):
            dim = (type_mask == i).sum()
            in_dims.append(dim)
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list.append(torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(args.device))
    elif args.feats_type == 1:
        for i in range(args.num_ntype):
            dim = 10
            num_nodes = (type_mask == i).sum()
            in_dims.append(dim)
            features_list.append(torch.zeros((num_nodes, 10)).to(args.device))

    return features_list, in_dims

