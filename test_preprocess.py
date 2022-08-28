#!/usr/bin/env python
# coding: utf-8
import os
import pathlib
import pickle
import random
import numpy as np
import scipy.sparse as sp
import scipy.io
import pandas as pd
import json

seed = 12345
np.random.seed(seed)
random.seed(seed)

def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)

def metapath_xx(x_x_list, num, sample=None):
    x_x = []
    for x, x_list in x_x_list.items():
        if sample is not None:
            candidate_list = np.random.choice(len(x_list), min(len(x_list), sample), replace=False)
            x_list = x_list[candidate_list]
        x_x.extend([(x, x1) for x1 in x_list])
    x_x = np.array(x_x)
    x_x = x_x + num
    sorted_index = sorted(list(range(len(x_x))), key=lambda i: x_x[i].tolist())
    x_x = x_x[sorted_index]
    return x_x

def metapath_yxy(x_y_list, num1, num2, sample=None):
    y_x_y = []
    for x, y_list in x_y_list.items():
        if sample is not None:
            candidate_list1 = np.random.choice(len(y_list), min(len(y_list), sample), replace=False)
            candidate_list2 = np.random.choice(len(y_list), min(len(y_list), sample), replace=False)
            # print(len(candidate_list1))
            y_list1 = y_list[candidate_list1]
            y_list2 = y_list[candidate_list2]
            y_x_y.extend([(y1, x, y2) for y1 in y_list1 for y2 in y_list2])
        else:
            y_x_y.extend([(y1, x, y2) for y1 in y_list for y2 in y_list])
    y_x_y = np.array(y_x_y)
    y_x_y[:, [0, 2]] += num1
    y_x_y[:, 1] += num2
    sorted_index = sorted(list(range(len(y_x_y))), key=lambda i: y_x_y[i, [0, 2, 1]].tolist())
    y_x_y = y_x_y[sorted_index]
    return y_x_y

def metapath_yxxy(x_x, x_y_list, num1, num2, sample=None):
    y_x_x_y = []
    for x1, x2 in x_x:
        if sample is not None:
            candidate_list1 = np.random.choice(len(x_y_list[x1 - num2]), min(len(x_y_list[x1 - num2]), sample), replace=False)
            candidate_list2 = np.random.choice(len(x_y_list[x2 - num2]), min(len(x_y_list[x2 - num2]), sample), replace=False)
            # print(len(candidate_list1))
            x_y_list1 = x_y_list[x1 - num2][candidate_list1]
            x_y_list2 = x_y_list[x2 - num2][candidate_list2]
            y_x_x_y.extend([(y1, x1, x2, y2) for y1 in x_y_list1 for y2 in x_y_list2])
        else:
            y_x_x_y.extend([(y1, x1, x2, y2) for y1 in x_y_list[x1 - num2] for y2 in x_y_list[x2 - num2]])
    y_x_x_y = np.array(y_x_x_y)
    y_x_x_y[:, [0, 3]] += num1
    sorted_index = sorted(list(range(len(y_x_x_y))), key=lambda i: y_x_x_y[i, [0, 3, 1, 2]].tolist())
    y_x_x_y = y_x_x_y[sorted_index]
    return y_x_x_y

def metapath_zyxyz(y_x_y, y_z_list, num1, num2, ratio):
    # z-y-x-y-z
    z_y_x_y_z = []
    for y1, x, y2 in y_x_y:
        if len(y_z_list[y1 - num2]) == 0 or len(y_z_list[y2 - num2]) == 0:
            continue
        if ratio <= 1:
            candidate_z1_list = np.random.choice(len(y_z_list[y1 - num2]), int(ratio * len(y_z_list[y1 - num2])) + 1, replace=False)
            candidate_z2_list = np.random.choice(len(y_z_list[y2 - num2]), int(ratio * len(y_z_list[y2 - num2])) + 1, replace=False)
        else:
            candidate_z1_list = np.random.choice(len(y_z_list[y1 - num2]), min(ratio, len(y_z_list[y1 - num2])), replace=False)
            candidate_z2_list = np.random.choice(len(y_z_list[y2 - num2]), min(ratio, len(y_z_list[y2 - num2])), replace=False)
        candidate_z1_list = y_z_list[y1 - num2][candidate_z1_list]
        candidate_z2_list = y_z_list[y2 - num2][candidate_z2_list]

        z_y_x_y_z.extend([(z1, y1, x, y2, z2) for z1 in candidate_z1_list for z2 in candidate_z2_list])
    z_y_x_y_z = np.array(z_y_x_y_z)
    z_y_x_y_z[:, [0, 4]] += num1
    sorted_index = sorted(list(range(len(z_y_x_y_z))), key=lambda i: z_y_x_y_z[i, [0, 4, 1, 2, 3]].tolist())
    z_y_x_y_z = z_y_x_y_z[sorted_index]
    return z_y_x_y_z

def sampling(array_list, num, offset):
    target_list = np.arange(num).tolist()
    sampled_list = []
    k = 100 # number of samiling

    left = 0
    right = 0
    for target_idx in target_list:
        while right < len(array_list) and array_list[right, 0] == target_idx + offset:
            right += 1
        target_array = array_list[left:right, :]

        if len(target_array) > 0:
            samples = min(k, len(target_array))
            sampled_idx = np.random.choice(len(target_array), samples, replace=False)
            target_array = target_array[sampled_idx]

        sampled_list.append(target_array)
        left = right
    sampled_array = np.concatenate(sampled_list, axis=0)
    sorted_index = sorted(list(range(len(sampled_array))), key=lambda i: sampled_array[i, [0, 2, 1]].tolist())
    sampled_array = sampled_array[sorted_index]

    return sampled_array


def get_metapath(metapath, num_drug, num_protein, num_disease, num_se, save_prefix):
    if len(metapath) == 2:
        # (0, 0)
        if metapath == (0, 0):
            metapath_indices = metapath_xx(drug_drug_list, num=0)
        # (1, 1)
        elif metapath == (1, 1):
            metapath_indices = metapath_xx(protein_protein_list, num=num_drug)

    elif len(metapath) == 3:
        # (0, 1, 0)
        if metapath == (0, 1, 0):
            metapath_indices = metapath_yxy(protein_drug_list, num1=0, num2=num_drug)
        # (0, 2, 0)
        elif metapath == (0, 2, 0):
            metapath_indices = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=100)
        # (0, 3, 0)
        elif metapath == (0, 3, 0):
            metapath_indices = metapath_yxy(se_drug_list, num1=0, num2=num_drug + num_protein + num_disease, sample=100)
        # (1, 0, 1)
        elif metapath == (1, 0, 1):
            metapath_indices = metapath_yxy(drug_protein_list, num1=num_drug, num2=0)
        # (1, 2, 1)
        elif metapath == (1, 2, 1):
            metapath_indices = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=100)

    elif len(metapath) == 4:
        # (0, 1, 1, 0)
        if metapath == (0, 1, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 1))) + '.npy'):
            #     p_p = np.load(save_prefix + '-'.join(map(str, (1, 1))) + '.npy')
            # # else:
            #     p_p = metapath_xx(protein_protein_list, num=num_drug)
            #     np.save(save_prefix + '-'.join(map(str, (1, 1))) + '.npy', p_p)
            p_p = metapath_xx(protein_protein_list, num=num_drug, sample=50)
            metapath_indices = metapath_yxxy(p_p, protein_drug_list, num1=0, num2=num_drug, sample=30)
        # (1, 0, 0, 1)
        elif metapath == (1, 0, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 0))) + '.npy'):
            #     d_d = np.load(save_prefix + '-'.join(map(str, (0, 0))) + '.npy')
            # else:
            #     d_d = metapath_xx(drug_drug_list, num=0)
            #     np.save(save_prefix + '-'.join(map(str, (0, 0))) + '.npy', d_d)
            d_d = metapath_xx(drug_drug_list, num=0, sample=100)
            metapath_indices = metapath_yxxy(d_d, drug_protein_list, num1=num_drug, num2=0, sample=10)

    elif len(metapath) == 5:
        # 0-1-0-1-0
        if metapath == (0, 1, 0, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy'):
            #     p_d_p = np.load(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy')
            # else:
            #     p_d_p = metapath_yxy(drug_protein_list, num1=num_drug, num2=0)
            #     np.save(save_prefix + '-'.join(map(str, (1, 0, 1))) + '.npy', p_d_p)
            p_d_p = metapath_yxy(drug_protein_list, num1=num_drug, num2=0, sample=30)
            p_d_p = sampling(p_d_p, num=num_protein, offset=num_drug)
            metapath_indices = metapath_zyxyz(p_d_p, protein_drug_list, num1=0, num2=num_drug, ratio=5)
        # 0-1-2-1-0
        elif metapath == (0, 1, 2, 1, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy'):
            #     p_i_p = np.load(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy')
            # else:
            #     p_i_p = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (1, 2, 1))) + '.npy', p_i_p)
            p_i_p = metapath_yxy(disease_protein_list, num1=num_drug, num2=num_drug + num_protein, sample=80)
            p_i_p = sampling(p_i_p, num=num_protein, offset=num_drug)
            metapath_indices = metapath_zyxyz(p_i_p, protein_drug_list, num1=0, num2=num_drug, ratio=5)
        # 0-2-0-2-0
        elif metapath == (0, 2, 0, 2, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy'):
            #     i_d_i = np.load(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy')
            # else:
            #     i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy', i_d_i)
            i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            i_d_i = sampling(i_d_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_d_i, disease_drug_list, num1=0, num2=num_drug + num_protein, ratio=5)
        # 0-3-0-3-0
        elif metapath == (0, 3, 0, 3, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy'):
            #     s_d_s = np.load(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy')
            # else:
            #     s_d_s = metapath_yxy(drug_se_list, num1=num_drug + num_protein + num_disease, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (3, 0, 3))) + '.npy', s_d_s)
            s_d_s = metapath_yxy(drug_se_list, num1=num_drug + num_protein + num_disease, num2=0, sample=80)
            s_d_s = sampling(s_d_s, num=num_se, offset=num_drug + num_protein + num_disease)
            metapath_indices = metapath_zyxyz(s_d_s, se_drug_list, num1=0, num2=num_drug + num_protein + num_disease, ratio=5)
        # 0-2-1-2-0
        elif metapath == (0, 2, 1, 2, 0):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy'):
            #     i_p_i = np.load(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy')
            # else:
            #     i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy', i_p_i)
            i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            i_p_i = sampling(i_p_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_p_i, disease_drug_list, num1=0, num2=num_drug + num_protein, ratio=5)
        # 1-0-1-0-1
        elif metapath == (1, 0, 1, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy'):
            #     d_p_d = np.load(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy')
            # else:
            #     d_p_d = metapath_yxy(protein_drug_list, num1=0, num2=num_drug)
            #     np.save(save_prefix + '-'.join(map(str, (0, 1, 0))) + '.npy', d_p_d)
            d_p_d = metapath_yxy(protein_drug_list, num1=0, num2=num_drug, sample=50)
            d_p_d = sampling(d_p_d, num=num_drug, offset=0)
            metapath_indices = metapath_zyxyz(d_p_d, drug_protein_list, num1=num_drug, num2=0, ratio=5)
        # 1-0-2-0-1
        elif metapath == (1, 0, 2, 0, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy'):
            #     d_i_d = np.load(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy')
            # else:
            #     d_i_d = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (0, 2, 0))) + '.npy', d_i_d)
            d_i_d = metapath_yxy(disease_drug_list, num1=0, num2=num_drug + num_protein, sample=80)
            d_i_d = sampling(d_i_d, num=num_drug, offset=0)
            metapath_indices = metapath_zyxyz(d_i_d, drug_protein_list, num1=num_drug, num2=0, ratio=5)
        # 1-2-0-2-1
        elif metapath == (1, 2, 0, 2, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy'):
            #     i_d_i = np.load(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy')
            # else:
            #     i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 0, 2))) + '.npy', i_d_i)
            i_d_i = metapath_yxy(drug_disease_lit, num1=num_drug + num_protein, num2=0, sample=80)
            i_d_i = sampling(i_d_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_d_i, disease_protein_list, num1=num_drug, num2=num_drug + num_protein, ratio=5)
        # 1-2-1-2-1
        elif metapath == (1, 2, 1, 2, 1):
            # if os.path.isfile(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy'):
            #     i_p_i = np.load(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy')
            # else:
            #     i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            #     np.save(save_prefix + '-'.join(map(str, (2, 1, 2))) + '.npy', i_p_i)
            i_p_i = metapath_yxy(protein_disease_list, num1=num_drug + num_protein, num2=num_drug, sample=80)
            i_p_i = sampling(i_p_i, num=num_disease, offset=num_drug + num_protein)
            metapath_indices = metapath_zyxyz(i_p_i, disease_protein_list, num1=num_drug, num2=num_drug + num_protein, ratio=5)

    return metapath_indices

def target_metapath_and_neightbors(edge_metapath_idx_array, target_idx_list, offset):
    # write all things
    target_metapaths_mapping = {}
    target_neighbors = {}
    left = 0
    right = 0
    for target_idx in target_idx_list:
        # target_metapaths_mapping = {}
        # target_neighbors = {}
        while right < len(edge_metapath_idx_array) and edge_metapath_idx_array[right, 0] == target_idx + offset:
            right += 1
        target_metapaths_mapping[target_idx] = edge_metapath_idx_array[left:right, ::-1]
        neighbors = edge_metapath_idx_array[left:right, -1] - offset_list[i]
        # neighbors = list(map(str, neighbors))
        target_neighbors[target_idx] = [target_idx] + neighbors.tolist()
        left = right

    return target_metapaths_mapping, target_neighbors


def Load_Adj_Togerther(dir_lists, ratio=0.01):
    a = np.loadtxt(dir_lists[0])
    print('Before Interactions: ', sum(sum(a)))

    for i in range(len(dir_lists) - 1):
        b_new = np.zeros_like(a)

        b = np.loadtxt(dir_lists[i + 1])
        # remove diagonal elements
        b = b - np.diag(np.diag(b))
        # if the matrix are symmetrical, get the triu matrix
        if (b == b.T).all():
            b = np.triu(b)
        index = np.nonzero(b)
        values = b[index]
        index = np.transpose(index)
        edgelist = np.concatenate([index, values.reshape(-1, 1)], axis=1)
        topK_idx = np.argpartition(edgelist[:, 2], int(ratio * len(edgelist)))[-(int(ratio * len(edgelist))):]
        print(len(topK_idx))
        select_idx = index[topK_idx]
        b_new[select_idx[:, 0], select_idx[:, 1]] = b[select_idx[:, 0], select_idx[:, 1]]
        a = a + b_new

    a = a + a.T
    a[a > 0] = 1
    a[a <= 0] = 0
    a = a + np.eye(a.shape[0], a.shape[1])
    a = a.astype(int)
    print('After Interactions: ', sum(sum(a)))

    return a

def get_adjM(drug_drug, drug_protein, drug_disease, drug_sideEffect, protein_protein, protein_disease,
             num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    adjM = np.zeros((dim, dim), dtype=int)
    adjM[:num_drug, :num_drug] = drug_drug
    adjM[:num_drug, num_drug: num_drug + num_protein] = drug_protein
    adjM[:num_drug, num_drug + num_protein: num_drug + num_protein + num_disease] = drug_disease
    adjM[:num_drug, num_drug + num_protein + num_disease:] = drug_sideEffect
    adjM[num_drug: num_drug + num_protein, num_drug: num_drug + num_protein] = protein_protein
    adjM[num_drug: num_drug + num_protein, num_drug + num_protein: num_drug + num_protein + num_disease] = protein_disease

    adjM[num_drug: num_drug + num_protein, :num_drug] = drug_protein.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease, :num_drug] = drug_disease.T
    adjM[num_drug + num_protein + num_disease:, :num_drug] = drug_sideEffect.T
    adjM[num_drug + num_protein: num_drug + num_protein + num_disease, num_drug: num_drug + num_protein] = protein_disease.T
    
    return adjM

def fold_test_idx(pos_folds, neg_folds, foldID):
    fold_idx = [[], []]
    fold_posIdx = pos_folds['fold_' + str(foldID)]
    fold_negIdx = neg_folds['fold_' + str(foldID)]
    fold_idx[0] = fold_posIdx[0] + fold_negIdx[0]
    fold_idx[1] = fold_posIdx[1] + fold_negIdx[1]
    return fold_idx

def get_type_mask(num_drug, num_protein, num_disease, num_se):
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    dim = num_drug + num_protein + num_disease + num_se
    type_mask = np.zeros((dim), dtype=int)
    type_mask[num_drug: num_drug + num_protein] = 1
    type_mask[num_drug + num_protein: num_drug + num_protein + num_disease] = 2
    type_mask[num_drug + num_protein + num_disease:] = 3
    return type_mask

if __name__ == '__main__':
    data_set = 'data_luo'
    nFold = 10
    neg_times = 1
    data_dir = './hetero_dataset/{}/'.format(data_set)
    fold_path = data_dir + '/{}_folds/'.format(str(nFold))
    pos_folds = json.load(open(fold_path + 'pos_folds.json', 'r'))
    neg_folds = json.load(open(fold_path + 'neg_folds_times_{}.json'.format(str(neg_times), 'r')))
    num_repeats = 1 # (repeat 10 times)

    save_prefix = data_dir + '/processed/'
    os.makedirs(save_prefix, exist_ok=True)

    expected_metapaths = [[(0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 1, 1, 0),
                          (0, 1, 0, 1, 0), (0, 2, 0, 2, 0), (0, 3, 0, 3, 0), (0, 1, 2, 1, 0), (0, 2, 1, 2, 0)],
                          [(1, 1), (1, 0, 1), (1, 2, 1), (1, 0, 0, 1),
                           (1, 0, 1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1, 2, 1)]]

    ## Step 1: Reconstruct Drug-Drug interaction network and Protein-Protein interaxtion network
    # Reconstruct Drug-Drug interaction network
    # 1 interaction + 4 sim
    drug_drug_path = data_dir + '/mat_data/mat_drug_drug.txt'
    drug_drug_sim_chemical_path = data_dir + '/sim_network/Sim_mat_drugs.txt'
    drug_drug_sim_interaction_path = data_dir + '/sim_network/Sim_mat_drug_drug.txt'
    drug_drug_sim_se_path = data_dir + '/sim_network/Sim_mat_drug_se.txt'
    drug_drug_sim_disease_path = data_dir + '/sim_network/Sim_mat_drug_disease.txt'

    # Reconstruct Protein-Protein interaxtion network
    # 1interaction + 3 sim
    protein_protein_path = data_dir + '/mat_data/mat_protein_protein.txt'
    protein_protein_sim_sequence_path = data_dir + '/sim_network/Sim_mat_proteins.txt'
    protein_protein_sim_disease_path = data_dir + '/sim_network/Sim_mat_protein_disease.txt'
    protein_protein_sim_interaction_path = data_dir + '/sim_network/Sim_mat_protein_protein.txt'

    # About drug and protein (others)...
    drug_protein_path = data_dir + '/mat_data/mat_drug_protein.txt'
    drug_disease_path = data_dir + '/mat_data/mat_drug_disease.txt'
    drug_sideEffect_path = data_dir + '/mat_data/mat_drug_se.txt'
    protein_disease_path = data_dir + '/mat_data/mat_protein_disease.txt'

    # drug_drug and protein_protein combine the simNets and interactions
    # print('Load_Drug_Adj_Togerther ...')
    # drug_drug = Load_Adj_Togerther(dir_lists=[drug_drug_path, drug_drug_sim_chemical_path,
    #                                           drug_drug_sim_interaction_path, drug_drug_sim_se_path,
    #                                           drug_drug_sim_disease_path], ratio=0.01)
    #
    # print('Load_Protein_Adj_Togerther ...')
    # protein_protein = Load_Adj_Togerther(dir_lists=[protein_protein_path, protein_protein_sim_sequence_path,
    #                                                 protein_protein_sim_disease_path, protein_protein_sim_interaction_path],
    #                                      ratio=0.005)

    drug_drug = np.loadtxt(drug_drug_path, dtype=int)
    drug_protein = np.loadtxt(drug_protein_path, dtype=int)
    drug_disease = np.loadtxt(drug_disease_path, dtype=int)
    protein_protein = np.loadtxt(protein_protein_path, dtype=int)
    drug_sideEffect = np.loadtxt(drug_sideEffect_path, dtype=int)
    protein_disease = np.loadtxt(protein_disease_path, dtype=int)

    print(sum(sum(drug_drug)), sum(sum(drug_protein)), sum(sum(drug_disease)), sum(sum(protein_protein)),
          sum(sum(drug_sideEffect)), sum(sum(protein_disease)))

    num_drug, num_protein = drug_protein.shape
    num_disease = drug_disease.shape[1]
    num_se = drug_sideEffect.shape[1]
    type_mask = get_type_mask(num_drug, num_protein, num_disease, num_se)
    np.save(save_prefix + 'node_types.npy', type_mask)

    ## Syep 2: Build the Adjacency Matrix
    # Drug-0, Protein-1, Disease-2, Side-effect-3
    adjM = get_adjM(drug_drug, drug_protein, drug_disease, drug_sideEffect, protein_protein, protein_disease,
                    num_drug, num_protein, num_disease, num_se)
    # sp.save_npz(save_prefix + 'adjM_test.npz', sp.csr_matrix(adjM))

    drug_drug_list = {i: adjM[i, :num_drug].nonzero()[0] for i in range(num_drug)}
    drug_protein_list = {i: adjM[i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_drug)}
    drug_disease_lit = {i: adjM[i, num_drug + num_protein:num_drug + num_protein + num_disease].nonzero()[0] for i in range(num_drug)}
    drug_se_list = {i: adjM[i, num_drug + num_protein + num_disease:].nonzero()[0] for i in range(num_drug)}
    protein_drug_list = {i: adjM[num_drug + i, :num_drug].nonzero()[0] for i in range(num_protein)}
    protein_protein_list = {i: adjM[num_drug + i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_protein)}
    protein_disease_list = { i: adjM[num_drug + i, num_drug + num_protein:num_drug + num_protein + num_disease].nonzero()[0]
                             for i in range(num_protein)}
    disease_drug_list = {i: adjM[num_drug + num_protein + i, :num_drug].nonzero()[0] for i in range(num_disease)}
    disease_protein_list = {i: adjM[num_drug + num_protein + i, num_drug:num_drug + num_protein].nonzero()[0] for i in range(num_disease)}
    se_drug_list = {i: adjM[num_drug + num_protein + num_disease + i, : num_drug].nonzero()[0] for i in range(num_se)}

    # Step 3: Get target metapaths and neighbors for each test fold
    target_idx_lists = [np.arange(num_drug).tolist(), np.arange(num_protein).tolist()]
    offset_list = [0, num_drug]

    for counter in range(num_repeats): # repeat ten times
        print('\nThis is the {} repeat...'.format(counter))
        for i, metapaths in enumerate(expected_metapaths):
            # print(metapaths)
            for metapath in metapaths:
                metapath_dir = save_prefix + 'repeat{}/{}/test/neg_times_{}/'.format(counter, i, neg_times)
                make_dir(metapath_dir)
                # Get all the metapaths in the schema of 'metapath'
                if os.path.isfile(metapath_dir + '-'.join(map(str, metapath)) + '.npy'):
                    edge_metapath_idx_array = np.load(metapath_dir + '-'.join(map(str, metapath)) + '.npy')
                else:
                    edge_metapath_idx_array = get_metapath(metapath, num_drug, num_protein, num_disease, num_se, metapath_dir)
                    np.save(metapath_dir + '-'.join(map(str, metapath)) + '.npy', edge_metapath_idx_array)
                print(metapath, len(edge_metapath_idx_array))

                target_metapaths, target_neighbors = target_metapath_and_neightbors(edge_metapath_idx_array, target_idx_lists[i], offset=offset_list[i])
                # print('\n')
                for foldID in range(nFold):
                    # print('Fold {}, metapath {}'.format(foldID, metapath))
                    metapath_fold_dir = metapath_dir + '/fold_{}/'.format(str(foldID)) + '-'.join(map(str, metapath))
                    make_dir(os.path.dirname(metapath_fold_dir))
                    test_fold_idx = fold_test_idx(pos_folds, neg_folds, foldID)
                    fold_target_metapaths = {target: target_metapaths[target] for target in test_fold_idx[i]}
                    fold_target_neighbors = {target: target_neighbors[target] for target in test_fold_idx[i]}
                    pickle.dump(fold_target_metapaths, open(metapath_fold_dir + '.idx.pkl', 'wb'))
                    pickle.dump(fold_target_neighbors, open(metapath_fold_dir + '.adjlist.pkl', 'wb'))
