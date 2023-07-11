import os
import sys
import time
import argparse
import numpy as np
import json
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from utils.pytorchtools import EarlyStopping
from utils.data2 import load_data, fold_train_test_idx, mydataset, collate_fc, get_features
from model.MAGNN_dti import MAGNN_lp

loss_bec = nn.BCELoss()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

setup_seed(20)



class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)

def get_MSE(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse

def get_adj(dti):
    len_dti = len(dti)
    dpp_adj = np.zeros((len_dti, len_dti), dtype=int)
    for i, dpp1 in enumerate(dti):
        for j, dpp2 in enumerate(dti):
            if (dpp1[0] == dpp2[0]) | (dpp1[1] == dpp2[1]):
                dpp_adj[i][j] = 1
    return dpp_adj


def training(net, optimizer, train_loader, features_list, type_mask):
    net.train()
    train_loss = 0
    total = 0
    for i, (train_g_lists, train_indices_lists, train_idx_batch_mapped_lists, y_train, batch_list) in enumerate(train_loader):
        y_train = torch.tensor(y_train).long().to(features_list[0].device)
        adj_i = get_adj(batch_list)
        adj_i = torch.FloatTensor(adj_i).to(features_list[0].device)
        # forward
        output = net((train_g_lists, features_list, type_mask, train_indices_lists, train_idx_batch_mapped_lists, adj_i))
        loss = F.nll_loss(torch.log(output), y_train)
        # loss = loss_bec(output[:, 1], y_train.float()) # the same to above

        # autograd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item() * len(y_train)
        total = total + len(y_train)

    return train_loss/total

def evaluate(net, test_loader, features_list, type_mask, y_true):
    net.eval()
    pred_val = []
    y_true_s = []
    with torch.no_grad():
        for i, (val_g_lists, val_indices_lists, val_idx_batch_mapped_lists, y_true, batch_list) in enumerate(test_loader):
            # forward
            adj_i = get_adj(batch_list)
            adj_i = torch.FloatTensor(adj_i).to(features_list[0].device)
            output = net((val_g_lists, features_list, type_mask, val_indices_lists, val_idx_batch_mapped_lists, adj_i))
            pred_val.append(output)
            y_true_s.append(torch.tensor(y_true).long().to(features_list[0].device))

        val_pred = torch.cat(pred_val)
        y_true = torch.cat(y_true_s)
        val_loss = F.nll_loss(torch.log(val_pred), y_true)
        val_pred = val_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        acc = accuracy_score(y_true, np.argmax(val_pred, axis=1))
        auc = roc_auc_score(y_true, val_pred[:, 1])
        aupr = average_precision_score(y_true, val_pred[:, 1])

    return val_loss, acc, auc, aupr, val_pred

def testing(net, test_loader, features_list, type_mask, y_true_test):
    net.eval()
    proba_list = []
    with torch.no_grad():
        for i, (test_g_lists, test_indices_lists, test_idx_batch_mapped_lists, y_test, batch_list) in enumerate(test_loader):
            # forward
            adj_i = get_adj(batch_list)
            adj_i = torch.FloatTensor(adj_i).to(features_list[0].device)
            output = net((test_g_lists, features_list, type_mask, test_indices_lists, test_idx_batch_mapped_lists, adj_i))
            proba_list.append(output)

        y_proba_test = torch.cat(proba_list)
        y_proba_test = y_proba_test.cpu().numpy()
    auc = roc_auc_score(y_true_test, y_proba_test[:, 1])
    aupr = average_precision_score(y_true_test, y_proba_test[:, 1])
    return auc, aupr, y_true_test, y_proba_test

def run_model(args):
    fold_path = args.data_dir + '/{}_folds/'.format(str(args.nFold))
    pos_folds = json.load(open(fold_path + 'pos_folds.json', 'r'))
    neg_folds = json.load(open(fold_path + 'neg_folds_times_{}.json'.format(str(args.neg_times), 'r')))
    type_mask = np.load(args.data_dir + '/processed/node_types.npy')
    drug_protein = np.loadtxt(args.data_dir + '/mat_data/mat_drug_protein.txt', dtype=int)

    f_csv = open(args.save_dir + 'results.csv', 'a')
    f_csv.write('Fold,AUC,AUPR\n')
    f_csv.close()

    for fold in range(args.nFold):
        results = {}
        print('\nThis is fold ', fold, '...')
        if os.path.exists(args.save_dir + '/checkpoint/checkpoint_fold_{}_best.pt'.format(fold)):
            print('The training of this fold has been completed!\n')
            continue
        train_fold_idx, test_fold_idx = fold_train_test_idx(pos_folds, neg_folds, args.nFold, fold)
        train_adjlists, train_edge_metapath_indices_list = load_data(args, fold, args.rp, args.neg_times, 'train')
        test_adjlists, test_edge_metapath_indices_list = load_data(args, fold, args.rp, args.neg_times, 'test')
        y_true_train = drug_protein[train_fold_idx[:,0], train_fold_idx[:,1]]
        y_true_test = drug_protein[test_fold_idx[:,0], test_fold_idx[:,1]]
        [num_metapaths_drug, num_metapaths_protein] = len(train_adjlists[0]), len(train_adjlists[1])

        # training set
        train_dataset = mydataset(train_fold_idx, y_true_train)
        train_collate = collate_fc(train_adjlists, train_edge_metapath_indices_list, num_samples=args.samples,
                                   offset=drug_protein.shape[0], device=args.device)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False,
                                  collate_fn=train_collate.collate_func)

        # test set
        test_dataset = mydataset(test_fold_idx, y_true_test)
        test_collate = collate_fc(test_adjlists, test_edge_metapath_indices_list, num_samples=args.samples,
                                   offset=drug_protein.shape[0], device=args.device)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False,
                                  collate_fn=test_collate.collate_func)

        # Input features
        features_list, in_dims = get_features(args, type_mask)

        # network
        net = MAGNN_lp([num_metapaths_drug, num_metapaths_protein], args.num_etypes, args.etypes_lists, in_dims,
                           args.hidden_dim, args.hidden_dim, args.num_heads, args.attn_vec_dim, args.rnn_type,
                           args.dropout_rate, args.attn_switch, args)
        net.to(args.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # early_stopping = EarlyStopping(patience=args.patience, verbose=True,
        #                                save_path=args.save_dir + '/checkpoint/checkpoint_{}.pt'.format(fold))
        make_dir(args.save_dir + '/checkpoint')

        best_acc = 0
        best_auc = 0
        best_aupr = 0
        pred = None
        counter = 0
        if args.only_test:
            # Test
            net.load_state_dict(torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold)))
            auc, aupr, ground_truth, y_pred = testing(net, test_loader, features_list, type_mask, y_true_test)
            best_auc, best_aupr, pred = auc, aupr, y_pred
        else:
            if os.path.exists(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold)):
                print('Load mdeol weights from /checkpoint/checkpoint_fold_{}.pt'.format(fold))
                net.load_state_dict(torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold), map_location=args.device))
            for epoch in range(args.epoch):
                # training
                train_loss = training(net, optimizer, train_loader, features_list, type_mask)
                # validation
                val_loss, acc, auc, aupr, y_pred = evaluate(net, test_loader, features_list, type_mask, y_true_test)
                print('Epoch {:d} | Train loss {:.6f} | Val loss {:.6f} | acc {:.4f} | auc {:.4f} | aupr {:.4f}'.format(
                    epoch, train_loss, val_loss, acc, auc, aupr))
                # early stopping
                if (best_aupr < aupr) | (best_acc < acc):
                    best_acc = acc
                    best_auc = auc
                    best_aupr, pred = aupr, y_pred
                    torch.save(net.state_dict(), args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold))
                    counter = 0
                else:
                    counter += 1

                if counter > args.patience:
                    print('Early stopping!')
                    break
            f_csv = open(args.save_dir + 'results.csv', 'a')
            f_csv.write(','.join(map(str, [fold, best_auc, best_aupr])) + '\n')
            f_csv.close()
            best_weights = torch.load(args.save_dir + '/checkpoint/checkpoint_fold_{}.pt'.format(fold), map_location=args.device)
            torch.save(best_weights, args.save_dir + '/checkpoint/checkpoint_fold_{}_best.pt'.format(fold))

        results['pred'] = pred.tolist()
        results['ground_truth'] = y_true_test.tolist()
        results['AUC'] = best_auc.item()
        results['AUPR'] = best_aupr.item()
        json.dump(results, open(args.save_dir.format(rp) + f'/fold000{fold}_pred_results.json', 'w'))


    res = pd.read_csv(args.save_dir + 'results.csv')
    try:
        auc_list = [float(res[res['Fold'] == i]['AUC'].values[0]) for i in range(args.nFold)]
        aupr_list = [float(res[res['Fold'] == i]['AUPR'].values[0]) for i in range(args.nFold)]
    except:
        auc_list = [float(res[res['Fold'] == str(i)]['AUC'].values[0]) for i in range(args.nFold)]
        aupr_list = [float(res[res['Fold'] == str(i)]['AUPR'].values[0]) for i in range(args.nFold)]

    print('----------------------------------------------------------------')
    print('Link Prediction Tests Summary')
    print('AUC_mean = {}, AUC_std = {}'.format(np.mean(auc_list), np.std(auc_list)))
    print('AUPR_mean = {}, AUPR_std = {}'.format(np.mean(aupr_list), np.std(aupr_list)))

    f_csv = open(args.save_dir + 'results.csv', 'a')
    f_csv.write(','.join(map(str, ['mean', np.mean(auc_list), np.mean(aupr_list)])) + '\n')
    f_csv.write(','.join(map(str, ['std', np.std(auc_list), np.std(aupr_list)])) + '\n')
    f_csv.close()


# Params
def parser():
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--feats_type', type=int, default=0,
                    help='Type of the node features used. 0 - all id vectors; 1 - all zero vector. Default is 0.')
    ap.add_argument('--hidden_dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num_heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--attn_vec_dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')
    ap.add_argument('--attn_switch', type=bool, default=True, help='attention considers the center node embedding or not')
    ap.add_argument('--rnn_type', default='max-pooling', help='Type of the aggregator. max-pooling, average, linear, neighbor, RotatE0.')
    ap.add_argument('--predictor', default='gcn', help='options: linear, gcn.')
    ap.add_argument('--semantic_fusion', default='concatenation', help='options: concatenation, attention, max-pooling, average.')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 100.')
    ap.add_argument('--patience', type=int, default=15, help='Patience. Default is 5.')
    ap.add_argument('--batch_size', type=int, default=256, help='Batch size. Default is 8.')
    ap.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--num_ntype', default=4, type=int, help='Number of node types')
    ap.add_argument('--lr', default=0.0001)
    ap.add_argument('--weight_decay', default=1e-5)
    ap.add_argument('--dropout_rate', default=0.5)
    ap.add_argument('--num_workers', default=0, type=int)

    ap.add_argument('--nFold', default=10, type=int)
    ap.add_argument('--neg_times', default=1, type=int, help='The ratio between positive samples and negative samples')
    ap.add_argument('--data_dir', default='/media/data2/lm/Experiments/MHAN-DTI/hetero_dataset/{}/')
    ap.add_argument('--save_dir',
                    default='./results_dpp/{}/repeat{}/neg_times{}_{}_{}_{}_num_head{}_hidden_dim{}_batch_sz{}_semantic_fusion_{}'
                            '_predictor_{}/',
                    help='Postfix for the saved model and result. Default is LastFM.')
    ap.add_argument('--only_test', default=False, type=bool)
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parser()
    args.dataset = 'data'
    args.data_dir = args.data_dir.format(args.dataset)
    # args.save_dir = args.save_dir.format(args.dataset)
    # make_dir(args.save_dir)

    etypes_lists = [
        [[None], [0, 1], [2, 3], [4, 5], [0, None, 1]],# [0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5], [0, 6, 7, 1], [2, 7, 6, 3]],
        [[None], [1, 0], [6, 7], [1, None, 0]]#, [1, 0, 1, 0], [1, 2, 3, 0], [6, 3, 2, 7], [6, 7, 6, 7]]
    ]

    expected_metapaths = [
        [(0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (0, 1, 1, 0)],
         # (0, 1, 0, 1, 0), (0, 2, 0, 2, 0), (0, 3, 0, 3, 0), (0, 1, 2, 1, 0), (0, 2, 1, 2, 0)],
        [(1, 1), (1, 0, 1), (1, 2, 1), (1, 0, 0, 1)],
         # (1, 0, 1, 0, 1), (1, 0, 2, 0, 1), (1, 2, 0, 2, 1), (1, 2, 1, 2, 1)]
    ]

    args.etypes_lists = etypes_lists
    args.num_etypes = 8
    args.expected_metapaths = expected_metapaths

    for rp in range(args.repeat):
        print('This is repeat ', rp)
        args.rp = rp
        save_dir = args.save_dir
        args.save_dir = args.save_dir.format(args.dataset, args.rp, args.neg_times, args.rnn_type.capitalize(),
                                             len(args.expected_metapaths[0]), len(args.expected_metapaths[1]),
                                             args.num_heads, args.hidden_dim, args.batch_size,
                                             args.semantic_fusion, args.predictor)
        print('Save path ', args.save_dir)
        make_dir(args.save_dir)
        sys.stdout = Logger(args.save_dir + 'log.txt')

        run_model(args)

        print('Save path ', args.save_dir)
        args.save_dir = save_dir
