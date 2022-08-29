import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from model.base_MAGNN_dti import MAGNN_ctr_ntype_specific

def adj_normalize(adj):
    rowsum = adj.sum(1)
    r_inv = torch.pow(rowsum, -0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    adj_ = r_mat_inv * adj * r_mat_inv
    return adj_

def MinMax_scalar(x):
    min = x.min(1).values
    min = min.view(-1, 1).repeat(1, x.shape[1])
    max = x.max(1).values
    max = max.view(-1, 1).repeat(1, x.shape[1])
    scalar = (x - min) / (max - min)
    return scalar

def normalize(x):
    rowsum = x.sum(1)
    rowsum = rowsum.view(-1, 1).repeat(1, x.shape[1])
    x_norm = x / rowsum
    return x_norm


# for link prediction task
class MAGNN_lp_layer(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 in_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 attn_drop=0.5,
                 attn_switch=False,
                 args=None):
        super(MAGNN_lp_layer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # etype-specific parameters
        r_vec = None
        if rnn_type == 'TransE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim)))
        elif rnn_type == 'TransE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim)))
        elif rnn_type == 'RotatE0':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type // 2, in_dim // 2, 2)))
        elif rnn_type == 'RotatE1':
            r_vec = nn.Parameter(torch.empty(size=(num_edge_type, in_dim // 2, 2)))
        if r_vec is not None:
            nn.init.xavier_normal_(r_vec.data, gain=1.414)

        # ctr_ntype-specific layers
        self.user_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[0],
                                                   etypes_lists[0],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   args=args)
        self.item_layer = MAGNN_ctr_ntype_specific(num_metapaths_list[1],
                                                   etypes_lists[1],
                                                   in_dim,
                                                   num_heads,
                                                   attn_vec_dim,
                                                   rnn_type,
                                                   r_vec,
                                                   attn_drop,
                                                   use_minibatch=True,
                                                   attn_switch=attn_switch,
                                                   args=args)

        # note that the acutal input dimension should consider the number of heads
        # as multiple head outputs are concatenated together
        # self.fc_user = nn.Linear(in_dim * num_heads * num_metapaths_list[0], out_dim * num_heads, bias=True)
        # self.fc_item = nn.Linear(in_dim * num_heads * num_metapaths_list[1], out_dim * num_heads, bias=True)
        # nn.init.xavier_normal_(self.fc_user.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc_item.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features, type_mask, edge_metapath_indices_lists, target_idx_lists = inputs

        # ctr_ntype-specific layers
        h_user = self.user_layer(
            (g_lists[0], features, type_mask, edge_metapath_indices_lists[0], target_idx_lists[0]))
        h_item = self.item_layer(
            (g_lists[1], features, type_mask, edge_metapath_indices_lists[1], target_idx_lists[1]))

        return [h_user, h_item]

        # logits_user = self.fc_user(h_user)
        # logits_item = self.fc_item(h_item)
        # return [logits_user, logits_item], [h_user, h_item]

class GCN_layer(nn.Module):
    def __init__(self, dim, mp_ls=None):
        super(GCN_layer, self).__init__()
        self.gcn1 = nn.Parameter(torch.zeros([dim, 128]), requires_grad=True)
        self.gcn2 = nn.Parameter(torch.zeros([128, 2]), requires_grad=True)
        nn.init.xavier_normal_(self.gcn1, gain=1.414)
        nn.init.xavier_normal_(self.gcn2, gain=1.414)

    def forward(self, x, adj):
        # x = MinMax_scalar(x)
        # x = normalize(x)
        adj = F.softmax(torch.matmul(x, x.T), dim=-1)
        # adj = adj_normalize(adj)
        x = F.relu(torch.matmul(torch.matmul(adj, x), self.gcn1))
        x = torch.matmul(torch.matmul(adj, x), self.gcn2)

        return x


class linear_module(nn.Module):
    def __init__(self, dim):
        super(linear_module, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim/2), bias=True)
        self.fc2 = nn.Linear(int(dim/2), 2, bias=False)
        # weight initialization
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    def forward(self, x, x2=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class MAGNN_lp(nn.Module):
    def __init__(self,
                 num_metapaths_list,
                 num_edge_type,
                 etypes_lists,
                 feats_dim_list,
                 hidden_dim,
                 out_dim,
                 num_heads,
                 attn_vec_dim,
                 rnn_type='gru',
                 dropout_rate=0.5,
                 attn_switch=False,
                 args=None):
        super(MAGNN_lp, self).__init__()
        self.hidden_dim = hidden_dim
        self.args = args

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # MAGNN_lp layers
        self.layer1 = MAGNN_lp_layer(num_metapaths_list,
                                     num_edge_type,
                                     etypes_lists,
                                     hidden_dim,
                                     out_dim,
                                     num_heads,
                                     attn_vec_dim,
                                     rnn_type,
                                     attn_drop=dropout_rate,
                                     attn_switch=attn_switch,
                                     args=args)
        dim = out_dim * num_heads * 2
        if self.args.semantic_fusion == 'concatenation':
            dim = out_dim * num_heads * (num_metapaths_list[0] + num_metapaths_list[1])
        # predictor
        if self.args.predictor =='gcn':
            self.classifier = GCN_layer(dim=dim)
        elif self.args.predictor == 'linear':
            self.classifier = linear_module(dim=dim)

    def forward(self, inputs):
        g_lists, features_list, type_mask, edge_metapath_indices_lists, target_idx_lists, adj = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        # hidden layers
        # [logits_user, logits_item], [h_user, h_item] = self.layer1(
        #     (g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        [h_user, h_item] = self.layer1((g_lists, transformed_features, type_mask, edge_metapath_indices_lists, target_idx_lists))
        x = torch.cat([h_user, h_item], dim=1)
        x_out = self.classifier(x, adj)

        return F.softmax(x_out, dim=-1)

    def feature_transform(self, type_mask, features_list):
        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        return transformed_features
