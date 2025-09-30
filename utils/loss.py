import torch
import torch.nn.functional as F
from loss_util import get_positive_expectation, get_negative_expectation
from torch_geometric.utils import negative_sampling

EPS = 1e-15
MAX_LOGVAR = 10


def mutual_loss(node_features, adj, order, measure='JSD'):
    num_nodes = adj.shape[0]
    node_features_use = node_features[:num_nodes]
    adj_new = (adj != 0).float()  # weighted adj->0/1

    ordered_adj = adj_new.clone()
    pos_mask = adj_new.clone()

    for i in range(order - 1):
        ordered_adj = torch.matmul(ordered_adj, adj_new)
        pos_mask = ((pos_mask != 0) | (ordered_adj != 0).int()).cuda()

    neg_mask = (torch.ones_like(adj_new)).cuda() - pos_mask
    res = torch.mm(node_features_use, node_features_use.t())
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()

    pos_num = torch.count_nonzero(pos_mask)
    neg_num = torch.count_nonzero(neg_mask)

    E_pos = E_pos / pos_num
    E_neg = E_neg / (neg_num + (1e-10))

    return E_neg - E_pos


def recon_loss(node_features, pos_edge_index, sampling_times=1):
    pos_value = (node_features[pos_edge_index[0]] * node_features[pos_edge_index[1]]).sum(dim=1)
    pos_value = torch.sigmoid(pos_value)
    pos_loss = -torch.log(pos_value + EPS).mean()
    neg_edge_index = negative_sampling(pos_edge_index,
                                       num_neg_samples=pos_edge_index.size(1) * sampling_times)

    neg_value = (node_features[neg_edge_index[0]] * node_features[neg_edge_index[1]]).sum(dim=1)
    neg_value = torch.sigmoid(neg_value)
    neg_loss = -torch.log(1 - neg_value + EPS).mean()

    return pos_loss + neg_loss

def consistency_loss(structral_feature, temporal_feature):
    p = F.softmax(structral_feature, dim=-1) if structral_feature.sum(-1).abs().max() > 1.01 else structral_feature
    q = F.softmax(temporal_feature, dim=-1) if temporal_feature.sum(-1).abs().max() > 1.01 else temporal_feature

    m = 0.5 * (p + q)

    loss = 0.5 * (
            F.kl_div(p.log(), m, reduction='batchmean') +
            F.kl_div(q.log(), m, reduction='batchmean')
    )
    return loss


def independence_loss(structural_feature, temporal_feature):
    std_x = torch.sqrt(structural_feature.var(dim=0) + 0.0001)
    std_y = torch.sqrt(temporal_feature.var(dim=0) + 0.0001)

    loss = torch.sum(torch.sqrt(((1 - std_x) ** 2) + 0.0001)) / \
           2 + torch.sum(torch.sqrt(((1 - std_y) ** 2) + 0.0001)) / 2

    return loss
