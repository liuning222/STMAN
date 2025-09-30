import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import softmax
from torch_scatter import scatter


class TimeEncodingLayer(nn.Module):

    def __init__(self, time_dim, parameter_requires_grad: bool = True):
        """
        TGAT time encoder, used for encoding time information
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """

        super(TimeEncodingLayer, self).__init__()

        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim // 2)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 5, time_dim // 2, dtype=np.float32))).reshape(time_dim // 2, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim // 2))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, times_slices):
        """
        compute time encodings of time for neighbor_slice
        :param times_slices: Tensor, shape (num_nodes, delta_t_seq),delta_t_seq maybe equal to num_nodes->[N,N]
        :return:
        """
        # Tensor, shape [N,N,1]
        time_slices = times_slices.unsqueeze(dim=2).float()

        # tensor, shape [N,N,F]
        output_cos = torch.cos_(self.w(time_slices))
        output_sin = torch.sin_(self.w(time_slices))
        output = torch.cat([output_cos, output_sin], dim=-1)

        return output

class SparseTimeEncodingLayer(nn.Module):
    def __init__(self, time_dim, parameter_requires_grad: bool = True):
        """
        TGAT time encoder
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """

        super(SparseTimeEncodingLayer, self).__init__()

        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim // 2)
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 5, time_dim // 2, dtype=np.float32))).reshape(time_dim // 2,
                                                                                                     -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim // 2))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, delta_time):
        # tensor, shape(num_nodes, delta_t_seq, time_dim)->[N,N,F]
        output_cos = torch.cos(self.w(delta_time))
        output_sin = torch.sin(self.w(delta_time))
        output = torch.cat([output_cos, output_sin], dim=-1)
        # graph.edge_attr = output

        return output


class TAStructralGATLayer(nn.Module):
    """ TAS module """
    def __init__(self,
                 input_dim,
                 output_dim,
                 time_feature_dim,
                 n_heads,
                 structural_drop,
                 residual):
        super(TAStructralGATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        # parameters for node feature
        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        # parameters for edge weight
        self.att_e = nn.Parameter(torch.Tensor(1, n_heads, 1))  # edge weight scaling factor [1, n_heads, 1]

        if time_feature_dim is not None:
            self.edge_encoder = nn.Linear(time_feature_dim, n_heads * self.out_dim)  # edge attribute linear transformer
        else:
            self.edge_encoder = None

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(structural_drop)
        self.ffd_drop = nn.Dropout(structural_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()


    def forward(self, tuple):
        graph = tuple[0]
        edge_attr = tuple[1]
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.reshape(-1, 1)

        x_j = graph.x[edge_index[0]]  # [E, F] source node
        x_i = graph.x[edge_index[1]]  # [E, F] target node

        H, C = self.n_heads, self.out_dim
        x_i = self.lin(x_i).view(-1, H, C)  # [E,heads,out_dim]
        x_j = self.lin(x_j).view(-1, H, C)  # [E,heads,out_dim]

        # Fuse edge attributes into source node features
        if edge_attr is not None and self.edge_encoder is not None:
            edge_feat = self.edge_encoder(edge_attr).view(-1, H, C)  # [E, heads, out_channels]
            x_j = x_j + edge_feat

        alpha_l = (x_j * self.att_l).sum(dim=-1).squeeze()  # [E,heads]
        alpha_r = (x_i * self.att_r).sum(dim=-1).squeeze()  # [E,heads]
        alpha_node = alpha_r + alpha_l

        alpha_edge = (edge_weight.unsqueeze(-1) * self.att_e).squeeze()  # [E,heads]
        alpha = alpha_node + alpha_edge
        # alpha = edge_weight * alpha_node
        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            # x_i = self.ffd_drop(x_i)  # [num_edges, heads, out_dim]
            x_j = self.ffd_drop(x_j)  # [num_edges, heads, out_dim]

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], edge_index[1], dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            out = out + self.lin_residual(graph.x)
        return out

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)
        nn.init.xavier_uniform_(self.att_e)


class SATemporalGATLayer(nn.Module):
    """ SAT module """
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 temporal_drop,
                 residual):
        super(SATemporalGATLayer, self).__init__()
        self.input_dim = input_dim
        self.out_dim = output_dim // n_heads
        self.n_heads = n_heads
        self.act = nn.ELU()

        # parameters for node feature (time embedding)
        self.lin = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)
        self.att_l = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))
        self.att_r = nn.Parameter(torch.Tensor(1, n_heads, self.out_dim))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.attn_drop = nn.Dropout(temporal_drop)
        self.ffd_drop = nn.Dropout(temporal_drop)

        self.residual = residual
        if self.residual:
            self.lin_residual = nn.Linear(input_dim, n_heads * self.out_dim, bias=False)

        self.xavier_init()

    def forward(self, tuple):
        graph = tuple[0]
        edge_attr = tuple[1]
        edge_index = graph.edge_index
        source_nodes = edge_index[1]
        target_nodes = edge_index[0]

        source_features = edge_attr  # [E, F] source node
        mask = (source_nodes == target_nodes)  # [E]

        filtered_edge_attr = edge_attr[mask]

        # Copy the values of edge_attr only at positions where the mask is True
        target_features = filtered_edge_attr[edge_index[0]]

        H, C = self.n_heads, self.out_dim
        x_i = self.lin(target_features).view(-1, H, C)  # [E,heads,out_dim]
        x_j = self.lin(source_features).view(-1, H, C)  # [E,heads,out_dim]

        alpha_l = (x_j * self.att_l).sum(dim=-1).squeeze()  # [E,heads]
        alpha_r = (x_i * self.att_r).sum(dim=-1).squeeze()  # [E,heads]
        alpha = alpha_r + alpha_l

        alpha = self.leaky_relu(alpha)
        coefficients = softmax(alpha, edge_index[1])  # [num_edges, heads]

        # dropout
        if self.training:
            coefficients = self.attn_drop(coefficients)
            x_j = self.ffd_drop(x_j)  # [num_edges, heads, out_dim]

        # output
        out = self.act(scatter(x_j * coefficients[:, :, None], source_nodes, dim=0, reduce="sum"))
        out = out.reshape(-1, self.n_heads * self.out_dim)  # [num_nodes, output_dim]
        if self.residual:
            x = filtered_edge_attr
            out = out + self.lin_residual(x)
        return out

    def xavier_init(self):
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

class TemporalAttentionLayer(nn.Module):
    """ Historical attention module """
    def __init__(self,
                 input_dim,
                 output_dim,
                 n_heads,
                 num_time_steps,
                 temporal_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        self.temporal_drop = nn.Dropout(temporal_drop)

        # define weights
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, output_dim))

        self.temporal_drop = nn.Dropout(temporal_drop)
        self.xavier_init()

    def forward(self, inputs):
        # 1: Add position embeddings to input
        position_inputs = torch.arange(0, self.num_time_steps-1).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        test = self.position_embeddings[position_inputs]
        temporal_inputs = inputs + test  # [N, T, F]

        # 2: Query, Key based multi-head self attention.
        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))  # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))  # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))  # [N, T, F]

        # 3: Split, concat and scale.
        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)  # [hN, T, F/h]

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))  # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)
        # 4: Masked (causal) softmax to compute attention weights.
        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)  # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs = torch.where(masks == 0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs  # [h*N, T, T]

        # 5: Dropout on attention weights.
        if self.training:
            outputs = self.temporal_drop(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0] / self.n_heads), dim=0),
                            dim=2)  # [N, T, F]

        # 6: Feedforward and residual
        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = torch.cat([outputs, temporal_inputs], dim=2)
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(inputs)
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
