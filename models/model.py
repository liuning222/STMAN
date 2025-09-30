import torch
import torch.nn as nn
from models.layers import TimeEncodingLayer, SparseTimeEncodingLayer, TAStructralGATLayer, SATemporalGATLayer, \
    TemporalAttentionLayer
from utils.loss import mutual_loss, consistency_loss, independence_loss


class STMAN(nn.Module):
    def __init__(self, args, node_feature_dim, time_feature_dim, time_length):
        """[summary]
        Args:
            args ([type]): [description]
            node_feature_dim (int) :
            time_length (int): Total timesteps in dataset.
        """
        super(STMAN, self).__init__()
        self.args = args

        self.num_time_steps = time_length

        self.node_feature_dim = node_feature_dim
        self.time_feature_dim = time_feature_dim

        self.num_ta_structural_heads = list(map(int, args.num_ta_structural_heads.split(",")))
        self.ta_structural_layer_dims = list(map(int, args.ta_structural_layer_dims.split(",")))
        self.ta_structural_drop = args.ta_structural_drop

        self.num_sa_temporal_heads = list(map(int, args.num_sa_temporal_heads.split(",")))
        self.sa_temporal_layer_dims = list(map(int, args.sa_temporal_layer_dims.split(",")))
        self.sa_temporal_drop = args.sa_temporal_drop

        self.num_temporal_heads = list(map(int, args.num_temporal_heads.split(",")))
        self.temporal_layer_dims = list(map(int, args.temporal_layer_dims.split(",")))
        self.temporal_drop = args.temporal_drop

        self.time_encoder, self.ta_structural_gat, self.struct_temporal_attn, self.sa_temporal_gat, self.time_temporal_attn = self.build_model()

    def forward(self, graphs, delta_times):
        # Time encoding forward
        time_encoding_out = []  # list of [Ni, Ni, F]
        for t in range(0, self.num_time_steps - 1):
            time_encoding_out.append(self.time_encoder(delta_times[t]))

        # Time Aware Structural Attention forward
        ta_structural_outs = []  # list of [Ni, F]
        for t in range(0, self.num_time_steps - 1):
            ta_structural_outs.append(self.ta_structural_gat([graphs[t], time_encoding_out[t]]))
        ta_structural_outs_ext = [out[:, None, :] for out in ta_structural_outs]  # list of [Ni, 1, F]

        # padding outputs along with Ni
        maximum_node_num = ta_structural_outs_ext[-1].shape[0]
        out_dim = ta_structural_outs_ext[-1].shape[-1]
        ta_structural_outs_padded = []
        for out in ta_structural_outs_ext:
            ta_structural_zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, out_dim).to(out.device)
            padded = torch.cat((out, ta_structural_zero_padding), dim=0)
            ta_structural_outs_padded.append(padded)
        ta_structural_outs_padded = torch.cat(ta_structural_outs_padded, dim=1)  # [N, T, F]

        # Struct-Temporal Attention forward
        struct_temporal_out = self.struct_temporal_attn(ta_structural_outs_padded)

        # Structural Aware Time Attention forward
        sa_temporal_out = []  # list of [Ni,F]
        for t in range(0, self.num_time_steps - 1):
            sa_temporal_out.append(self.sa_temporal_gat([graphs[t], time_encoding_out[t]]))
        sa_temporal_out_ext = [out[:, None, :] for out in sa_temporal_out]  # list of [Ni,1,F]

        # padding neighbor_time_out along with [Ni]
        sa_temporal_out_dim = sa_temporal_out_ext[-1].shape[-1]
        sa_temporal_out_padded = []
        for out in sa_temporal_out_ext:
            sa_temporal_zero_padding = torch.zeros(maximum_node_num - out.shape[0], 1, sa_temporal_out_dim).to(
                out.device)
            padded = torch.cat((out, sa_temporal_zero_padding), dim=0)
            sa_temporal_out_padded.append(padded)
        sa_temporal_out_padded = torch.cat(sa_temporal_out_padded, dim=1)  # [N,T,F]

        # Temporal Attention forward
        time_temporal_out = self.time_temporal_attn(sa_temporal_out_padded)

        final_out = torch.cat([struct_temporal_out, time_temporal_out], dim=-1)
        return struct_temporal_out, time_temporal_out, final_out

    def build_model(self):
        time_feature_dim = self.time_feature_dim
        structural_input_dim = self.node_feature_dim

        # 1: Time encoding Layer, output used for structural attention layer and neighbor time attention layer
        time_encoding_layer = SparseTimeEncodingLayer(time_dim=time_feature_dim)

        # 2: Time Aware Structural attention Layers. Produce input for struct-temporal attention layer
        ta_structural_gat_layers = nn.Sequential()
        for i in range(len(self.ta_structural_layer_dims)):
            layer = TAStructralGATLayer(input_dim=structural_input_dim,
                                        output_dim=self.ta_structural_layer_dims[i],
                                        time_feature_dim=time_feature_dim,
                                        n_heads=self.num_ta_structural_heads[i],
                                        structural_drop=self.ta_structural_drop,
                                        residual=self.args.residual)
            ta_structural_gat_layers.add_module(name="ta_structural_gat_layer_{}".format(i), module=layer)
            structural_input_dim = self.ta_structural_layer_dims[i]

        # 3: struct-historical Attention Layers
        struct_temporal_input_dim = self.ta_structural_layer_dims[-1]
        struct_temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_dims)):
            layer = TemporalAttentionLayer(input_dim=struct_temporal_input_dim,
                                           output_dim=self.temporal_layer_dims[i],
                                           n_heads=self.num_temporal_heads[i],
                                           num_time_steps=self.num_time_steps,
                                           temporal_drop=self.temporal_drop,
                                           residual=self.args.residual)
            struct_temporal_attention_layers.add_module(name="struct_temporal_layer_{}".format(i), module=layer)
            struct_temporal_input_dim = self.temporal_layer_dims[i]

        # 4: Structural Aware Time attention Layers. Produce input for time-temporal attention layer
        sa_temporal_gat_layers = nn.Sequential()
        for i in range(len(self.sa_temporal_layer_dims)):
            layer = SATemporalGATLayer(input_dim=time_feature_dim,
                                       output_dim=self.sa_temporal_layer_dims[i],
                                       n_heads=self.num_sa_temporal_heads[i],
                                       temporal_drop=self.sa_temporal_drop,
                                       residual=self.args.residual)
            sa_temporal_gat_layers.add_module(name="sa_temporal_gat_layers_{}".format(i), module=layer)
            time_feature_dim = self.sa_temporal_layer_dims[i]

        # 5: time-Temporal Attention Layers
        time_temporal_input_dim = self.sa_temporal_layer_dims[-1]
        time_temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_dims)):
            layer = TemporalAttentionLayer(input_dim=time_temporal_input_dim,
                                           output_dim=self.temporal_layer_dims[i],
                                           n_heads=self.num_temporal_heads[i],
                                           num_time_steps=self.num_time_steps,
                                           temporal_drop=self.temporal_drop,
                                           residual=self.args.residual)
            time_temporal_attention_layers.add_module(name="time_temporal_layer_{}".format(i), module=layer)
            time_temporal_input_dim = self.temporal_layer_dims[i]

        return time_encoding_layer, ta_structural_gat_layers, struct_temporal_attention_layers, sa_temporal_gat_layers, time_temporal_attention_layers

    def get_loss(self, graphs, adjs, delta_times):
        # run gnn
        final_structural, final_temporal, final_emb = self.forward(graphs, delta_times)  # [N, T-1, F]
        self.graph_loss = 0
        ## multi-step prediction
        # for t in range(self.num_time_steps - 6):

        # single step prediction
        for t in range(self.num_time_steps - 1):
            emb_t_structural, emb_t_temporal, emb_t = final_structural[:, t, :].squeeze(), final_temporal[:, t,
                                                                                           :].squeeze(), final_emb[:, t,
                                                                                                         :].squeeze()  # [N, F]
            adj = adjs[t]
            mul_loss = mutual_loss(emb_t, adj, self.args.adj_order)
            consist_loss = consistency_loss(emb_t_structural, emb_t_temporal)
            indenp_loss = independence_loss(emb_t_structural, emb_t_temporal)
            graphloss = mul_loss + self.args.consist_weight * consist_loss + self.args.indenp_weight * indenp_loss

            self.graph_loss += graphloss
        return self.graph_loss
