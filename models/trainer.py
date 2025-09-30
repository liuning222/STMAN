import numpy as np
import torch
import torch_geometric as tg
import scipy.sparse as sp
from torch_geometric.data import Data

from utils.utilities import to_device_graphs
from utils.link_prediction import evaluate_classifier
from utils.preprocess import get_evaluation_data
from models.model import STMAN


class Trainer:
    def __init__(self, graphs, adjs, features, data_snapshots_num, device, args):

        self.graphs = graphs
        self.data_snapshots_num = data_snapshots_num
        self.device = device
        self.args = args

        self.features = [self._preprocess_features(feat) for feat in features]
        self.raw_adjs = adjs
        self.adjs = [self._normalize_graph_gcn(a) for a in self.raw_adjs]  # normalized adj with self loop
        self.pyg_graphs = self._build_pyg_graphs()
        self.delta_times = self._build_delta_time()
        self.model = self._create_model(device)

    def _preprocess_features(self, features):
        """Row-normalize feature matrix and convert to tuple representation"""
        features = np.array(features.todense())
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def _normalize_graph_gcn(self, adj):
        """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

        return adj_normalized

    def _build_pyg_graphs(self):
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index,
                        edge_weight=edge_weight)
            pyg_graphs.append(data)

        return pyg_graphs

    def _build_delta_time(self):
        """preprocess to generate historical co-occurrence time between neighboring nodes from  the first snapshot. The result is organized as edge_attr"""
        delta_times = []
        for t in range(len(self.pyg_graphs)):
            current_graph = self.pyg_graphs[t]
            edge_index = current_graph.edge_index
            num_edges = edge_index.size(1)

            # initialize current co-occurrence time as 1 at the first snapshot
            edge_attr = torch.ones(num_edges, 1)

            # from the second snapshot
            if t > 0:
                prev_graph = self.pyg_graphs[t - 1]
                prev_edge_index = prev_graph.edge_index
                prev_edge_attr = delta_times[t - 1]

                # get previous snapshot edge set using node id
                prev_edges = set()
                for i in range(prev_edge_index.size(1)):
                    src = prev_edge_index[0, i].item()
                    dst = prev_edge_index[1, i].item()
                    prev_edges.add((src, dst))

                # check if each current edge exist in the previous snapshot
                for i in range(num_edges):
                    src = edge_index[0, i].item()
                    dst = edge_index[1, i].item()

                    if (src, dst) in prev_edges:
                        # finding previous edge indec
                        mask = (prev_edge_index[0] == src) & (prev_edge_index[1] == dst)
                        if mask.any():
                            prev_edge_idx = mask.nonzero(as_tuple=True)[0].item()
                            edge_attr[i] = prev_edge_attr[prev_edge_idx] + 1

            delta_times.append(edge_attr)

        return delta_times

    def _create_model(self, device):
        model = STMAN(self.args, self.features[0].shape[1], self.args.time_encoding_dim,
                      self.data_snapshots_num[self.args.dataset]).to(device)
        print("create model!!!")

        return model

    def run(self, device):
        # Load evaluation data for link prediction.
        train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
            test_edges_pos, test_edges_neg = get_evaluation_data(self.graphs)

        print("No. Train: Pos={}, Neg={} \nNo. Val: Pos={}, Neg={} \nNo. Test: Pos={}, Neg={}".format(
            len(train_edges_pos), len(train_edges_neg), len(val_edges_pos), len(val_edges_neg),
            len(test_edges_pos), len(test_edges_neg)))

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay)

        graphs = self.pyg_graphs[:-1]
        adjs = [torch.from_numpy(adj.toarray()) for adj in self.adjs][:-1]
        delta_times = self.delta_times[:-1]
        graphs, adjs, delta_times = to_device_graphs(graphs, adjs, delta_times, device)

        best_epoch_val = 0
        patient = 0
        for epoch in range(0, self.args.epochs):
            self.model.train()
            epoch_loss = []

            opt.zero_grad()
            loss = self.model.get_loss(graphs, adjs, delta_times)

            loss.backward()
            opt.step()
            epoch_loss.append(loss.item())

            del loss
            torch.cuda.empty_cache()

            # lr_decay
            if epoch % self.args.lr_decay_step == 0:
                for param_group in opt.param_groups:
                    param_group['lr'] = self.args.lr_decay_factor * param_group['lr']

            with torch.no_grad():
                self.model.eval()
                _, _, emb = self.model(graphs, delta_times)
                emb = emb[:, -1, :].detach().cpu().numpy()
                val_results, test_results, _, _, train_results, _ = evaluate_classifier(train_edges_pos,
                                                                                        train_edges_neg,
                                                                                        val_edges_pos,
                                                                                        val_edges_neg,
                                                                                        test_edges_pos,
                                                                                        test_edges_neg,
                                                                                        emb,
                                                                                        emb)

                epoch_auc_val = val_results["HAD"][0]
                epoch_auc_test = test_results["HAD"][0]
                epoch_auc_train = train_results["HAD"][0]

                epoch_ap_val = val_results["HAD"][1]
                epoch_ap_test = test_results["HAD"][1]
                epoch_ap_train = train_results["HAD"][1]

            # used for auc metric
            if epoch_auc_val > best_epoch_val:
                best_epoch_val = epoch_auc_val
                torch.save(self.model.state_dict(), "./model_checkpoints/model.pt")
                patient = 0
            else:
                patient += 1
                if patient > self.args.patience:
                    break

            print(
                "Epoch {:<3},  Loss = {:.3f}, Train AUC {:.3f}  Train AP {:.3f} Val AUC {:.3f}  Val AP {:.3f} Test AUC {:.3f}  Test AP {:.3f}".format(
                    epoch,
                    np.mean(epoch_loss),
                    epoch_auc_train, epoch_ap_train,
                    epoch_auc_val, epoch_ap_val,
                    epoch_auc_test, epoch_ap_test))

            # Test Best Model
        self.model.load_state_dict(torch.load("./model_checkpoints/model.pt"))
        with torch.no_grad():
            self.model.eval()
            _, _, emb = self.model(graphs, delta_times)
            emb = emb[:, -1, :].detach().cpu().numpy()
            val_results, test_results, _, _, train_results, _ = evaluate_classifier(train_edges_pos,
                                                                                    train_edges_neg,
                                                                                    val_edges_pos,
                                                                                    val_edges_neg,
                                                                                    test_edges_pos,
                                                                                    test_edges_neg,
                                                                                    emb,
                                                                                    emb)

            auc_test = test_results["HAD"][0]
            ap_test = test_results["HAD"][1]

        return auc_test, ap_test
