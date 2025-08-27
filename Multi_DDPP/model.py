import torch
import torch.nn as nn
import dgl

class Moe(nn.Module):
    def __init__(self, in_feature, num_experts, dropout_rate, hid_dim):
        super(Moe, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_feature, hid_dim, bias=True),
                self.activation,
                self.dropout,
                nn.Linear(hid_dim, hid_dim, bias=True),
                self.activation,
                self.dropout,
                nn.Linear(hid_dim, 1, bias=True)
            ) for _ in range(num_experts)
        ])

        self.gating = nn.Sequential(
            nn.Linear(in_feature, hid_dim),
            self.activation,
            self.dropout,
            nn.Linear(hid_dim, hid_dim),
            self.activation,
            self.dropout,
            nn.Linear(hid_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.dropout(x)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        gating_outputs = self.gating(x)

        final_outputs = torch.sum(expert_outputs * gating_outputs.unsqueeze(-1), dim=1).reshape(-1, 1)

        return final_outputs


class dmpnn(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, edge_output_dim, node_output_dim, extra_dim, num_rounds,
                 dropout_rate, num_experts, moe_hid_dim):
        super(dmpnn, self).__init__()
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)


        self.edge_mlp_solute = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, edge_output_dim, bias=False),
            self.activation
        )

        self.edge_update_mlp_solute = nn.Sequential(
            nn.Linear(edge_output_dim, edge_output_dim, bias=False)
        )

        self.node_mlp_solute = nn.Sequential(
            nn.Linear(node_feat_dim + edge_output_dim, node_output_dim, bias=True),
            self.activation,
            self.dropout
        )

        self.num_rounds = num_rounds


        self.moe = Moe(node_output_dim + extra_dim, num_experts, dropout_rate, moe_hid_dim)


        self.final_output_layer = nn.Linear(1, 1)  

    def forward(self, batched_graph, extra_features):
        # 初始化边特征
        batched_graph.edata['h0'] = self.initialize_edge_features1(batched_graph)
        batched_graph.edata['h'] = batched_graph.edata['h0']
        self.setup_reverse_edges(batched_graph)


        for _ in range(self.num_rounds):
            self.message_passing1(batched_graph)


        batched_graph.update_all(self.message_func_sum, self.reduce_func_sum)
        new_node_feats1 = torch.cat([batched_graph.ndata['feat'], batched_graph.ndata['m']], dim=1)
        batched_graph.ndata['h'] = self.node_mlp_solute(new_node_feats1)


        solute_features = dgl.readout_nodes(batched_graph, 'h', op='sum')
        combined_features = torch.cat([solute_features, extra_features], dim=-1)


        combined_features = combined_features.view(combined_features.size(0), -1)


        output = self.moe(combined_features)


        logits = self.final_output_layer(output)
        return logits

    def initialize_edge_features1(self, g):
        edge_features = torch.cat([g.ndata['feat'][g.edges()[0]], g.edata['feat']], dim=1)
        return self.edge_mlp_solute(edge_features)

    def setup_reverse_edges(self, g):
        src, dst = g.edges()
        g.edata['reverse_edge'] = g.edge_ids(dst, src)

    def message_passing1(self, g):
        g.update_all(self.message_func, self.reduce_func)
        g.apply_edges(self.apply_edges_func1)

    def message_func(self, edges):
        return {'m': edges.data['h']}

    def reduce_func(self, nodes):
        return {'sum0': torch.sum(nodes.mailbox['m'], dim=1)}

    def apply_edges_func1(self, edges):
        edges.data['sum'] = edges.src['sum0']
        edges.data['m'] = edges.data['sum'] - edges.data['h'][edges.data['reverse_edge']]

        weighted_m = self.edge_update_mlp_solute(edges.data['m'])
        edges.data['h'] = self.activation(weighted_m + edges.data['h0'])
        edges.data['h'] = self.dropout(edges.data['h'])
        return {'h': edges.data['h']}

    def message_func_sum(self, edges):
        return {'m2': edges.data['h']}

    def reduce_func_sum(self, nodes):
        return {'m': torch.sum(nodes.mailbox['m2'], dim=1)}
