import torch
import torch.nn as nn
import dgl


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        )

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, x):
        N, seq_length, embed_size = x.shape
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.heads, self.head_dim)
        keys = keys.view(N, seq_length, self.heads, self.head_dim)
        queries = queries.view(N, seq_length, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, self.heads * self.head_dim)

        return self.fc_out(out)


class Moe(nn.Module):
    def __init__(self, in_feature, num_experts, dropout_rate, hid_dim):
        super(Moe, self).__init__()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.experts = nn.ModuleList([nn.Sequential(
            nn.Linear(in_feature, hid_dim, bias=True), self.activation, self.dropout,
            nn.Linear(hid_dim, hid_dim, bias=True), self.activation, self.dropout,
            nn.Linear(hid_dim, 1, bias=True)
        ) for _ in range(num_experts)])

        self.gating = nn.Sequential(
            nn.Linear(in_feature, hid_dim), self.activation, self.dropout,
            nn.Linear(hid_dim, hid_dim), self.activation, self.dropout,
            nn.Linear(hid_dim, num_experts), nn.Softmax(dim=1)
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
                 dropout_rate, num_experts, moe_hid_dim, num_heads):
        super(dmpnn, self).__init__()
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.edge_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim, edge_output_dim, bias=False),
            self.activation
        )

        self.edge_update_mlp = nn.Sequential(
            nn.Linear(edge_output_dim, edge_output_dim, bias=False)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + edge_output_dim, node_output_dim, bias=True),
            self.activation,
            self.dropout
        )

        self.num_rounds = num_rounds

        self.multi_head_attention = MultiHeadAttention(node_output_dim, num_heads)

        self.moe = Moe(node_output_dim + extra_dim, num_experts, dropout_rate, moe_hid_dim)

    def forward(self, batched_graph, extra_features):
        batched_graph.edata['h0'] = self.initialize_edge_features1(batched_graph)
        batched_graph.edata['h'] = batched_graph.edata['h0']
        self.setup_reverse_edges(batched_graph)

        for _ in range(self.num_rounds):
            self.message_passing1(batched_graph)

        batched_graph.update_all(self.message_func_sum, self.reduce_func_sum)
        new_node_feats1 = torch.cat([batched_graph.ndata['feat'], batched_graph.ndata['m']], dim=1)
        batched_graph.ndata['h'] = self.node_mlp(new_node_feats1)

        attention_output = self.multi_head_attention(batched_graph.ndata['h'].unsqueeze(1))

        features = dgl.readout_nodes(batched_graph, 'h', op='sum')
        combined_features = torch.cat([features, extra_features], dim=-1)
        output = self.moe(combined_features)

        return output

    def initialize_edge_features1(self, g):
        edge_features = torch.cat([g.ndata['feat'][g.edges()[0]], g.edata['feat']], dim=1)
        return self.edge_mlp(edge_features)

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

        weighted_m = self.edge_update_mlp(edges.data['m'])
        edges.data['h'] = self.activation(weighted_m + edges.data['h0'])
        edges.data['h'] = self.dropout(edges.data['h'])
        return {'h': edges.data['h']}

    def message_func_sum(self, edges):
        return {'m2': edges.data['h']}

    def reduce_func_sum(self, nodes):
        return {'m': torch.sum(nodes.mailbox['m2'], dim=1)}