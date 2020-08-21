import torch
from torch import nn
import torch.nn.functional as F

from utils.utils import MergeLayer


class RelativeAttention(nn.Module):
    def __init__(self, embed_dim, kdim, vdim, num_heads, dropout):
        super().__init__()
        assert embed_dim == kdim == vdim

        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.scale = 1 / (self.d_k ** 0.5)

        self.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(3)])
        self.r_layer = nn.Linear(embed_dim, embed_dim)
        # self.r_bias = nn.Parameter(torch.FloatTensor(1, self.h, 1, self.d_k))
        self.r_bias = nn.Parameter(torch.rand(1, self.h, 1, self.d_k))
        self.output_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, r, mask=None):
        # query : B x Q x H
        # key : B x K x H
        # value : B x K x H
        # r : B x Q x K x H
        # mask : B x K

        batch_size, Q = query.size(0), query.size(1)
        K = key.size(1)

        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]  # B x n x (Q or K) x d

        r = self.r_layer(r)  # B x Q x K x H
        r = r.view(batch_size, Q, K, self.h, self.d_k)  # B x T x T x n x d
        r = r.permute(0, 3, 1, 2, 4)  # B x n x Q x K x d

        x, attn = self.attention(query, key, value, r, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        x = self.output_linear(x)
        return x

    def attention(self, query, key, value, r, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1))  # B x n x Q x K
        scores += torch.einsum('bnqd,bnqkd->bnqk', query + self.r_bias, r)  # B x n x Q x K
        scores = scores * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x K
            scores = scores.masked_fill(mask == 1, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class TemporalRelativeAttentionLayer(torch.nn.Module):
    """
    Temporal attention layer. Return the temporal embedding of a node given the node itself,
     its neighbors and the edge timestamps.
    """

    def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
                 output_dimension, n_head=2,
                 dropout=0.1):
        super(TemporalRelativeAttentionLayer, self).__init__()

        self.n_head = n_head

        self.feat_dim = n_node_features
        self.time_dim = time_dim

        # self.query_dim = n_node_features + time_dim
        # self.key_dim = n_neighbors_features + time_dim + n_edge_features
        # self.key_dim = n_neighbors_features + time_dim
        self.query_dim = self.key_dim = n_node_features

        self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

        self.multi_head_target = RelativeAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features, neighbors_padding_mask):
        """
        "Temporal attention model
        :param src_node_features: float Tensor of shape [batch_size, n_node_features]
        :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
        :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
        :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
        time_dim]
        :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
        :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
        :return:
        attn_output: float Tensor of shape [1, batch_size, n_node_features]
        attn_output_weights: [batch_size, 1, n_neighbors]
        """

        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

        # query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        # key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)
        # key = torch.cat([neighbors_features, neighbors_time_features], dim=2)
        query = src_node_features_unrolled
        key = neighbors_features
        r = neighbors_time_features

        # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
        # Reshape tensors so to expected shape by multi head attention
        # query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
        # key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]
        # Keep it B x (Q or K) x H

        # Compute mask of which source nodes have no valid neighbors
        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
        # force the attention to just 'attend' on this neighbor (which has the same features as all
        # the others since they are fake neighbors) and will produce an equivalent result to the
        # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
        # neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False
        # don't

        # print(query.shape, key.shape)

        # r = None
        # attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key, r=r,
        #                                                           mask=neighbors_padding_mask)
        attn_output = self.multi_head_target(query=query, key=key, value=key, r=r,
                                             mask=neighbors_padding_mask)

        # mask = torch.unsqueeze(neighbors_padding_mask, dim=2)  # mask [B, N, 1]
        # mask = mask.permute([0, 2, 1])
        # attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
        #                                                           mask=mask)

        attn_output = attn_output.squeeze()
        # attn_output_weights = attn_output_weights.squeeze()

        # Source nodes with no neighbors have an all zero attention output. The attention output is
        # then added or concatenated to the original source node features and then fed into an MLP.
        # This means that an all zero vector is not used.
        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
        # attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

        # Skip connection with temporal attention over neighborhood and the features of the node itself
        attn_output = self.merger(attn_output, src_node_features)
        attn_output_weights = None
        return attn_output, attn_output_weights
