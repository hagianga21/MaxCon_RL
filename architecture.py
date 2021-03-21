import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Dropout, BatchNorm1d as BN
from torch_geometric.nn import GCNConv, DynamicEdgeConv, global_max_pool
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops, degree

#************************************************
# DGCNN
#************************************************
def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class DGCNN_Net(torch.nn.Module):
    def __init__(self, dimension, out_channels = 1, k=10, aggr='max'):
        super(DGCNN_Net, self).__init__()
        self.conv1 = DynamicEdgeConv(MLP([2 * dimension, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)
        self.lin1 = MLP([3 * 64, 1024])
        self.mlp = Seq(MLP([1216, 256]), MLP([256, 128]), Lin(128, out_channels))

    def forward(self, data):
        x, batch = data.x.float(), data.batch
        batch_size = data.th.size(0)
        #print(x.shape)

        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x4 = self.lin1(torch.cat([x1, x2, x3], dim=1))
        #print(x4.shape)

        x5 = global_max_pool(x4, batch).repeat(1, int(x.size(0)/batch_size)).view(-1, 1024)
        #print(x5.shape)
        
        x6 = self.mlp(torch.cat([x1, x2, x3, x5], dim=1))
        #print(x6.shape)
        
        return x6

#************************************************
# MPNN
#************************************************
class EdgeAndNodeEmbeddingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeAndNodeEmbeddingLayer, self).__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels-1, bias=False)
        self.lin2 = torch.nn.Linear(out_channels, out_channels, bias=False)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv = deg.pow(-1).unsqueeze(1)
        x = F.relu(self.lin1(x))
        y = deg_inv*self.propagate(edge_index, x=x)
        y = torch.cat((y, (deg/deg.max()).unsqueeze(1)), axis = 1)
        y = F.relu(self.lin2(y))
        return y

    def message(self, x_j):
        return x_j

class UpdateNodeEmbeddingLayer(MessagePassing):
    def __init__(self, n_features):
        super(UpdateNodeEmbeddingLayer, self).__init__(aggr='add')
        self.message_layer = nn.Linear(2*n_features, n_features, bias=False)
        self.update_layer = nn.Linear(2*n_features, n_features, bias=False)

    def forward(self, current_node_embeddings, edge_embeddings, edge_index):
        row, col = edge_index
        deg = degree(col, current_node_embeddings.size(0), dtype=current_node_embeddings.dtype)
        deg_inv = deg.pow(-1).unsqueeze(1)
        node_embeddings_aggregated = deg_inv*self.propagate(edge_index, x=current_node_embeddings)

        message = F.relu(self.message_layer(torch.cat([node_embeddings_aggregated, edge_embeddings], dim=-1)))
        new_node_embeddings = F.relu(self.update_layer(torch.cat([current_node_embeddings, message], dim=-1)))
        return new_node_embeddings

    def message(self, x_j):
        return x_j

class ReadoutLayer(MessagePassing):
    def __init__(self, n_features):
        super(ReadoutLayer, self).__init__(aggr='add')
        self.n_features = n_features
        self.layer_pooled = nn.Linear(n_features, n_features, bias=False)
        self.layers_readout = nn.Linear(2*n_features, 1, bias=True)
    
    def forward(self, node_embeddings, batch, batch_size):
        f_local = node_embeddings
        global_pool = global_mean_pool(node_embeddings, batch).repeat(1, int(node_embeddings.size(0)/batch_size)).view(-1, self.n_features)
        f_pooled = self.layer_pooled(global_pool)
        features = F.relu(torch.cat([f_pooled, f_local], dim=-1))
        out = self.layers_readout(features)
        return out


class MPNN(torch.nn.Module):
    def __init__(self, n_obs_in = 4, n_layers=3, n_features = 64):
        super(MPNN, self).__init__()
        self.n_layers = n_layers
        self.node_init_embedding_layer = nn.Sequential(
            nn.Linear(n_obs_in, n_features, bias=False),
            nn.ReLU()
        )
        self.edge_embedding_layer = EdgeAndNodeEmbeddingLayer(n_obs_in, n_features)
        #Updating
        self.update_node_embedding_layer = nn.ModuleList([UpdateNodeEmbeddingLayer(n_features) for _ in range(self.n_layers)])
        self.readout_layer = ReadoutLayer(n_features)

    def forward(self, data):
        x, batch = data.x.float(), data.batch
        edge_index = data.edge_index
        batch_size = data.th.size(0)
        current_node_embeddings = self.node_init_embedding_layer(x)
        edge_embeddings = self.edge_embedding_layer(x, edge_index)

        for i in range(self.n_layers):
            current_node_embeddings = self.update_node_embedding_layer[i](current_node_embeddings, edge_embeddings, edge_index)

        Q_value = self.readout_layer(current_node_embeddings, batch, batch_size)
        return Q_value