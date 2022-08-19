import torch
from torch.nn.functional import leaky_relu, dropout
from torch.nn import Linear, ReLU, BatchNorm1d, ModuleList, L1Loss, LeakyReLU, Dropout, MSELoss,SELU
from torch_geometric.nn import Sequential, SAGEConv, Linear, to_hetero,HeteroConv,GATConv,GINEConv


class GNNEncoder(torch.nn.Module):
    def __init__(self,hidden_channels,out_channels,num_layers):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        metadata = (['node1', 'node2', 'node3'],
                    [('node2', 'to', 'node3'),
                     ('node1', 'to', 'node2'),
                     ('node3', 'rev_to', 'node2'),
                     ('node2', 'rev_to', 'node1')])
        self.convs = torch.nn.ModuleList()
        layers = (self.hidden_channels, self.out_channels)
        for i in range(self.num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), layers[i])
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: leaky_relu(x) for key, x in x_dict.items()}

        return x_dict


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels,out_channels):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.network = torch.nn.Sequential(
            Linear(-1, hidden_channels),
            SELU(),
            Dropout(p=0.2),
            Linear(hidden_channels, out_channels),
            SELU(),
            Dropout(p=0.2),
            Linear(out_channels, 32),
            SELU(),
            Dropout(p=0.2),
            Linear(32, 1)
        )



    def forward(self, z_dict, edge_index_dict,edge_attr_dict):
        ship, cust = edge_index_dict[('node1', 'to', 'node2')]
        cust,prod = edge_index_dict[('node2', 'to', 'node3')]
        attr1 = edge_attr_dict[('node1', 'to', 'node2')]
        attr2 = edge_attr_dict[('node2', 'to', 'node3')]
        # Concatenate the embeddings and edge attributed
        z1 = leaky_relu(z_dict['node1'][ship])
        z2 = leaky_relu(z_dict['node2'][cust])
        z3 = leaky_relu(z_dict['node3'][prod])
        z4 = leaky_relu(torch.cat([z1,z2,z3,attr1,attr2], dim=-1))
        z4 = self.network(z4)

        return z4.view(-1)