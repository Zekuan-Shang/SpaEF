import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class adj_AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=64, latent_dim=32, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dims // heads, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dims, latent_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATConv):
                m.reset_parameters()

    def encode(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        z = self.dropout(x)
        return z
    
    def decode(self, z):
        A_hat = torch.sigmoid(torch.matmul(z, z.t()))
        return A_hat
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        A_hat = self.decode(z)
        return A_hat, z 
    

class feat_AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), latent_dim=64, heads=(4, 4, 4), dropout=0.2):
        super().__init__()
        
        h1, h2, h3 = heads

        # encoder
        self.gat1 = GATConv(input_dim, hidden_dims[0], heads=h1, concat=True)
        self.lin1 = nn.Linear(hidden_dims[0] * h1, hidden_dims[0])  

        self.gat2 = GATConv(hidden_dims[0], hidden_dims[1], heads=h2, concat=True)
        self.lin2 = nn.Linear(hidden_dims[1] * h2, latent_dim)  

        # decoder
        self.lin3 = nn.Linear(latent_dim, hidden_dims[1] * h3)
        self.gat3 = GATConv(hidden_dims[1] * h3, hidden_dims[0], heads=h2, concat=True)

        self.lin4 = nn.Linear(hidden_dims[0] * h2, hidden_dims[0])
        self.gat4 = GATConv(hidden_dims[0], input_dim, heads=h1, concat=False)

        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, GATConv):
                m.reset_parameters()

    def encode(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.lin1(x))
        x = self.dropout(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.dropout(x)

        z = self.lin2(x)
        return z
    
    def decode(self, z, edge_index):
        x = F.elu(self.lin3(z))
        x = self.dropout(x)

        x = F.elu(self.gat3(x, edge_index))
        x = self.dropout(x)

        x = F.elu(self.lin4(x))
        x = self.dropout(x)

        x = F.relu(self.gat4(x, edge_index))
        return x
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_output = self.decode(z, edge_index)
        return x_output
    

class DenoiseAE(nn.Module):
    def __init__(self, cell_num, gene_num, latent_dim=54):
        super().__init__()
        
        self.CellGraphAE = feat_AutoEncoder(input_dim=gene_num, latent_dim=latent_dim)
        self.GeneGraphAE = feat_AutoEncoder(input_dim=cell_num, latent_dim=latent_dim)

        # zibn layers
        self.zinb_mean_layer = nn.Linear(gene_num, gene_num)
        self.zinb_dispersion_layer = nn.Linear(gene_num, gene_num)
        self.zinb_dropout_layer = nn.Linear(gene_num, gene_num)
        
        # Logits for fusion weights 
        self.fusion_weight_layer = nn.Linear(gene_num, 1)  

        # Init
        for layer in [self.zinb_mean_layer, self.zinb_dispersion_layer, self.zinb_dropout_layer, self.fusion_weight_layer]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, expression_matrix, cell_graph_index, gene_graph_index):
        # Prepare features
        cell_x_in = expression_matrix
        gene_x_in = expression_matrix.T

        # Autoencoder reconstructions
        cell_x = self.CellGraphAE(cell_x_in, cell_graph_index)
        gene_x = self.GeneGraphAE(gene_x_in, gene_graph_index)
        gene_x = gene_x.transpose(0, 1)
        
        # Combine fused reconstruction and adjustment
        fusion_weight = torch.sigmoid(self.fusion_weight_layer(cell_x + gene_x))  # shape: [cell_num, 1]

        output = fusion_weight * cell_x + (1 - fusion_weight) * gene_x  # shape: [cell_num, gene_num]
        
        mean = torch.clamp(F.softplus(self.zinb_mean_layer(output)), min=1e-3, max=1e3)
        dispersion = torch.clamp(F.softplus(self.zinb_dispersion_layer(output)), min=1e-4, max=1e3)
        dropout = torch.clamp(torch.sigmoid(self.zinb_dropout_layer(output)), min=1e-3, max=1 - 1e-3)
        
        return mean, dispersion, dropout


'''
# use case
cell_num = 10000
gene_num = 200
cell_edges = 100000
gene_edges = 40000
lr = 1e-3
epochs = 100

expression_matrix = torch.randn(cell_num, gene_num)
cell_graph_index = torch.randint(0, cell_num, (2, cell_edges), dtype=torch.long)
gene_graph_index = torch.randint(0, gene_num, (2, gene_edges), dtype=torch.long)
gene_graph_attn = F.softmax(torch.randn(gene_num, gene_num), dim=-1)
print(cell_graph_index.shape)
print(cell_graph_index)
'''