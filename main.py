import os
import sys
import csv
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch.optim as optim
import torch.nn.functional as F
from model import DenoiseAE, adj_AutoEncoder
from scipy.sparse import csr_matrix
from torch.optim import lr_scheduler
from scipy.stats import spearmanr
import random
import warnings
import multiprocessing
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'Loki'))
from Loki.src.loki import utils
from Loki.src.loki import preprocess
from Loki.src.loki import decompose
warnings.filterwarnings('ignore')
num_cores = multiprocessing.cpu_count() 
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["MKL_NUM_THREADS"] = str(num_cores)
torch.set_num_threads(num_cores)
torch.set_num_interop_threads(num_cores)
from sklearn.linear_model import LogisticRegression


class WorkFlow():
    def __init__(self, root_path, epochs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs # work methods
        self.root_path = root_path
        exp_path    = os.path.join(root_path, "expression.csv")
        coordi_path = os.path.join(root_path, "coordinate.csv")
     
        
        # read data
        self.expression_matrix = pd.read_csv(exp_path, index_col=0)
        self.cell_num, self.gene_num = self.expression_matrix.shape
        self.coordinate_matrix = pd.read_csv(coordi_path, index_col=0)
        if not self.expression_matrix.index.equals(self.coordinate_matrix.index):
            raise ValueError("Spots' order is not match in expression and coordinate files")

        print(f"Expression matrix loaded {self.expression_matrix.shape}")
        print(f"Coordinate matrix loaded {self.coordinate_matrix.shape}")

       
        # load graph
        self.cell_path = os.path.join(root_path, 'cell_graph_index.pt')
        self.gene_path = os.path.join(root_path, 'gene_graph_index.pt')

        if os.path.exists(self.cell_path):
          self.cell_graph_index = torch.load(self.cell_path).to(self.device)
          print(f"Loaded existing cell graph index from {self.cell_path},{self.cell_graph_index.shape}")
        else:
          self.cell_graph(epochs=1000, k=10)  

        if os.path.exists(self.gene_path):
          self.gene_graph_index = torch.load(self.gene_path).to(self.device)
          print(f"Loaded existing gene graph index from {self.gene_path},{self.gene_graph_index.shape}")
        else:
          self.gene_graph()
        
        
        self.work()


    def cell_graph(self, epochs=1000, k=10):
        # load loki 
        model_path = "checkpoint.pt"
        device = self.device
        model, _, tokenizer = utils.load_model(model_path, device)
        model.eval()
        print("Loki model loaded")

        expr_df = self.expression_matrix 
        coords_df = self.coordinate_matrix
        st_adata = sc.AnnData(X=csr_matrix(expr_df.values), obs=coords_df, var=pd.DataFrame(index=expr_df.columns))
        st_adata.obsm['spatial'] = coords_df[['x', 'y']].values 
        h4kp_gene_path = "housekeeping_genes.csv"
        house_keeping = pd.read_csv(h4kp_gene_path, index_col=0)
        st_text_features = preprocess.generate_gene_df(st_adata, house_keeping_genes=house_keeping)
        st_embeddings = utils.encode_text_df(model, tokenizer, st_text_features, 'label', device)
        embeddings_tensor = st_embeddings.to(dtype=torch.float32)

        # reconstruct delaunay cell graph with Loki spot embedding
        from scipy.spatial import Delaunay
        coords = coords_df[['x', 'y']].values
        tri = Delaunay(coords)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(edge)
        edges = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        input_dim = embeddings_tensor.shape[1]
        adj_ae = adj_AutoEncoder(input_dim).to(device)
        optimizer = torch.optim.Adam(adj_ae.parameters(), lr=0.01)
        num_epochs = epochs
        print("Training adjacency autoencoder...")
        for epoch in range(num_epochs):
            adj_ae.train()
            optimizer.zero_grad()
            A_hat, z = adj_ae(embeddings_tensor, edges)
            adj_size = embeddings_tensor.size(0)
            A_real = torch.sparse_coo_tensor(
                edges, 
                torch.ones(edges.size(1), device=device),
                (adj_size, adj_size)
            ).to_dense()
            pos_weight = torch.tensor((adj_size**2 - edges.size(1)) / edges.size(1))
            loss = F.binary_cross_entropy_with_logits(A_hat, A_real, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        adj_ae.eval()
        with torch.no_grad():
            _, z = adj_ae(embeddings_tensor, edges)

        z_norm = F.normalize(z, p=2, dim=1)
        similarity = torch.mm(z_norm, z_norm.t())
        n = similarity.size(0)
        topk_vals, topk_indices = torch.topk(similarity, k=k+1, dim=1)  
        source_nodes = []
        target_nodes = []
        for i in range(n):
            for j in topk_indices[i]:
                if i != j:  
                    if i < j:
                        source_nodes.append(i)
                        target_nodes.append(j)
        
        source_nodes = torch.tensor(source_nodes, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes, dtype=torch.long)
        self.cell_graph_index = torch.stack([source_nodes, target_nodes]).to(self.device)
        
        self.cell_graph_index = torch.stack([source_nodes, target_nodes])
        print(f"Cell graph built with {self.cell_graph_index.size(1)} edges (K={k})")
        torch.save(self.cell_graph_index, self.cell_path)
        print("Cell graph index saved to cell_graph_index.pt")
        
    def gene_graph(self, top_k=10):
        gg_embedding_path = 'GenePT_gene_embedding_ada_text.pickle'
        with open(gg_embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        gg_model_path = "gene_interaction_model.pkl"
        try:
            with open(gg_model_path, 'rb') as f:
                model = pickle.load(f)
        except FileNotFoundError:
            gg_text_labels = 'train_text.txt'
            gg_label_path = 'train_label.txt'
            pairs = []
            with open(gg_text_labels) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        gene1 = parts[0]
                        gene2 = ' '.join(parts[1:])
                        pairs.append((gene1, gene2))
            labels = pd.read_csv(gg_label_path, header=None).squeeze().values
            X, y = [], []
            for (g1, g2), lbl in zip(pairs, labels):
                if g1 in embeddings and g2 in embeddings:
                    X.append(embeddings[g1] + embeddings[g2])
                    y.append(lbl)
            X = np.array(X)
            y = np.array(y)
            model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
            model.fit(X, y)
            with open(gg_model_path, 'wb') as f:
                pickle.dump(model, f)

        genes = [g for g in self.expression_matrix.columns if g in embeddings]
        expr_data = self.expression_matrix[genes].values  
        expr_data_t = expr_data.T
        if np.isnan(expr_data_t).any():
            row_means = np.nanmean(expr_data_t, axis=1, keepdims=True)
            expr_data_t = np.where(np.isnan(expr_data_t), row_means, expr_data_t)
        corr_matrix = np.corrcoef(expr_data_t)
        corr = pd.DataFrame(corr_matrix, index=genes, columns=genes)
        candidate_pairs = set()
        gene_list = list(genes)
        top_indices_per_gene = []
        for i, gene in enumerate(gene_list):
            correlations = corr_matrix[i].copy()
            correlations[i] = -np.inf
            top_indices = np.argpartition(correlations, -top_k)[-top_k:]
            top_indices_per_gene.append(top_indices)
            for j in top_indices:
                if i < j:
                    candidate_pairs.add((gene, gene_list[j]))
                else:
                    candidate_pairs.add((gene_list[j], gene))
        
        candidate_pairs = list(candidate_pairs)
        features = []
        correlations_values = []
        for (g1, g2) in candidate_pairs:
            features.append(embeddings[g1] + embeddings[g2])
            correlations_values.append(corr.loc[g1, g2])
        features = np.vstack(features)
        interaction_probs = model.predict_proba(features)[:, 1]
        results = pd.DataFrame({
            'gene1': [g1 for g1, _ in candidate_pairs],
            'gene2': [g2 for _, g2 in candidate_pairs],
            'correlation': correlations_values,
            'confidence': interaction_probs,
            'interaction': (interaction_probs >= 0.5).astype(int) 
        })
        
        results = results.sort_values('confidence', ascending=False).reset_index(drop=True)
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        source_indices = []
        target_indices = []
        interaction_rows = results[results['interaction'] == 1]
        for _, row in interaction_rows.iterrows():
            idx1 = gene_to_idx[row['gene1']]
            idx2 = gene_to_idx[row['gene2']]
            source_indices.extend([idx1, idx2])
            target_indices.extend([idx2, idx1])
        self.gene_graph_index = torch.tensor( [source_indices, target_indices], dtype=torch.long).to(self.device)
        edge_count = self.gene_graph_index.size(1)
        torch.save(self.gene_graph_index, self.gene_path)
        print(f"Gene graph built with {edge_count} edges (top_k={top_k})")
        print(f"Gene graph index saved to {self.gene_path}")
        return self.gene_graph_index

    def zinb_nll_loss(self, x, mu, theta, pi, eps=1e-8):
        t1 = torch.lgamma(theta + x) - torch.lgamma(x + 1) - torch.lgamma(theta)
        t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
        t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
        nb_case = t1 + t2 + t3

        nb_prob = torch.exp(nb_case)
        zinb_prob = pi + (1. - pi) * nb_prob
        zero_mask = (x < eps).float()
        result = zero_mask * torch.log(zinb_prob + eps) + (1. - zero_mask) * (torch.log(1. - pi + eps) + nb_case)
        result = torch.where(torch.isnan(result), torch.zeros_like(result), result)
        return -torch.mean(result)

    def work(self, lr=1e-4, gamma=1):
        cell_x_in = torch.tensor(self.expression_matrix.values,dtype=torch.float32).to(self.device)
        model = DenoiseAE(self.cell_num, self.gene_num).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        best_loss = float('inf')

        for epoch in range(1, self.epochs+1):
            model.train()
            optimizer.zero_grad()
            mean, disp, pi = model(cell_x_in, self.cell_graph_index, self.gene_graph_index)
            loss = self.zinb_nll_loss(cell_x_in, mean, disp, pi) + 1e-3 * torch.mean(mean**2)
            loss.backward() 
            optimizer.step()
            scheduler.step()
            print(f"Epoch {epoch}/{self.epochs}, Loss: {loss.item():.4f}")
            print(f"Epoch {epoch}, miu mean/max: {mean.mean().item():.2f}/{mean.max().item():.2f}, disp mean/max: {disp.mean().item():.2f}/{disp.max().item():.2f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                df_out = pd.DataFrame(mean.detach().cpu().numpy(),
                                        columns=self.expression_matrix.columns,
                                        index=self.expression_matrix.index)
                df_out.to_csv(os.path.join(self.root_path, f'best_output_{epoch}.csv'))
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=800)
    args = parser.parse_args()
    WorkFlow(args.root_path, args.epochs)

