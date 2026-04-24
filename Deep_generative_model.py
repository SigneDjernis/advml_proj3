import torch
import torch.nn as nn
import torch.distributions as td
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# --- 1. Probabilistic Encoder ---
class GNN_Encoder(nn.Module):
    def __init__(self, node_feature_dim, state_dim, latent_dim, num_rounds):
        super().__init__()
        self.state_dim = state_dim
        self.num_rounds = num_rounds
        self.input_net = nn.Sequential(nn.Linear(node_feature_dim, state_dim), nn.ReLU())
        self.message_nets = nn.ModuleList([nn.Linear(state_dim, state_dim) for _ in range(num_rounds)])
        self.update_nets = nn.ModuleList([nn.Linear(state_dim, state_dim) for _ in range(num_rounds)])
        
        self.mu_head = nn.Linear(state_dim, latent_dim)
        self.logvar_head = nn.Linear(state_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        h = self.input_net(x)
        for r in range(self.num_rounds):
            m = self.message_nets[r](h)
            agg = torch.zeros(h.size(0), self.state_dim, device=x.device)
            agg.index_add_(0, edge_index[1], m[edge_index[0]])
            h = torch.relu(h + self.update_nets[r](agg))
        
        num_graphs = batch.max() + 1
        graph_state = torch.zeros(num_graphs, self.state_dim, device=x.device)
        graph_state.index_add_(0, batch, h)
        return self.mu_head(graph_state), self.logvar_head(graph_state)

# --- 2. Decoder ---
class MLP_Decoder(nn.Module):
    def __init__(self, latent_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, max_nodes * max_nodes)
        )

    def forward(self, z):
        logits = self.mlp(z).view(-1, self.max_nodes, self.max_nodes)
        return logits

# --- 3. VAE Wrapper ---
class GraphVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encoder(data.x, data.edge_index, data.batch)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        adj_target = to_dense_adj(data.edge_index, data.batch, max_num_nodes=self.decoder.max_nodes)
        recon_loss = nn.functional.binary_cross_entropy_with_logits(logits, adj_target, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.01 * kl_loss, logits, mu

# --- 4. Main Functionality ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'])
    parser.add_argument('--num_samples', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TUDataset(root='./data/', name='MUTAG')
    max_nodes = max([d.num_nodes for d in dataset])
    latent_dim = 2 # Keep 2 for easy viz
    
    model = GraphVAE(
        GNN_Encoder(dataset.num_features, 32, latent_dim, 4),
        MLP_Decoder(latent_dim, max_nodes)
    ).to(device)

    model_path = "graph_vae.pt"

    if args.mode == 'train':
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        plt.ion()
        fig, ax = plt.subplots(figsize=(6, 5))

        for epoch in range(1, 501):
            model.train()
            total_loss = 0
            all_mu, all_y = [], []

            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                loss, _, mu = model(data)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                all_mu.append(mu.detach().cpu().numpy())
                all_y.append(data.y.detach().cpu().numpy())
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
                ax.clear()
                mus = np.concatenate(all_mu)
                ys = np.concatenate(all_y)
                ax.scatter(mus[:, 0], mus[:, 1], c=ys, cmap='tab10', alpha=0.6)
                ax.set_title(f"Latent Space (Epoch {epoch})")
                plt.pause(0.1)
        
        torch.save(model.state_dict(), model_path)
        plt.ioff()
        plt.show()

    elif args.mode == 'sample':
        if not os.path.exists(model_path):
            print(f"Error: No trained model found at {model_path}. Please train first.")
            return
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Sampling {args.num_samples} graphs...")

        with torch.no_grad():
            # 1. Sample latent vectors from the prior P(z) ~ N(0, I)
            z = torch.randn(args.num_samples, latent_dim).to(device)
            
            # 2. Decode latents into Adjacency Matrix Logits
            logits = model.decoder(z)
            
            # 3. Convert to probabilities and then to binary (0 or 1)
            # We use a 0.5 threshold to decide if an edge exists
            probs = torch.sigmoid(logits)
            adj_binary = (probs > 0.5).float() # Shape: [num_samples, max_nodes, max_nodes]

            # 4. Flatten each adjacency matrix to a vector
            # Shape becomes: [num_samples, max_nodes * max_nodes]
            adj_flattened = adj_binary.view(args.num_samples, -1).cpu().numpy()

            # 5. Save to CSV
            import pandas as pd
            df = pd.DataFrame(adj_flattened)
            output_file = "sampled_graphs.csv"
            df.to_csv(output_file, index=False, header=False)
            
            print(f"Success! Saved {args.num_samples} flattened adjacency matrices to {output_file}")
            print(f"CSV Shape: {df.shape} (Rows: Samples, Columns: Flattened Nodes^2)")

if __name__ == "__main__":
    main()