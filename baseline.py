import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import numpy as np
import networkx as nx
import pandas as pd

device = 'cpu'

dataset = TUDataset(root='./data/', name='MUTAG').to(device)
node_feature_dim = 7

# Split into training and validation
rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
train_loader = DataLoader(train_dataset, batch_size=100)


class Baseline():

    def __init__(self, dataloader, max_nodes):
        
        self.dataloader = dataloader
        self.values, self.probs = self.count_nodes()
        self.max_nodes = max_nodes
        
    def count_nodes(self):
        all_node_counts = []

        for data in self.dataloader:

            counts = torch.bincount(data.batch)
            all_node_counts.extend(counts.tolist())
        
        all_node_counts = np.array(all_node_counts)

        values, counts = np.unique(all_node_counts, return_counts=True) # Return the different options for N and the counts
        probs = counts / counts.sum()

        return values, probs
    
    def sample_N(self):
        N = np.random.choice(self.values, size=1, p=self.probs)
        return N

    def compute_link_prob(self):
            
            # we have data.edge_index: the index of two nodes which are connected
            # we have data.batch: the graph index for each node

            # First we find the elements of node_counts which are equal to N. Then we know the index of the graphs we are meant to explore
            # Then we go through data.batch and get the nodes which belong to these graphs
            # Then we count the edges the involves these nodes
            # And we use N to count the number of possible edges and from that value count the number of possible edges
            # At last we can compute the link probability   

            N = self.sample_N()[0]
            possible_edges_per_graph = N * (N - 1) / 2 # Number of edges in a complete graph

            n_edges = 0
            n_possible_edges = 0

            for data in self.dataloader:

                batch = data.batch
                edge_index = data.edge_index

                counts = torch.bincount(batch)

                selected_graphs = (counts == N).nonzero(as_tuple=True)[0] # Get indices of graphs in the batch with N nodes
                num_graphs = len(selected_graphs)

                if num_graphs == 0:
                    continue
                
                node_mask = torch.isin(batch, selected_graphs) # Find the nodes belonging to the selected graphs

                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                num_edges = edge_mask.sum().item() # Count edges which connectes the nodes in the node_mask

                n_edges += num_edges
                n_possible_edges += num_graphs * possible_edges_per_graph

            if n_possible_edges == 0:
                return N, 0.0
            
            return N, n_edges / n_possible_edges

                   
    def erdos_reyi(self, nsamples = 100):

        adj_matrices = []

        for i in range(nsamples):
            N, r = self.compute_link_prob()

            # Generate each possible edge independently with a fixed probability

            # using networkx
            A = np.zeros((self.max_nodes, self.max_nodes))
            G = nx.erdos_renyi_graph(N, r)
            adj = nx.adjacency_matrix(G).toarray()

            # using torch
            # upper = torch.rand((N, N))
            # upper = (upper < r).int()
            # upper = torch.triu(upper, diagonal=1)
            # adj = upper + upper.T

            A[:N, :N] = adj
            adj_matrices.append(A.flatten())
        
        adj_matrices = np.vstack(adj_matrices)

        return adj_matrices
        

# Task 1: Sample the number of nodes N from the emperical distribution of the number of nodes in the training data
# i.e. emperical distribution over the number of nodes each graph has (graph size)
# when generating a sample, randomly pick a value N with probabilities proportional to the height of the histogram

# Task 2: Compute the link probability r  (number of edges divided by total possible number of edges) only using the graphs which have N nodes

# Task 3: Sample a random graph with N nodes and edge probability r according to the Erdos-Renyi model

base = Baseline(train_loader, max_nodes = 28)

sample = base.erdos_reyi(nsamples=2)


df = pd.DataFrame(sample)
output_file = "sampled_graphs_baseline.csv"
df.to_csv(output_file, index=False, header=False)