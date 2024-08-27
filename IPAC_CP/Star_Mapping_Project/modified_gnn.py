import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx

# Define the Modified GNN Model
class ModifiedGNN(nn.Module):
    def __init__(self, num_node_features, num_classes, num_heads=4):
        super(ModifiedGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=128, num_heads=num_heads, batch_first=True)
            for _ in range(num_heads)
        ])
        self.fc1 = nn.Linear(128 * num_heads, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        # Apply attention mechanisms
        attn_outputs = []
        for attn_head in self.attention_heads:
            x_attn, _ = attn_head(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
            attn_outputs.append(x_attn.squeeze(0))

        # Concatenate the attention outputs
        x = torch.cat(attn_outputs, dim=-1)
        x = global_mean_pool(x, batch)  # Pooling over all nodes

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Function to create GNN data from a graph
def create_gnn_data(graph):
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    node_features = torch.eye(graph.number_of_nodes())  # Example feature, identity matrix
    data = Data(x=node_features, edge_index=edge_index)
    return data

# Function to train the model
def train(model, loader, optimizer, criterion):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

# Example usage
def main():
    G = nx.read_edgelist('graphs/example_graph.edgelist', nodetype=int, data=(('weight', float),))
    gnn_data = create_gnn_data(G)
    
    model = ModifiedGNN(num_node_features=G.number_of_nodes(), num_classes=2)  # Assuming binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader([gnn_data], batch_size=1, shuffle=True)
    for epoch in range(200):
        train(model, loader, optimizer, criterion)

    # Evaluate
    model.eval()
    with torch.no_grad():
        for data in loader:
            out = model(data)
            _, pred = out.max(dim=1)
            print(f"Prediction: {pred.item()}")

if __name__ == "__main__":
    main()
