import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple


class GCN(nn.Module):
    """Graph Convolutional Network for learning node embeddings."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Initialize the GCN model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output embeddings
            dropout: Dropout probability
        """
        super(GCN, self).__init__()

        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass of the GCN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edge indices [2, num_edges]
            edge_weight: Optional edge weights [num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # First graph convolution
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.dropout(x)

        # Second graph convolution
        x = self.conv2(x, edge_index, edge_weight)

        # Output embeddings
        return x


class GNNModule:
    """
    GNN module that learns node embeddings for the knowledge graph.
    Uses a Graph Convolutional Network (GCN) to generate embeddings.
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 64,
        learning_rate: float = 0.01,
        device: Optional[str] = None
    ):
        """
        Initialize the GNN module.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output embeddings
            learning_rate: Learning rate for optimizer
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate

        # Determine device
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model, optimizer, and node mappings
        self.model = GCN(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate)

        # Node ID to index mapping
        self.node_mapping = {}
        self.reverse_mapping = {}

        # Embedding cache
        self.embeddings = None

    def _prepare_pyg_data(self, graph: nx.Graph) -> Data:
        """
        Convert a networkx graph to a PyTorch Geometric Data object.

        Args:
            graph: NetworkX graph

        Returns:
            PyTorch Geometric Data object
        """
        # Reset node mappings
        self.node_mapping = {}
        self.reverse_mapping = {}

        # Map each node ID to a numeric index
        for i, node_id in enumerate(graph.nodes()):
            self.node_mapping[node_id] = i
            self.reverse_mapping[i] = node_id

        # Prepare edge index tensor
        edges = list(graph.edges())
        if not edges:
            # Handle case with no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros(0, dtype=torch.float)
        else:
            source_nodes = [self.node_mapping[e[0]] for e in edges]
            target_nodes = [self.node_mapping[e[1]] for e in edges]

            # Bidirectional edges for undirected graph
            edge_index = torch.tensor([
                source_nodes + target_nodes,
                target_nodes + source_nodes
            ], dtype=torch.long)

            # Edge weights
            weights = [graph.get_edge_data(
                *e).get('weight', 1.0) for e in edges]
            edge_weight = torch.tensor(weights + weights, dtype=torch.float)

        # Initialize random features if not present
        num_nodes = len(graph.nodes())
        x = torch.randn(num_nodes, self.input_dim)

        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_weight
        )

        return data.to(self.device)

    def train(self, graph: nx.Graph, epochs: int = 100) -> Dict[str, float]:
        """
        Train the GNN model on the given graph.

        Args:
            graph: NetworkX graph
            epochs: Number of training epochs

        Returns:
            Dictionary with training statistics
        """
        # Convert graph to PyG format
        data = self._prepare_pyg_data(graph)

        # Ensure we have at least one edge to train on
        if data.edge_index.shape[1] == 0:
            print("Warning: Graph has no edges. Skipping training.")
            # Initialize random embeddings instead
            self.embeddings = torch.randn(len(graph.nodes()), self.output_dim)
            return {"loss": 0.0}

        # Training loop
        self.model.train()
        losses = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()

            # Forward pass
            embeddings = self.model(data.x, data.edge_index, data.edge_attr)

            # Define loss: similarity of connected nodes should be high
            src, dst = data.edge_index
            pos_score = torch.sum(embeddings[src] * embeddings[dst], dim=1)

            # Negative sampling: random node pairs not connected by edges
            neg_src = src[torch.randperm(src.size(0))]
            neg_dst = dst[torch.randperm(dst.size(0))]
            neg_score = torch.sum(
                embeddings[neg_src] * embeddings[neg_dst], dim=1)

            # Contrastive loss: maximize pos_score, minimize neg_score
            loss = F.margin_ranking_loss(
                pos_score, neg_score,
                torch.ones_like(pos_score),
                margin=0.5
            )

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # Cache the final embeddings
        self.model.eval()
        with torch.no_grad():
            self.embeddings = self.model(
                data.x, data.edge_index, data.edge_attr).cpu()

        return {"loss": np.mean(losses)}

    def get_node_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Get embeddings for all nodes in the graph.

        Returns:
            Dictionary mapping node IDs to embedding arrays
        """
        if self.embeddings is None:
            return {}

        result = {}
        for idx, node_id in self.reverse_mapping.items():
            result[node_id] = self.embeddings[idx].numpy()

        return result

    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a specific node.

        Args:
            node_id: ID of the node

        Returns:
            Embedding array or None if not available
        """
        if self.embeddings is None or node_id not in self.node_mapping:
            return None

        idx = self.node_mapping[node_id]
        return self.embeddings[idx].numpy()

    def similarity(self, node_id1: str, node_id2: str) -> float:
        """
        Calculate cosine similarity between two nodes.

        Args:
            node_id1: First node ID
            node_id2: Second node ID

        Returns:
            Cosine similarity between the node embeddings
        """
        emb1 = self.get_embedding(node_id1)
        emb2 = self.get_embedding(node_id2)

        if emb1 is None or emb2 is None:
            return 0.0

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / \
            (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def find_similar_nodes(self, node_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find the k most similar nodes to the given node.

        Args:
            node_id: ID of the node
            k: Number of similar nodes to return

        Returns:
            List of tuples (node_id, similarity)
        """
        if self.embeddings is None or node_id not in self.node_mapping:
            return []

        query_emb = self.get_embedding(node_id)
        similarities = []

        for other_id in self.node_mapping:
            if other_id == node_id:
                continue

            sim = self.similarity(node_id, other_id)
            similarities.append((other_id, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def save_model(self, path: str):
        """
        Save the model to a file.

        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'node_mapping': self.node_mapping,
            'reverse_mapping': self.reverse_mapping,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim
        }, path)

    def load_model(self, path: str):
        """
        Load the model from a file.

        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Recreate model with correct dimensions
        self.input_dim = checkpoint['input_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.output_dim = checkpoint['output_dim']

        self.model = GCN(self.input_dim, self.hidden_dim,
                         self.output_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.node_mapping = checkpoint['node_mapping']
        self.reverse_mapping = checkpoint['reverse_mapping']
