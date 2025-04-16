import torch
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from typing import Tuple, Dict, Any
import numpy as np
import functools

# Monkey patch torch.load to use weights_only=False for OGB dataset loading
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

class ArxivDataset:
    def __init__(self, root: str = './data'):
        """
        Initialize the Arxiv dataset handler.
        
        Args:
            root (str): Root directory where the dataset will be stored
        """
        self.root = root
        self.dataset = None
        self.graph = None
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        
    def load(self) -> None:
        """
        Load the OGBN-Arxiv dataset and prepare it for link prediction.
        """
        # Load the dataset
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=self.root)
        
        # Get the graph data
        self.graph = self.dataset[0]
        
        # Get the split masks
        split_idx = self.dataset.get_idx_split()
        self.train_mask = split_idx['train']
        self.val_mask = split_idx['valid']
        self.test_mask = split_idx['test']
        
    def get_graph_data(self) -> Data:
        """
        Get the processed graph data.
        
        Returns:
            Data: PyTorch Geometric Data object containing the graph
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph
    
    def get_split_masks(self) -> Dict[str, torch.Tensor]:
        """
        Get the train/val/test split masks.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing the split masks
        """
        if self.train_mask is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return {
            'train': self.train_mask,
            'val': self.val_mask,
            'test': self.test_mask
        }
    
    def get_node_features(self) -> torch.Tensor:
        """
        Get the node features.
        
        Returns:
            torch.Tensor: Node feature matrix
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph.x
    
    def get_edge_index(self) -> torch.Tensor:
        """
        Get the edge index tensor.
        
        Returns:
            torch.Tensor: Edge index tensor
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph.edge_index
    
    def get_num_nodes(self) -> int:
        """
        Get the number of nodes in the graph.
        
        Returns:
            int: Number of nodes
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph.num_nodes
    
    def get_num_edges(self) -> int:
        """
        Get the number of edges in the graph.
        
        Returns:
            int: Number of edges
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph.edge_index.shape[1]
    
    def get_feature_dim(self) -> int:
        """
        Get the dimension of node features.
        
        Returns:
            int: Feature dimension
        """
        if self.graph is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self.graph.x.shape[1] 