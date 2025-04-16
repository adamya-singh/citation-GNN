import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Tuple, Dict, Optional

class GATLinkPredictor(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.6):
        """
        Initialize the GAT link prediction model.
        
        Args:
            in_channels (int): Input feature dimension
            hidden_channels (int): Hidden dimension size
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super().__init__()
        
        # First GAT layer
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False
        )
        
        # Second GAT layer
        self.conv2 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=False
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)
        
        # Store attention weights
        self.attention_weights = None
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            return_attention (bool): Whether to return attention weights
            
        Returns:
            Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]: 
                Node embeddings and attention weights if requested
        """
        # First GAT layer
        x, att1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = self.dropout(x)
        
        # Second GAT layer
        x, att2 = self.conv2(x, edge_index, return_attention_weights=True)
        
        # Store attention weights if requested
        if return_attention:
            self.attention_weights = {
                'layer1': att1,
                'layer2': att2
            }
        
        return x, self.attention_weights if return_attention else None
    
    def predict_link(self, 
                    x: torch.Tensor,
                    edge_index: torch.Tensor,
                    node_pairs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict link probability between node pairs.
        
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
            node_pairs (torch.Tensor): Pairs of nodes to predict links for
            
        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: 
                Link probabilities and attention weights
        """
        # Get node embeddings
        node_embeddings, attention_weights = self(x, edge_index, return_attention=True)
        
        # Get embeddings for the node pairs
        src_embeddings = node_embeddings[node_pairs[0]]
        dst_embeddings = node_embeddings[node_pairs[1]]
        
        # Compute link probabilities using dot product
        link_probs = torch.sigmoid(torch.sum(src_embeddings * dst_embeddings, dim=1))
        
        return link_probs, attention_weights 