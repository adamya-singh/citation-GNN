import os
import logging
import argparse
from datetime import datetime
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from models.gat import GATLinkPredictor
from data.dataset import ArxivDataset

# Set up logging
def setup_logging(log_dir: str = "logs") -> None:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")

def get_device() -> torch.device:
    """Get the appropriate device (MPS for M1, CPU otherwise)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        logging.info("MPS not available, using CPU")
    return device

def load_dataset(data_dir: str = "data") -> Tuple[Data, Data, Data]:
    """Load the OGB Arxiv dataset and return graph data with link prediction splits."""
    try:
        logging.info(f"Loading dataset from {data_dir}")
        dataset = ArxivDataset(root=data_dir)
        dataset.load()
        data = dataset.get_graph_data()
        
        # Create link prediction splits
        logging.info("Creating link prediction splits...")
        transform = RandomLinkSplit(
            num_val=0.05,
            num_test=0.1,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=False
        )
        
        # Apply the transform to get train, val, and test data
        train_data, val_data, test_data = transform(data)
        
        # Log basic information
        logging.info("Dataset loaded successfully:")
        logging.info(f"  Number of nodes: {data.num_nodes}")
        logging.info(f"  Number of edges: {data.edge_index.size(1)}")
        logging.info(f"  Node feature dimension: {data.num_node_features}")
        
        # Log split information
        logging.info("Split information:")
        logging.info(f"  Train edges: {train_data.edge_index.size(1)}")
        logging.info(f"  Val edges: {val_data.edge_index.size(1)}")
        logging.info(f"  Test edges: {test_data.edge_index.size(1)}")
        
        # Log available attributes
        logging.info("Available attributes in splits:")
        logging.info(f"  Train: {train_data.keys()}")
        logging.info(f"  Val: {val_data.keys()}")
        logging.info(f"  Test: {test_data.keys()}")
        
        return train_data, val_data, test_data
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

def evaluate(model: GATLinkPredictor, 
            data: Data,
            device: torch.device) -> Tuple[float, float]:
    """Evaluate model performance on a given split."""
    model.eval()
    with torch.no_grad():
        # Get node embeddings
        h, _ = model(data.x.to(device), data.edge_index.to(device))
        
        # Get positive and negative edges
        pos_edge = data.pos_edge_label_index.to(device)
        neg_edge = data.neg_edge_label_index.to(device)
        
        # Calculate logits
        pos_logits = torch.sum(h[pos_edge[0]] * h[pos_edge[1]], dim=1)
        neg_logits = torch.sum(h[neg_edge[0]] * h[neg_edge[1]], dim=1)
        
        # Combine and calculate metrics
        logits = torch.cat([pos_logits, neg_logits]).cpu().numpy()
        labels = torch.cat([torch.ones_like(pos_logits), 
                          torch.zeros_like(neg_logits)]).cpu().numpy()
        
        auc = roc_auc_score(labels, logits)
        ap = average_precision_score(labels, logits)
        
        return auc, ap

def train(args: argparse.Namespace) -> None:
    """Main training function."""
    # Setup
    setup_logging()
    device = get_device()
    
    # Load data
    train_data, val_data, test_data = load_dataset(args.data_dir)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Initialize model
    logging.info("Initializing model...")
    model = GATLinkPredictor(
        in_channels=train_data.num_node_features,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    logging.info("Starting training...")
    best_val_auc = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        h, _ = model(train_data.x, train_data.edge_index)
        
        # Get positive edges
        pos_train_edge = train_data.pos_edge_label_index.to(device)
        
        # Sample negative edges
        neg_train_edge = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=pos_train_edge.size(1)
        ).to(device)
        
        # Calculate logits
        pos_logits = torch.sum(h[pos_train_edge[0]] * h[pos_train_edge[1]], dim=1)
        neg_logits = torch.sum(h[neg_train_edge[0]] * h[neg_train_edge[1]], dim=1)
        
        # Combine and calculate loss
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), 
                          torch.zeros_like(neg_logits)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Log training progress
        if (epoch + 1) % args.log_interval == 0:
            logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}")
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0:
            val_auc, val_ap = evaluate(model, val_data, device)
            logging.info(f"Epoch {epoch+1}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1
                torch.save(model.state_dict(), args.model_save_path)
                logging.info(f"New best model saved with Val AUC: {best_val_auc:.4f}")
    
    # Final evaluation
    logging.info(f"Training completed. Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(args.model_save_path))
    test_auc, test_ap = evaluate(model, test_data, device)
    logging.info(f"Final Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train GAT for link prediction on OGB-Arxiv')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to store/load dataset')
    parser.add_argument('--model_save_path', type=str, default='trained_models/gat_arxiv.pt',
                        help='Path to save the best model')
    
    # Model arguments
    parser.add_argument('--hidden_channels', type=int, default=256,
                        help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout probability')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log training loss every N epochs')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='Evaluate on validation set every N epochs')
    
    args = parser.parse_args()
    
    # Create model save directory
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    train(args)

if __name__ == "__main__":
    main() 