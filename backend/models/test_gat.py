import torch
from models.gat import GATLinkPredictor
from data.dataset import ArxivDataset

def test_gat_model():
    # Initialize dataset
    print("Loading dataset...")
    dataset = ArxivDataset(root='./data')
    dataset.load()
    
    # Get graph data
    graph = dataset.get_graph_data()
    x = graph.x
    edge_index = graph.edge_index
    
    # Initialize model
    print("\nInitializing GAT model...")
    model = GATLinkPredictor(
        in_channels=dataset.get_feature_dim(),
        hidden_channels=256,
        num_heads=8
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    node_embeddings, attention_weights = model(x, edge_index, return_attention=True)
    print(f"Node embeddings shape: {node_embeddings.shape}")
    print(f"Attention weights keys: {list(attention_weights.keys())}")
    
    # Test link prediction
    print("\nTesting link prediction...")
    # Create some random node pairs for testing
    num_test_pairs = 10
    node_pairs = torch.randint(0, dataset.get_num_nodes(), (2, num_test_pairs))
    
    link_probs, att_weights = model.predict_link(x, edge_index, node_pairs)
    print(f"Link probabilities shape: {link_probs.shape}")
    print(f"Sample link probabilities: {link_probs[:5]}")
    
    print("\nGAT model test completed successfully!")

if __name__ == "__main__":
    test_gat_model() 