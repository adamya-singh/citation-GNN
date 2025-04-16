from dataset import ArxivDataset

def test_dataset_loading():
    # Initialize the dataset
    dataset = ArxivDataset(root='./data')
    
    # Load the dataset
    print("Loading dataset...")
    dataset.load()
    
    # Get basic statistics
    print("\nDataset Statistics:")
    print(f"Number of nodes: {dataset.get_num_nodes()}")
    print(f"Number of edges: {dataset.get_num_edges()}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    
    # Get split sizes
    split_masks = dataset.get_split_masks()
    print("\nSplit Sizes:")
    for split_name, mask in split_masks.items():
        print(f"{split_name}: {len(mask)} nodes")
    
    # Get graph data
    graph = dataset.get_graph_data()
    print("\nGraph Data:")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    
    print("\nDataset loading test completed successfully!")

if __name__ == "__main__":
    test_dataset_loading() 