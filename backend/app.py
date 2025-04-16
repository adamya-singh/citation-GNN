from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import torch

from data.dataset import ArxivDataset
from models.gat import GATLinkPredictor

# Initialize FastAPI app
app = FastAPI(title="Citation GNN API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for dataset and model
dataset = None
model = None

class LinkPredictionRequest(BaseModel):
    node1: int
    node2: int

class LinkPredictionResponse(BaseModel):
    probability: float
    attention: Dict[str, float]

class GraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

@app.on_event("startup")
async def startup_event():
    """Initialize dataset and model on startup"""
    global dataset, model
    
    try:
        # Initialize and load dataset
        dataset = ArxivDataset(root='./data')
        dataset.load()
        
        # Initialize model
        model = GATLinkPredictor(
            in_channels=dataset.get_feature_dim(),
            hidden_channels=256,
            num_heads=8
        )
        
        # Load model weights (placeholder - will need to be implemented)
        # model.load_state_dict(torch.load('model_weights.pth'))
        
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise

@app.post("/predict_link", response_model=LinkPredictionResponse)
async def predict_link(request: LinkPredictionRequest):
    """Predict citation probability between two papers"""
    if dataset is None or model is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validate node indices
    num_nodes = dataset.get_num_nodes()
    if request.node1 >= num_nodes or request.node2 >= num_nodes:
        raise HTTPException(
            status_code=400,
            detail=f"Node indices must be less than {num_nodes}"
        )
    
    try:
        # Get graph data
        graph = dataset.get_graph_data()
        x = graph.x
        edge_index = graph.edge_index
        
        # Create node pairs tensor
        node_pairs = torch.tensor([[request.node1], [request.node2]])
        
        # Get prediction
        with torch.no_grad():
            link_probs, attention_weights = model.predict_link(
                x, edge_index, node_pairs
            )
        
        # Convert to response format
        probability = link_probs[0].item()
        attention = {
            'layer1': attention_weights['layer1'][1].mean().item(),
            'layer2': attention_weights['layer2'][1].mean().item()
        }
        
        return LinkPredictionResponse(
            probability=probability,
            attention=attention
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.get("/graph", response_model=GraphResponse)
async def get_graph(max_nodes: int = 30, max_edges_per_node: int = 3):
    """Get a subset of the citation graph for visualization"""
    if dataset is None:
        raise HTTPException(status_code=500, detail="Dataset not initialized")
    
    try:
        # Get graph data
        graph = dataset.get_graph_data()
        edge_index = graph.edge_index
        
        print(f"Total edges in dataset: {edge_index.shape[1]}")
        
        # Pre-compute degrees for all nodes (fast)
        degrees = torch.zeros(graph.num_nodes, dtype=torch.long)
        degrees.index_add_(0, edge_index[0], torch.ones_like(edge_index[0]))
        degrees.index_add_(0, edge_index[1], torch.ones_like(edge_index[1]))
        
        print(f"Highest degree node: {degrees.argmax().item()} with degree {degrees.max().item()}")
        
        # Start with highest degree node
        selected_nodes = {degrees.argmax().item()}
        print(f"Starting with node {selected_nodes}")
        
        # Select nodes based on degree and connectivity
        iteration = 0
        while len(selected_nodes) < max_nodes:
            iteration += 1
            print(f"\nIteration {iteration}:")
            print(f"Current selected nodes: {sorted(selected_nodes)}")
            
            # Find all edges where either source or target is in selected_nodes
            selected_tensor = torch.tensor(list(selected_nodes), dtype=edge_index.dtype)
            
            # Create masks using tensor operations
            src_mask = torch.isin(edge_index[0], selected_tensor)
            dst_mask = torch.isin(edge_index[1], selected_tensor)
            connected_edges = src_mask | dst_mask
            
            # Get all connected nodes (both sources and targets)
            connected_sources = edge_index[0][connected_edges]
            connected_targets = edge_index[1][connected_edges]
            all_connected = torch.cat([connected_sources, connected_targets])
            
            # Get unique connected nodes and remove already selected ones
            connected = torch.unique(all_connected)
            connected = connected[~torch.isin(connected, selected_tensor)]
            
            if len(connected) == 0:
                print("No more connected nodes found")
                break
            
            # Add node with highest degree
            new_node = connected[degrees[connected].argmax()].item()
            if new_node in selected_nodes:
                print("Warning: Attempted to add duplicate node")
                break
                
            selected_nodes.add(new_node)
            print(f"Added node {new_node} with degree {degrees[new_node].item()}")
            print(f"Total selected nodes: {len(selected_nodes)}")
        
        print(f"\nFinal selection: {len(selected_nodes)} nodes")
        print(f"Selected nodes: {sorted(selected_nodes)}")
        
        # Create response
        nodes = [{'features': graph.x[i].tolist(), 'id': i} for i in sorted(selected_nodes)]
        edges = []
        
        # Find edges between selected nodes
        selected_tensor = torch.tensor(list(selected_nodes), dtype=edge_index.dtype)
        src_mask = torch.isin(edge_index[0], selected_tensor)
        dst_mask = torch.isin(edge_index[1], selected_tensor)
        valid_edges = src_mask & dst_mask
        
        print(f"Edges between selected nodes: {valid_edges.sum().item()}")
        
        # Track edge counts for both source and target nodes
        edge_counts = {node: {'source': 0, 'target': 0} for node in selected_nodes}
        
        # Get all valid edges first
        valid_edge_indices = torch.where(valid_edges)[0]
        edge_degrees = torch.zeros(len(valid_edge_indices))
        
        # Calculate edge degrees (sum of degrees of source and target)
        for i, edge_idx in enumerate(valid_edge_indices):
            src = edge_index[0, edge_idx].item()
            dst = edge_index[1, edge_idx].item()
            edge_degrees[i] = degrees[src] + degrees[dst]
        
        # Sort edges by degree and take top max_edges_per_node * len(selected_nodes)
        sorted_indices = torch.argsort(edge_degrees, descending=True)
        max_total_edges = max_edges_per_node * len(selected_nodes)
        selected_edge_indices = valid_edge_indices[sorted_indices[:max_total_edges]]
        
        # Add selected edges
        for edge_idx in selected_edge_indices:
            src = edge_index[0, edge_idx].item()
            dst = edge_index[1, edge_idx].item()
            
            edges.append({
                'source': src,
                'target': dst,
                'id': f'e{len(edges)}'
            })
            
            edge_counts[src]['source'] += 1
            edge_counts[dst]['target'] += 1
        
        print("Edge counts per node:")
        for node in sorted(selected_nodes):
            print(f"Node {node}: source={edge_counts[node]['source']}, target={edge_counts[node]['target']}")
        
        return GraphResponse(nodes=nodes, edges=edges)
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting graph data: {str(e)}"
        )
