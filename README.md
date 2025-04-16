## Project Description: Future Citation Link Prediction with GNNs + Graph Visualization Interface

### Project Overview

The goal of this project is to build a system that predicts future citations between research papers using a Graph Neural Network (GNN) and visualizes the citation graph through an interactive web interface. The system will:
- Use historical citation data to train a GNN model for link prediction.
- Provide an API to query predicted citation probabilities between papers.
- Offer a frontend interface to visualize the citation graph and display prediction results.
- Show attention weights to explain citation importance.

This is a solo weekend project, so the focus is on creating a functional Minimum Viable Product (MVP) that can be easily demonstrated to others. The system does not need to scale to many users, and data will be stored in memory to avoid the complexity of maintaining a database.

### Current Progress

#### ✅ Completed Components:
1. **Project Setup**
   - Directory structure created
   - Virtual environment set up
   - Dependencies installed and documented

2. **Data Handling**
   - ArxivDataset class implemented
   - Dataset loading and processing
   - Basic data statistics and access methods
   - Test file for data loading

3. **Model Development**
   - GAT model implementation
   - Attention weight tracking
   - Link prediction functionality
   - Test file for model functionality

4. **Backend API**
   - FastAPI application setup
   - Two endpoints implemented:
     - `/predict_link` (POST): Predicts citation probability between papers
     - `/graph` (GET): Returns graph subset for visualization
   - Error handling and input validation
   - Basic API test suite

#### 🚧 In Progress:
1. **Model Training**
   - Model architecture implemented
   - Training code needs to be written
   - Model weights need to be saved/loaded

2. **Frontend Development**
   - Not started yet
   - Will use Next.js with graph visualization

#### 📝 TODO:
1. **Model Training**
   - Implement training loop
   - Add evaluation metrics
   - Save/load model weights

2. **Frontend Development**
   - Set up Next.js project
   - Implement graph visualization
   - Create prediction form
   - Add attention weight visualization

3. **Integration**
   - Connect frontend to backend
   - Test full system flow
   - Add error handling

4. **Testing**
   - Enhance API test suite
   - Add model validation tests
   - Add frontend tests

### Tech Stack

- **Backend**:
  - **Framework**: FastAPI (Python)
  - **GNN Library**: PyTorch Geometric (PyG)
  - **Server**: Uvicorn (ASGI server for FastAPI)
  - **Data Handling**: In-memory storage for the dataset and model
- **Frontend**:
  - **Framework**: Next.js (React)
  - **Graph Visualization**: Cytoscape.js or Sigma.js
  - **API Calls**: Fetch or Axios for interacting with the backend API

**Rationale**:
- **FastAPI**: Lightweight, fast, and easy to set up for RESTful APIs.
- **PyTorch Geometric**: Leading library for GNNs, integrates well with the dataset.
- **Next.js**: Provides a robust React framework with server-side rendering and easy API integration.
- **Cytoscape.js/Sigma.js**: Specialized for interactive graph visualization, with React wrappers available.

### Dataset: OGBN-Arxiv

The project uses the `ogbn-arxiv` dataset from the Open Graph Benchmark:
- **Nodes**: ~170,000 arXiv papers
- **Edges**: ~1.1 million directed citation links (i → j if paper i cites j)
- **Node Features**: 128-dimensional embeddings from paper abstracts/titles
- **Labels**: Subject areas (not used in this project)
- **Timestamps**: Year of publication (used for temporal splitting)

**Data Handling**:
- The dataset is loaded using PyTorch Geometric's built-in dataset loader.
- Data is split based on publication year:
  - **Training**: Citations between papers published before 2018
  - **Testing**: Citations involving papers published in 2018 or later
- For the MVP, node indices (0 to N-1) are used as identifiers. Paper titles can be added later if time permits.

### Model Architecture

The project employs a GNN for link prediction:
- **Model**: GAT (Graph Attention Network)
  - **Layers**: 2-3 GAT layers with 8 attention heads each
  - **Hidden Dimension**: 256
  - **Link Prediction Head**: Dot product of node embeddings
- **Training**:
  - **Positive Samples**: Existing citation edges from the training set
  - **Negative Samples**: Randomly sampled non-edges
  - **Loss Function**: Binary cross-entropy
  - **Evaluation Metrics**: AUC or Average Precision on the test set

**Rationale**:
- GAT provides attention mechanisms to learn citation importance
- Attention weights offer interpretability for predictions
- Efficient for large graphs like `ogbn-arxiv`
- Naturally handles directed relationships

### API Endpoints

The backend provides the following key API endpoints:
- **`POST /predict_link`**:
  - **Input**: JSON with two node indices, e.g., `{"node1": 0, "node2": 1}`
  - **Output**: Predicted citation probability and attention weights, e.g., `{"probability": 0.75, "attention": {"layer1": 0.8, "layer2": 0.7}}`
- **`GET /graph`**:
  - **Output**: A subset of the citation graph (nodes and edges) for visualization
  - Returns first 100 nodes and all their connections
  - Includes node features and edge information

**Implementation**:
- The API is built using FastAPI, with the GNN model and dataset loaded into memory on startup.
- Error handling for invalid inputs and server errors.
- Basic test suite for API functionality.

### Frontend Components (Not Started)

The frontend will consist of two main components:
1. **Static Graph Visualization**:
   - Displays a fixed subset of the citation graph (e.g., 100 nodes and their edges).
   - Uses Cytoscape.js or Sigma.js for rendering.
   - Visualizes attention weights through edge thickness.
2. **Prediction Form**:
   - Provides input fields for two node indices.
   - Includes a button to query the backend API and display the predicted citation probability.
   - Shows attention weights for interpretability.

**Interaction**:
- The graph will be static for the MVP, with optional interactivity (e.g., clicking nodes to view details) if time allows.
- The form will use API calls to fetch and display predictions.

### Development Phases

The project is divided into the following phases:
1. ✅ **Project Setup** (1 hour):
   - Create directory structure, set up virtual environment, install initial dependencies.
2. ✅ **Data Handling** (2 hours):
   - Load and split the `ogbn-arxiv` dataset using PyTorch Geometric.
3. ✅ **Model Development** (4-5 hours):
   - Implement and train a GAT model for link prediction.
   - Add attention weight tracking.
4. ✅ **Backend API** (2-3 hours):
   - Create the `/predict_link` endpoint in FastAPI.
   - Add attention weight handling.
5. 🚧 **Frontend Visualization** (3-4 hours):
   - Build the Next.js app with graph visualization and prediction form.
   - Add attention weight visualization.
6. 🚧 **Integration and Testing** (2-3 hours):
   - Connect the frontend to the backend and test the full flow.
7. 📝 **Optional Enhancements** (if time permits):
   - Add interactivity to the graph, display paper titles, or improve UI/UX.

**Total Estimated Time**: 14-17 hours

### Open Design Choices

The following design decisions are still open and can be refined with LLM assistance:
1. **Model Architecture**:
   - Fine-tune number of attention heads and layers.
   - Consider multi-head attention aggregation strategies.
2. **Negative Sampling Strategy**:
   - Use random negatives for the MVP.
   - Experiment with hard negatives (e.g., nodes with shared neighbors) for improved performance.
3. **Storage and Querying**:
   - Currently in-memory; consider SQLite if memory constraints arise.
   - Defer advanced storage solutions (e.g., Neo4j) unless needed.
4. **Frontend Design**:
   - Start with a static graph; add interactivity if time allows.
   - Optionally, integrate paper titles or other metadata for a richer experience.

### LLM Assistance

The LLM can assist in the following ways:
- **Code Generation**:
  - Generate Python code for FastAPI routes, GNN model implementation, or data loading.
  - Create React components for the frontend, including graph visualization and form handling.
- **Design Recommendations**:
  - Suggest optimal GAT configurations or negative sampling strategies.
  - Advise on storage solutions or frontend libraries based on project needs.
- **Debugging and Refactoring**:
  - Help debug issues in code snippets or optimize existing implementations.
- **Documentation**:
  - Assist in writing README files, API documentation, or project summaries.

**Usage**:
- Provide the LLM with specific tasks or questions related to the project.
- Use the LLM to generate code snippets, clarify design choices, or break down implementation steps.

This detailed project description provides the necessary context for an LLM to assist effectively. It outlines the project's structure, goals, and technologies, while highlighting areas where the LLM can contribute. Use this as a reference to guide your interactions with the LLM throughout the development process.

