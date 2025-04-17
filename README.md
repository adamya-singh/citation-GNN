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
   - Directory structure created (backend/frontend)
   - Virtual environment set up (backend)
   - Dependencies installed and documented (backend/frontend)
   - Initial Git setup with `.gitignore`

2. **Data Handling (Backend)**
   - `ArxivDataset` class implemented using OGB
   - Dataset loading and processing
   - Basic data statistics and access methods
   - Test file for data loading

3. **Model Development (Backend - Implementation Only)**
   - GAT model implementation (`models/gat.py`)
   - Attention weight tracking capability
   - Link prediction head (dot product)
   - Test file for model structure and forward pass
   - **Note:** Model training is not yet implemented.

4. **Backend API**
   - FastAPI application setup (`backend/app.py`)
   - CORS middleware configured
   - Two endpoints implemented:
     - `POST /predict_link`: Predicts citation probability (placeholder, uses untrained model).
     - `GET /graph`: Returns a sampled subgraph for visualization using a degree-based expansion strategy.
   - Error handling and input validation.
   - Basic API test suite (`backend/test_api.py`).
   - Subgraph sampling logic debugged and fixed.

5. **Frontend Setup**
   - Vite + React + TypeScript project initialized.
   - Basic project structure created (`frontend/src`).
   - Dependencies installed (including ReactFlow, Shadcn UI, Axios).
   - Basic components likely set up for visualization.

#### 🚧 In Progress:
1. **Model Training (Backend)**
   - Training loop implementation.
   - Evaluation metrics definition.
   - Saving/loading trained model weights.

2. **Frontend Development**
   - Implementing graph visualization using ReactFlow.
   - Fetching graph data from the `/graph` backend endpoint.
   - Creating UI components (potentially using Shadcn UI).
   - Adding interaction logic (e.g., prediction form, node inspection).

3. **Integration**
   - Testing the full frontend-backend data flow for graph visualization.
   - Connecting prediction form (once built) to `/predict_link`.

#### 📝 TODO:
1. **Complete Model Training**
   - Train the GAT model on the Arxiv dataset.
   - Evaluate model performance (AUC/AP).
   - Integrate saved model weights into the backend API startup.

2. **Complete Frontend Development**
   - Finalize graph visualization layout and styling.
   - Implement the prediction form/interface.
   - Add attention weight visualization (if feasible).
   - Improve UI/UX based on Shadcn components.

3. **Testing**
   - Enhance API test suite (e.g., test with trained model).
   - Add model validation tests.
   - Add basic frontend interaction tests.

4. **Deployment (Optional)**
   - Document deployment steps (e.g., using Docker).

### Tech Stack

- **Backend**:
  - **Framework**: FastAPI (Python)
  - **GNN Library**: PyTorch Geometric (PyG)
  - **Server**: Uvicorn (ASGI server for FastAPI)
  - **Data Handling**: OGB (`ogbn-arxiv`), In-memory storage
- **Frontend**:
  - **Framework/Tooling**: Vite + React + TypeScript
  - **UI Components**: Shadcn UI
  - **Graph Visualization**: ReactFlow
  - **API Calls**: Axios
  - **Package Manager**: npm / Bun (presence of `bun.lockb` noted)

**Rationale**:
- **FastAPI**: Lightweight, fast, modern Python API framework.
- **PyTorch Geometric**: Leading library for GNNs, integrates well with OGB.
- **Vite/React/TypeScript**: Modern, efficient frontend stack.
- **ReactFlow**: Powerful library specifically for node-based UIs and graph visualizations in React.
- **Shadcn UI**: Utility-first component library for building consistent UIs.

### Dataset: OGBN-Arxiv

The project uses the `ogbn-arxiv` dataset from the Open Graph Benchmark:
- **Nodes**: ~170,000 arXiv papers
- **Edges**: ~1.1 million directed citation links (i → j if paper i cites j)
- **Node Features**: 128-dimensional embeddings from paper abstracts/titles
- **Labels**: Subject areas (not used in this project)
- **Timestamps**: Year of publication (potential future use for temporal splitting)

**Data Handling**:
- The dataset is loaded using PyTorch Geometric's built-in dataset loader.
- Currently, the entire graph is used without temporal splitting for the MVP.
- Node indices (0 to N-1) are used as identifiers.

### Model Architecture

The project employs a GNN for link prediction:
- **Model**: GAT (Graph Attention Network)
  - **Layers**: 2 GAT layers implemented (configurable)
  - **Heads**: 8 attention heads per layer (configurable)
  - **Hidden Dimension**: 256 (configurable)
  - **Attention Tracking**: Implemented to return attention weights per layer.
  - **Link Prediction Head**: Dot product of final node embeddings followed by sigmoid.
- **Training (TODO)**:
  - **Positive Samples**: Existing citation edges.
  - **Negative Samples**: Randomly sampled non-edges (strategy can be refined).
  - **Loss Function**: Binary cross-entropy.
  - **Evaluation Metrics**: AUC or Average Precision.

**Rationale**:
- GAT provides attention mechanisms to learn citation importance.
- Attention weights offer interpretability for predictions.

### API Endpoints

The backend provides the following key API endpoints:
- **`POST /predict_link`**:
  - **Input**: JSON with two node indices, e.g., `{"node1": 0, "node2": 1}`
  - **Output**: Predicted citation probability and mean attention weights per layer, e.g., `{"probability": 0.75, "attention": {"layer1": 0.8, "layer2": 0.7}}`
  - **Current Status**: Functional but uses an *untrained* model.
- **`GET /graph`**:
  - **Input**: Optional query parameters `max_nodes` (default 30) and `max_edges_per_node` (default 3 - applied during edge selection).
  - **Output**: A subset of the citation graph (nodes and edges) for visualization.
  - **Sampling Logic**: Starts with the highest degree node and iteratively adds the highest-degree neighbors until `max_nodes` is reached. Edges *between* these selected nodes are then identified and returned, potentially limited by total degree.
  - **Current Status**: Functional and debugged.

**Implementation**:
- API built using FastAPI, GNN model and dataset loaded into memory on startup.
- Error handling for invalid inputs and server errors.
- Basic test suite available.

### Frontend Components (In Progress)

The frontend aims to provide:
1. **Interactive Graph Visualization**:
   - Displays a subgraph fetched from the `/graph` endpoint.
   - Uses ReactFlow for rendering nodes and edges.
   - Styling potentially managed via Shadcn UI and Tailwind CSS.
   - Goal: Allow users to explore a portion of the citation network.
2. **Prediction Interface (TODO)**:
   - Input fields for selecting two papers (by node index).
   - Button to trigger a call to the `/predict_link` endpoint.
   - Display area for the predicted probability and potentially attention information.

**Interaction**:
- Users can view the fetched subgraph.
- Future interaction may include node dragging, zooming, panning (via ReactFlow).
- The prediction interface will allow querying specific links.

### Development Phases

The project is progressing through these phases:
1. ✅ **Project Setup** (Complete)
2. ✅ **Data Handling (Backend)** (Complete)
3. ✅ **Model Development (Backend - Implementation)** (Complete)
4. ✅ **Backend API** (Complete, pending model training integration)
5. 🚧 **Frontend Setup & Visualization** (In Progress)
6. 🚧 **Integration and Testing** (In Progress - basic graph fetching)
7. 📝 **Model Training** (TODO)
8. 📝 **Complete Frontend Features** (TODO)
9. 📝 **Final Testing & Refinement** (TODO)
10. 📝 **Optional Enhancements** (if time permits)

### Open Design Choices

1. **Model Architecture/Training**:
   - Fine-tune hyperparameters (layers, heads, dimensions).
   - Refine negative sampling strategy.
   - Decide on evaluation protocol.
2. **Storage and Querying**:
   - Currently in-memory; likely sufficient for MVP.
3. **Frontend Design/Features**:
   - Finalize graph layout and interactions (ReactFlow offers many options).
   - Decide how/if to display attention weights.
   - Integrate paper titles or other metadata if feasible.

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

