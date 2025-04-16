
export interface Node {
  id: number;
  features: number[];
  x?: number;
  y?: number;
}

export interface Edge {
  source: number;
  target: number;
  weight?: number;
}

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

export interface PredictionRequest {
  node1: number;
  node2: number;
}

export interface PredictionResponse {
  probability: number;
  attention: {
    layer1: number;
    layer2: number;
  };
}
