
import { useState, useEffect } from 'react';
import axios from 'axios';
import { GraphVisualization } from '@/components/GraphVisualization';
import { PredictionForm } from '@/components/PredictionForm';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { GraphData, PredictionResponse } from '@/types/graph';

const API_BASE_URL = 'http://localhost:8000';

const Index = () => {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | undefined>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    fetchGraphData();
  }, []);

  const fetchGraphData = async () => {
    try {
      const response = await axios.get<GraphData>(`${API_BASE_URL}/graph`);
      setGraphData(response.data);
    } catch (error) {
      setError('Failed to load graph data');
      console.error('Error fetching graph data:', error);
    }
  };

  const handlePredict = async (node1: number, node2: number) => {
    setLoading(true);
    setError(undefined);
    try {
      const response = await axios.post<PredictionResponse>(
        `${API_BASE_URL}/predict_link`,
        { node1, node2 }
      );
      setPredictionResult(response.data);
    } catch (error: any) {
      setError(error.response?.data?.detail || 'Failed to make prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-6">Citation Graph Predictor</h1>
      
      {graphData ? (
        <div className="mb-6">
          <GraphVisualization
            nodes={graphData.nodes}
            edges={graphData.edges}
          />
        </div>
      ) : (
        <div className="text-center p-4">Loading graph...</div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PredictionForm onPredict={handlePredict} isLoading={loading} />
        <ResultsDisplay result={predictionResult} error={error} />
      </div>
    </div>
  );
};

export default Index;
