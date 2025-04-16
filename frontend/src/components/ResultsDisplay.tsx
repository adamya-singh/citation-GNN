
import { Card } from '@/components/ui/card';
import { PredictionResponse } from '@/types/graph';

interface ResultsDisplayProps {
  result?: PredictionResponse;
  error?: string;
}

export const ResultsDisplay = ({ result, error }: ResultsDisplayProps) => {
  if (error) {
    return (
      <Card className="p-4 bg-red-50">
        <p className="text-red-600">{error}</p>
      </Card>
    );
  }

  if (!result) {
    return (
      <Card className="p-4">
        <p className="text-gray-500">Make a prediction to see results</p>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <h3 className="font-medium mb-4">Prediction Results</h3>
      <div className="space-y-2">
        <div>
          <p className="text-sm text-gray-600">Citation Probability</p>
          <p className="text-xl font-semibold">{(result.probability * 100).toFixed(2)}%</p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Attention Layer 1</p>
          <p>{result.attention.layer1.toFixed(4)}</p>
        </div>
        <div>
          <p className="text-sm text-gray-600">Attention Layer 2</p>
          <p>{result.attention.layer2.toFixed(4)}</p>
        </div>
      </div>
    </Card>
  );
};
