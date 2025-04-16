
import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';

interface PredictionFormProps {
  onPredict: (node1: number, node2: number) => void;
  isLoading?: boolean;
}

export const PredictionForm = ({ onPredict, isLoading }: PredictionFormProps) => {
  const [node1, setNode1] = useState('');
  const [node2, setNode2] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const n1 = parseInt(node1);
    const n2 = parseInt(node2);
    
    if (!isNaN(n1) && !isNaN(n2)) {
      onPredict(n1, n2);
    }
  };

  return (
    <Card className="p-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-1">Paper 1 Index</label>
          <Input
            type="number"
            value={node1}
            onChange={(e) => setNode1(e.target.value)}
            placeholder="Enter paper 1 index"
            required
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Paper 2 Index</label>
          <Input
            type="number"
            value={node2}
            onChange={(e) => setNode2(e.target.value)}
            placeholder="Enter paper 2 index"
            required
          />
        </div>
        <Button type="submit" disabled={isLoading}>
          {isLoading ? 'Predicting...' : 'Predict Citation'}
        </Button>
      </form>
    </Card>
  );
};
