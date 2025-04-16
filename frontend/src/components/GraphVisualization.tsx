
import { useCallback } from 'react';
import ReactFlow, { 
  Background, 
  Controls,
  Node as FlowNode,
  Edge as FlowEdge,
  useNodesState,
  useEdgesState
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Node, Edge } from '@/types/graph';

interface GraphVisualizationProps {
  nodes: Node[];
  edges: Edge[];
  onNodeSelect?: (nodeId: number) => void;
}

export const GraphVisualization = ({ nodes, edges, onNodeSelect }: GraphVisualizationProps) => {
  const initialNodes: FlowNode[] = nodes.map((node) => ({
    id: node.id.toString(),
    position: { x: node.x || Math.random() * 800, y: node.y || Math.random() * 600 },
    data: { label: `Paper ${node.id}` },
  }));

  const initialEdges: FlowEdge[] = edges.map((edge, index) => ({
    id: `e${index}`,
    source: edge.source.toString(),
    target: edge.target.toString(),
    type: 'straight',
    animated: true,
    style: { stroke: '#999', strokeWidth: edge.weight ? edge.weight * 2 : 1 },
  }));

  const [flowNodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [flowEdges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onNodeClick = useCallback((_, node: FlowNode) => {
    onNodeSelect?.(parseInt(node.id));
  }, [onNodeSelect]);

  return (
    <div style={{ width: '100%', height: '500px' }}>
      <ReactFlow
        nodes={flowNodes}
        edges={flowEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        fitView
      >
        <Background />
        <Controls />
      </ReactFlow>
    </div>
  );
};
