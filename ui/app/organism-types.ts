export type OrganismMetrics = {
  energy: number;
  viability: number;
  quiescentFraction: number;
  novelty: number;
  cellCredit: number;
  edgeCredit: number;
  perplexity: number;
  edgeFlow: number;
  energyInput?: number;
  energySpent?: number;
};

export type CheckpointInfo = {
  name: string;
  updates: number;
  cells: number;
  dendrites: number;
  metabolismEnabled: boolean;
};

export type OrganismTopology = {
  sources: number[][];
  weights: number[][];
  fastWeights: number[][];
  edgeFlow: number[][];
  probeFlow: number[];
  cellActivity: number[];
  cellEnergy: number[];
  cellViability: number[];
  sensoryCells: number[];
  outputCells: number[];
};

export type OrganismPayload = {
  mode: "live-checkpoint";
  output?: string;
  checkpoint: CheckpointInfo;
  clock: {
    ticks: number;
    lastInput: string | null;
    lastOutput: string;
  };
  metrics: OrganismMetrics;
  topology: OrganismTopology;
};

export const EMPTY_METRICS: OrganismMetrics = {
  energy: 0,
  viability: 0,
  quiescentFraction: 0,
  novelty: 0,
  cellCredit: 0,
  edgeCredit: 0,
  perplexity: Number.NaN,
  edgeFlow: 0,
};
