export interface BrainProbe {
  task_name: string;
  size: number;
  task: number;
  num_neurons: number;
  num_synapses: number;
  num_active_synapses: number;
  total_inferences: number;
  total_learning_steps: number;
  avg_sparsity: number;
  avg_inference_time_us: number;
  current_learning_rate: number;
  accuracy: number;
  memory_bytes: number;
  num_inputs: number;
  num_outputs: number;
  utilization: number;
  last_loss: number;
  is_cow_clone: boolean;
  cow_ref_count: number;
  cow_shared_bytes: number;
  cow_private_bytes: number;
}

export interface BrainInfo {
  id: number;
  name: string;
  created_at: string;
  dataset: string | null;
  parent_id: number | null;
  probe: BrainProbe | null;
}

export interface BrainDetail {
  id: number;
  name: string;
  created_at: string;
  size: number;
  size_label: string;
  task: number;
  task_label: string;
  num_inputs: number;
  num_outputs: number;
  dataset: string | null;
  parent_id: number | null;
  probe: BrainProbe | null;
  total_inferences: number;
  total_learning_steps: number;
  accuracy: number;
  last_loss: number;
  loss_history: number[];
}

export interface BrainCreate {
  name: string;
  size: number;
  task: number;
  num_inputs: number;
  num_outputs: number;
}

export interface TrainingConfig {
  loss_type: number;
  optimizer_type: number;
  scheduler_type: number;
  learning_rate: number;
  weight_decay: number;
  momentum: number;
  beta1: number;
  beta2: number;
  epsilon: number;
  scheduler_step_size: number;
  scheduler_gamma: number;
  warmup_steps: number;
  enable_gradient_clipping: boolean;
  gradient_clip_value: number;
  enable_biological_modulation: boolean;
  biological_blend: number;
}

export interface TrainingProgress {
  brain_id: number;
  running: boolean;
  epoch: number;
  step: number;
  total_steps: number;
  loss: number;
  accuracy: number;
  learning_rate: number;
  elapsed_seconds: number;
}

export interface DatasetInfo {
  id: string;
  name: string;
  num_inputs: number;
  num_outputs: number;
  num_classes: number;
  description: string;
  total_examples: number;
}

export interface ChatMessage {
  id: number;
  sender: 'user' | 'brain';
  text: string;
  label?: string | number;
  confidence?: number;
  time_ms?: number;
  mode?: string;
  explanation?: string;
  sparsity?: number;
  num_active_neurons?: number;
  inference_time_us?: number;
  output_vector?: number[];
}

export interface WSMessage {
  type: string;
  [key: string]: unknown;
}

export interface ScriptInfo {
  id: string;
  name: string;
  description: string;
  exists: boolean;
}

export interface ScriptStatus {
  script_id?: string;
  brain_id?: number;
  status: string;
  exit_code?: number | null;
  stdout_lines?: string[];
  total_lines?: number;
}

export type Tab = 'dashboard' | 'training' | 'chat' | 'datasets';
