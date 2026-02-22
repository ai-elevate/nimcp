import api from './api';

export interface BenchmarkRequest {
  benchmark_id: string;
  brain_size?: number;
  strategy?: string;
  epochs?: number;
  include_cognitive?: boolean;
}

export const listBenchmarks = () => api.get('/benchmarks/available');

export const runBenchmark = (req: BenchmarkRequest) =>
  api.post('/benchmarks/run', req);

export const getBenchmarkStatus = () => api.get('/benchmarks/status');

export const getBenchmarkResults = () => api.get('/benchmarks/results');

export const stopBenchmark = () => api.post('/benchmarks/stop');
