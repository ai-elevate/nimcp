import api from './api';
import type { BrainCreate, BrainInfo, BrainDetail, BrainProbe } from '../types';

export const createBrain = (data: BrainCreate) =>
  api.post<{ id: number; probe: BrainProbe }>('/brains/', data);

export const listBrains = () =>
  api.get<BrainInfo[]>('/brains/');

export const getBrain = (id: number) =>
  api.get<BrainInfo>(`/brains/${id}`);

export const getBrainDetail = (id: number) =>
  api.get<BrainDetail>(`/brains/${id}`);

export const renameBrain = (id: number, name: string) =>
  api.patch(`/brains/${id}`, { name });

export const deleteBrain = (id: number) =>
  api.delete(`/brains/${id}`);

export const probeBrain = (id: number) =>
  api.get<BrainProbe>(`/brains/${id}/probe`);

export const probeHistory = (id: number) =>
  api.get<(BrainProbe & { timestamp: number })[]>(`/brains/${id}/probe/history`);

export const predict = (id: number, features: number[]) =>
  api.post<{ label: string; confidence: number }>(`/brains/${id}/predict`, { features });

export const learn = (id: number, features: number[], label: string | number, confidence = 1.0) =>
  api.post(`/brains/${id}/learn`, { features, label, confidence });

export const createSnapshot = (id: number, name: string) =>
  api.post(`/brains/${id}/snapshots`, { name });

export const listSnapshots = (id: number) =>
  api.get<{ name: string; path: string }[]>(`/brains/${id}/snapshots`);

export const restoreSnapshot = (id: number, name: string) =>
  api.post(`/brains/${id}/snapshots/${name}/restore`);

export const deleteSnapshot = (id: number, name: string) =>
  api.delete(`/brains/${id}/snapshots/${name}`);

export const cowSnapshot = (id: number) =>
  api.post(`/brains/${id}/cow/snapshot`);

export const cowRestore = (id: number) =>
  api.post(`/brains/${id}/cow/restore`);
