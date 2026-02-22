import { useState, useEffect, useCallback, useMemo } from 'react';
import type { BrainInfo, BrainCreate, DatasetInfo, Tab, TrainingProgress } from './types';
import { Header } from './components/layout/Header';
import { Sidebar } from './components/layout/Sidebar';
import { TabNav } from './components/layout/TabNav';
import { BrainCreateModal } from './components/brains/BrainCreateModal';
import { BrainDetailModal } from './components/brains/BrainDetailModal';
import { SnapshotPanel } from './components/brains/SnapshotPanel';
import { DashboardPage } from './components/dashboard/DashboardPage';
import { TrainingPage } from './components/training/TrainingPage';
import { ChatPage } from './components/chat/ChatPage';
import { DatasetPage } from './components/datasets/DatasetPage';
import { LoginPage } from './components/auth/LoginPage';
import { RegisterPage } from './components/auth/RegisterPage';
import { useBrainProbe } from './hooks/useBrainProbe';
import api from './services/api';
import * as brainApi from './services/brainApi';
import { listDatasets } from './services/datasetApi';
import './App.css';

type AuthPage = 'login' | 'register';

function applyAuth(username: string, password: string) {
  const token = btoa(`${username}:${password}`);
  api.defaults.headers.common['Authorization'] = `Basic ${token}`;
}

function loadSavedAuth(): { username: string; password: string } | null {
  try {
    const saved = sessionStorage.getItem('nimcp_auth');
    if (saved) return JSON.parse(saved);
  } catch { /* ignore */ }
  // Also check URL credentials (existing behavior)
  const url = new URL(window.location.href);
  if (url.username) {
    return { username: decodeURIComponent(url.username), password: decodeURIComponent(url.password) };
  }
  return null;
}

export default function App() {
  const savedAuth = loadSavedAuth();
  const [authed, setAuthed] = useState(!!savedAuth);
  const [authPage, setAuthPage] = useState<AuthPage>('login');
  const [brains, setBrains] = useState<BrainInfo[]>([]);
  const [activeBrainId, setActiveBrainId] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>('dashboard');
  const [showCreate, setShowCreate] = useState(false);
  const [detailBrainId, setDetailBrainId] = useState<number | null>(null);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  // Apply saved credentials on mount
  useEffect(() => {
    if (savedAuth) {
      applyAuth(savedAuth.username, savedAuth.password);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLogin = useCallback((username: string, password: string) => {
    applyAuth(username, password);
    sessionStorage.setItem('nimcp_auth', JSON.stringify({ username, password }));
    setAuthed(true);
  }, []);

  const { probe, history, trainingProgress, connected, send, setChatCallback } =
    useBrainProbe(activeBrainId);

  const refreshBrains = useCallback(() => {
    brainApi.listBrains().then(r => setBrains(r.data)).catch(() => {});
  }, []);

  const refreshDatasets = useCallback(() => {
    listDatasets().then(r => setDatasets(r.data)).catch(() => {});
  }, []);

  useEffect(() => {
    refreshBrains();
    refreshDatasets();
    const id = setInterval(refreshBrains, 10000);
    return () => clearInterval(id);
  }, [refreshBrains, refreshDatasets]);

  // Derive training brain ID from WS progress messages
  const trainingBrainId = useMemo(() => {
    if (!trainingProgress) return null;
    const tp = trainingProgress as unknown as TrainingProgress & { type: string };
    if (tp.type === 'training_progress' && tp.running) return tp.brain_id;
    return null;
  }, [trainingProgress]);

  const handleCreate = async (data: BrainCreate) => {
    try {
      const r = await brainApi.createBrain(data);
      setShowCreate(false);
      refreshBrains();
      setActiveBrainId(r.data.id);
    } catch { /* */ }
  };

  const handleDelete = async (id: number) => {
    if (!window.confirm('Delete this brain? This cannot be undone.')) return;
    await brainApi.deleteBrain(id);
    if (activeBrainId === id) setActiveBrainId(null);
    if (detailBrainId === id) setDetailBrainId(null);
    refreshBrains();
  };

  const handleRename = async (id: number, name: string) => {
    try {
      await brainApi.renameBrain(id, name);
      refreshBrains();
    } catch { /* */ }
  };

  if (!authed) {
    if (authPage === 'register') {
      return (
        <RegisterPage
          onRegistered={(u, p) => { handleLogin(u, p); }}
          onSwitchToLogin={() => setAuthPage('login')}
        />
      );
    }
    return (
      <LoginPage
        onLogin={handleLogin}
        onSwitchToRegister={() => setAuthPage('register')}
      />
    );
  }

  return (
    <>
      <Header />
      <div className="app-layout">
        <Sidebar
          brains={brains}
          activeBrainId={activeBrainId}
          trainingBrainId={trainingBrainId}
          onSelect={setActiveBrainId}
          onDelete={handleDelete}
          onCreateClick={() => setShowCreate(true)}
          onRename={handleRename}
          onShowDetail={setDetailBrainId}
        />
        <div className="main-content">
          <TabNav active={tab} onChange={setTab} />
          <div className="tab-content">
            {tab === 'dashboard' && (
              <>
                <DashboardPage probe={probe} history={history} />
                {activeBrainId !== null && <SnapshotPanel brainId={activeBrainId} />}
              </>
            )}
            {tab === 'training' && (
              <TrainingPage brainId={activeBrainId} datasets={datasets} trainingProgress={trainingProgress} onRefresh={refreshDatasets} />
            )}
            {tab === 'chat' && (
              <ChatPage brainId={activeBrainId} send={send} connected={connected} setChatCallback={setChatCallback} />
            )}
            {tab === 'datasets' && (
              <DatasetPage datasets={datasets} onRefresh={refreshDatasets} />
            )}
          </div>
        </div>
      </div>
      {showCreate && (
        <BrainCreateModal onClose={() => setShowCreate(false)} onCreate={handleCreate} />
      )}
      {detailBrainId !== null && (
        <BrainDetailModal brainId={detailBrainId} onClose={() => setDetailBrainId(null)} />
      )}
    </>
  );
}
